from functools import reduce
from itertools import chain
import os
import re
import sys
import curses
import binascii
import multiprocessing as mp
from collections import defaultdict
from time import time

import pandas as pd
from tqdm import tqdm

from sqlalchemy import create_engine, MetaData, select, insert, delete, inspect
from sqlalchemy.orm import Session


from tools.utils.parallel import chunks
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.logging import logging_setup, info
from tools.utils.misc import is_valid_multi_key, is_valid_table, table_columns_to_rows, table_to_tokens, largest_overlap_sloth, table_rows_to_columns
from tools.utils.settings import DefaultPath as dp
from tools.mate.MATE import MATETableExtraction


def get_result_ids(s): 
    return list(map(int, re.findall(r'\d+', s)[::2])) if s else []
def get_result_overlaps(s): 
    return list(map(int, re.findall(r'\d+', s)[1::2])) if s else []
def parse_results(r): 
    return list(zip(get_result_ids(r), get_result_overlaps(r))) if str(r) != 'nan' else []



def josie_multi_query(queries:list[int:set[int]], k, results_file, dbname, tables_prefix) -> list[tuple[int, list[tuple[int, int]]]]:
    def create_query_table(queries:list[int:set[int]], dbname, tables_prefix):    
        # josiedb.open()
        # josiedb.clear_query_table()
        with Session(josie_db_engine) as session:
            session.execute(delete(josie__queries_table))
            session.commit()

            for table_id, tokens_ids in queries:
                session.execute(insert(josie__queries_table).values(
                    id=table_id,
                    tokens=tokens_ids)
                )
            session.commit()
                # try:
                #     josiedb._dbconn.execute(f"INSERT INTO {tables_prefix}_queries VALUES ({table_id}, ARRAY[{','.join(map(str, tokens_ids))}]);")
                # except:
                #     print('error: ', table_id, ' ### ', tokens_ids)
                #     raise Exception()
            # josiedb._dbconn.commit()

        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        # if cost sampling tables already exist we assume they are correct and won't recreate them
        # sample_costs_tables_exist = josiedb.cost_tables_exist()

        # josiedb.close()
        # josie__read_set_cost_table = metadata.tables[f'{tables_prefix}_read_set_cost_samples']
        # josie__read_list_cost_table = metadata.tables[f'{tables_prefix}_read_list_cost_samples']

        sample_costs_tables_exist = (inspect(josie_db_engine).has_table(f'{tables_prefix}_read_set_cost_samples') \
                                     or inspect(josie_db_engine).has_table(f'{tables_prefix}_read_list_cost_samples'))

        if not sample_costs_tables_exist:
            os.system(f'''
                      go run {josie_cmd_dir}/sample_costs/main.go
                        --pg-database={dbname}
                        --test_tag={tables_prefix}
                        --pg-table-queries={tables_prefix}_queries''')
    def query(results_file, k, results_directory, dbname, tables_prefix):
        # we are not considering the query preparation steps, since in some cases this will 
        # include also the cost sampling phase and in other cases it won't
        token_table_on_memory = False
        
        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        x = 'true' if token_table_on_memory else 'false'

        os.system(f'''
                  go run {josie_cmd_dir}/topk/main.go
                    --pg-database={dbname}
                    --test_tag={tables_prefix}
                    --outputDir={results_directory}
                    --resultsFile={results_file}
                    --useMemTokenTable={x}
                    --k={k}
                    --verbose=false''')
    results_directory = os.path.dirname(results_file)
    create_query_table(queries, dbname, tables_prefix)
    query(results_file, k, results_directory, dbname, tables_prefix)
    df = pd.read_csv(results_file)
    return [[row[0], parse_results(row[1])] for row in df[['query_id', 'results']].itertuples(index=False)]


def compute_columns_largest_overlap(rid, dlh:SimpleDataLakeHelper, query_cols, **sloth_args):
    rtobj = dlh.get_table_by_numeric_id(rid)
    columns_largest_overlap = largest_overlap_sloth(table_columns_to_rows(query_cols), rtobj['content'], [0] * len(query_cols), rtobj['numeric_columns'], **sloth_args)[0]
    return columns_largest_overlap


def task_compute_overlaps(data):
    qid, res_list, query_cols, min_h, min_w, max_w = data
    dlh = SimpleDataLakeHelper(datalake_location, datalake, size)
    x = []
    for rank, rid  in tqdm(enumerate(res_list), leave=False, disable=False if os.getpid() % N == 0 else True):
        rid = int(rid)
        columns_ov = compute_columns_largest_overlap(rid, dlh, query_cols, min_w=min_w, min_h=min_h)
        x.append([rid, rank, columns_ov])
    return qid, x


# 1 calcolo con singole colonne (baseline)
# 2 cercare coppie in cui l'overlap SLOTH e vicino a quello di JOSIE
# 2.1 controllare il discorso tail: se con k=10 tutti i risultati sono ottimi, magari occorre prendere k=20 per avere una coda di valori meno buoni da confrontare
# 2.2 considerare le coppie di query-result con differenza tra bag intersection e largest overlap bassa
# 3 calcolo multi column sulle query filtrate dal passaggio precedente (versione nostra nuova)
# 4 verifica quale dei due va meglio

# forse ci sarebbe da indicare anche la frequenza effettiva totale di una certa combinazione di chiavi
# che ad esempio 'title' o 'author' sono molto comuni, e probabilmente anche le loro singole combinazioni
# ma invece sono molto meno comuni combinazioni più elaborate e con chiavi specifiche, 
# come 'model_id:id' o 'htid', e in questi casi si potrebbe avere un maggiore distacco tra la modalità
# a singola colonna e quella multi-column
# in verità, sembra che siccome chiavi come "htid" o "model_id:id" sono poco frequenti e con valori specifici,
# JOSIE becca facilmente le tabelle da cui provengono, quindi ha maggiori possibilità di inserirle nei risultati
# anche lavorando in single column...

# JOSIE può dare alcuni problemi, vedi la coppia di tabelle (59240, 1470444) per gli attributi "Club-Stadium"
# di fatto nella seconda tabella la colonna "club" è un po' sporca e così non viene matchato niente...

datalake = 'wikiturlsnap'
test_name, size, mode = 'main', 'standard', 'bag'
multi_key_join_directory =  f"{dp.root_project_path}/experiments/multi_key_join/{datalake}_mate"

dbname = 'nanni'
tables_prefix = f'{test_name}_d{datalake}_m{mode}'
results_table = f'results_d{datalake}_s{size}'

# search N queries, but many of them may not provide any results wrt the thresholds 
# min_h, min_w and thus won't be considered in analyses
N = 600
K = 50

# we want to find overlaps where the join covers at least this percentage of rows (see SLOTH paper)
min_h = 0.9

# these will be set accordingly to the list of names used as candidate join columns
min_w = None
max_w = None

SEARCH_QUERIES = False

# MATE parameters
one_bits = 5
hash_size = 128
dbname = 'nanni'
dataset_name = f'mate_{datalake}'
table_name = f'mate__wikiturlsnap_table_{hash_size}'
josie_db_connection_info = {
    'url': f'postgresql+psycopg2://localhost:5442/{dbname}',
    # 'url': f'duckdb:///{f"{dp.data_path.base}/mate_index.db"}',
    'connect_args': {
    #     'read_only': True
        # "server_settings": {"jit": "off"}
    }
}
mate_db_connection_info = {
    'url': f'postgresql+psycopg2://localhost:5442/{dbname}',
    'connect_args': {
        'options': "-c default_transaction_read_only=on"
    }
}

print(josie_db_connection_info)
print(mate_db_connection_info)

josie_db_engine = create_engine(**josie_db_connection_info)
metadata = MetaData(josie_db_engine)
metadata.reflect()

josie__set_table = metadata.tables[f'{tables_prefix}_sets']
josie__pl_table = metadata.tables[f'{tables_prefix}_inverted_lists']
josie__queries_table = metadata.tables[f'{tables_prefix}_queries']

# MATE allows to directly set a minimum ratio for the join, as a fraction of the height
# or as a number of tuples in common in the selected join attributes (right?)
min_join_ratio = 0


match datalake:
    case "wikiturlsnap":
        datalake_location = 'mongodb'
        mapping_id_file = numeric_columns_file = None
        blacklist = [] 
        list_names = [
            # ['manager', 'league'],
            # ['cast', 'title'],
            # ['cast', 'genre'],
            ['director', 'genre'],
            ['party', 'district'], ['party', 'member'], 
            # ['home', 'road'], ['home', 'div']
            # ['city', 'club'], ['club', 'stadium'], ['city', 'province'], ['city', 'country'],
            # ['country', 'director'],
            # ['conference', 'overall'],
            # ['actor', 'role'], ['director', 'genre'], ['country', 'director'], ['role', 'film'],
            # ['party', 'member'], ['party', 'term'], ['party', 'candidate'], ['party', 'incumbent', 'candidates'], ['party', 'votes', 'candidates'],
            # ['home', 'div'], ['home', 'road'], 
            # ['driver', 'car'],
            # ['name', 'hometown'], ['name', 'from'], ['name', 'description'], ['name', 'nation'], ['name', 'country'], ['name', 'nationality'],
        ]
    case "gittables":
        datalake_location = 'mongodb'
        mapping_id_file = numeric_columns_file = None
        blacklist = {"comment", "story", "{\"$numberDouble\": \"NaN\"}"}
        results_table += '_blacklist'
        list_names = [
            ['artist', 'lastname'],
            ['first name', 'last name'],
            ['firstname', 'lastname'],
            ['benchmarkname', 'layername'],
            ['assignee', 'component'],
            ['player_name', 'team_city'],
            ['team', 'last name'],
            ['last name', 'race category'],
            ['age', 'first name', 'last name']
        ]
    case "santoslarge":
        datalake_location = "/data4/nanni/data/santos_large/datalake"
        mapping_id_file = "/data4/nanni/data/santos_large/mapping_id.pickle"
        numeric_columns_file = "/data4/nanni/data/santos_large/numeric_columns.pickle"
        blacklist = []
        list_names = [
            ('department family', 'entity'),
            ("parent department", "professional/occupational group"),
            ("organisation", "professional/occupational group"),
            ("organisation", "parent department", "professional/occupational group")
        ]        
    case _:
        raise ValueError(f"Unknown dataset {datalake}")




dlh = SimpleDataLakeHelper(datalake_location, datalake, size, mapping_id_file, numeric_columns_file)
ntables = dlh.get_number_of_tables()

logfile = f"{multi_key_join_directory}/N{N}_K{K}_T{min_h}.log"

# the names will be used as key of a dict, and tuple are hashables
list_names = list(map(lambda names: tuple(s.lower() for s in names), list_names)) 
queries_for_name = {names: [] for names in list_names}


# create the directory for the results and setup the logging system
for i, names in enumerate(list_names):
    test_directory = f"{multi_key_join_directory}/{'-'.join(names)}/N{N}_K{K}_T{min_h}".replace(' ', '_')
    if not os.path.exists(test_directory): os.makedirs(test_directory)
    if i == 0: logging_setup(logfile)


# start the search: for each composite key, scann the tables and find the first N valid tables
# valid means that have at least 5 rows and 2 columns, not considering numeric or null ones
# and in case the attributes are in the headers they actual form a valid key (i.e. no FD)
if SEARCH_QUERIES:
    stdscr = curses.initscr()
    for i, table_obj in enumerate(dlh.scan_tables()):
        table = table_obj['content']
        bad_columns = table_obj['numeric_columns']
        headers = table_obj['headers']
        if is_valid_table(table, bad_columns):
            headers = [str(h).lower().strip() for h in headers]
            for names in list_names:
                if sum(token in headers for token in names) == len(names) and len(queries_for_name[names]) < N:
                    if headers_idx := is_valid_multi_key(names, table, bad_columns, headers):
                        queries_for_name[names].append([table_obj, headers_idx])
                    
        if len(queries_for_name) > 0 and all([len(queries) >= N for _, queries in queries_for_name.items()]):
            break
        if i % 100 == 0:
            stdscr.addstr(0, 0, f'Scanned {i} tables')        
            for j, (names, queries) in enumerate(queries_for_name.items()):
                stdscr.addstr(j+1, 0, f'{len(queries)}/{N} - {names}')
            stdscr.refresh()
    curses.endwin()
    
    print('Saving queries IDs...')
    for i, names in enumerate(list_names):
        queries_file = f"{multi_key_join_directory}/{'-'.join(names)}/queries.txt".replace(' ', '_')
        with open(queries_file, 'w') as fw:
            fw.writelines(map(lambda x: ' '.join(map(str, x)) + '\n', [[q['_id_numeric'], *hidx] for q, hidx in queries_for_name[names]]))


for i, names in enumerate(list_names):
    test_directory =            f"{multi_key_join_directory}/{'-'.join(names)}".replace(' ', '_')
    queries_file =              f"{test_directory}/queries.txt"
    test_directory +=           f"/N{N}_K{K}_T{min_h}"
    josie_single_results_file = f"{test_directory}/results_single_josie.csv"
    josie_multi_results_file =  f"{test_directory}/results_multi_josie.csv"
    final_results_file =        f"{test_directory}/final_results.csv"
    mate_time_file =            f"{test_directory}/mate_time.txt"
    
    if SEARCH_QUERIES:
        queries = queries_for_name[names][:N]
    else:
        with open(queries_file) as fr:
            queries = [list(map(int, line.strip().split(' '))) for line in fr.readlines()]
            queries = [[dlh.get_table_by_numeric_id(x[0]), x[1:]] for x in queries][:N]
    
    if len(queries) < N * 0.5:
        print(f'Not enough queries for names {names}: only {len(queries)}')
        print()
    
    min_w, max_w = len(names), len(names)
    info(f' Working with names {names} '.center(60, '-'))
    info(f"N = {N}, K = {K}, min_h = {min_h}, min_w = {min_w}, max_w = {max_w}")
    info(f'Using {len(queries)}/{N} initial query tables')
    
    info('1. Take the query columns from the selected tables wrt the names we\'re currently working on')
    query_columns = defaultdict(list)
    for qtobj, hidx in queries:
        table, bad_columns = qtobj['content'], qtobj['numeric_columns']
        headers = qtobj['headers'] if 'headers' in qtobj else None
        table = table_rows_to_columns(table, 0, len(table[0]), [0] * len(table[0]))
        for i in hidx:
            query_columns[qtobj['_id_numeric']].append(table[i])


    info('2. Apply JOSIE with single-column queries (baseline)')
    single_column_bags = {qid: [table_to_tokens([column], mode, [0] * len(column), blacklist=blacklist) # the column is considered as a row-view table
                                for column in query_columns[qid]] for qid in query_columns.keys()}


    info('\ta. Converting tokens to integer tokens for query sets...')
    queries_set = []
    with Session(josie_db_engine) as session:
        for qid, qbag_list in tqdm(single_column_bags.items(), leave=False):
            result = list(chain(*session.execute(select(josie__set_table.c.tokens).where(josie__set_table.c.id == qid))))[0]
            for qbag in qbag_list:
                integer_tokens = set()
                for tid in result:
                    raw_token = list(session.execute(select(josie__pl_table.c.raw_token).where(josie__pl_table.c.token == tid)))[0][0]
                    
                    if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                        integer_tokens.add(tid)
                queries_set.append([qid, integer_tokens])


    info('\tb. Compute JOSIE overlaps...')
    d = josie_multi_query(queries_set, K, josie_single_results_file, dbname, tables_prefix)


    info('\tc. Merge distinct results for the same queries')
    single_column_results = defaultdict(lambda: defaultdict(list))
    for qid, josie_res in d:
        for jr in josie_res:
            single_column_results[qid][jr[0]].append(jr[1])

    for qid in single_column_results:
        for jr in single_column_results[qid]:
            single_column_results[qid][jr] = min(single_column_results[qid][jr]) * 2

    # take only the top-K results and drop the overlap computed with JOSIE, which won't be used in next steps
    single_column_results = {qid: sorted(list(r.items()), key=lambda x: x[1], reverse=True)[:K] for qid, r in single_column_results.items()}
    single_column_results = {qid: [rid for rid, _ in r] for qid, r in single_column_results.items()}


    info('\td. Compute largest overlap for single-column query results...')
    with mp.Pool() as pool:
        work = [(qid, single_column_results[qid], query_columns[qid], min_h, min_w, max_w) for qid in single_column_results.keys()]
        single_column_results = dict(pool.map(task_compute_overlaps, work))
    

    #info('\te. Filtering queries where results have all the same largest overlap (no tail)')
    #to_drop = set()
    #for qid, res_list in single_column_results.items():
    #    top = res_list[0][1]
    #    if len(res_list) > K / 2 and len(set(x[0] for x in res_list[1:])) <= 1 or res_list[-1][-1] == top - 1:
    #        to_drop.add(qid)
    #for qid in to_drop:
    #    del single_column_results[qid]
    #info(f'\tDropped {len(queries) - len(single_column_results)} queries')


    info('3. Apply JOSIE with multi-column queries (our)')
    info('\ta. Create multi-column results')
    multi_column_bags = {qid: table_to_tokens(columns, 'bag', [0] * len(columns[0]), blacklist=blacklist) for qid, columns in query_columns.items() if qid in single_column_results.keys()}
    with Session(josie_db_engine) as session:
        queries_set = [
            [qid, {
                    tid 
                    for tid in list(chain(*session.execute(select(josie__set_table.c.tokens).where(josie__set_table.c.id == qid))))[0]
                    if binascii.unhexlify(
                        list(session.execute(select(josie__pl_table.c.raw_token).where(josie__pl_table.c.token == tid)))[0][0]
                    ).decode('utf-8') in qbag
                }
            ]
            for qid, qbag in multi_column_bags.items()
        ]


    info('\tb. Compute JOSIE overlaps...')
    multi_columns_results = dict(josie_multi_query(queries_set, K, josie_multi_results_file, dbname, tables_prefix))


    info('\tc. Compute largest overlap for multi-column query results')
    with mp.Pool() as pool:
        work = [(qid, [r[0] for r in multi_columns_results[qid]], query_columns[qid], min_h, min_w, max_w) for qid in multi_columns_results.keys()]
        multi_columns_results = dict(pool.map(task_compute_overlaps, work))


    def task_mate(data):
        results = {}
        timestat = []
        mate = MATETableExtraction(dataset_name, None, 
                                   K, table_name, ones=one_bits, 
                                   min_join_ratio=min_join_ratio,
                                   is_min_join_ratio_absolute=False, 
                                   **mate_db_connection_info)
        
        for qid, qcolumns in data[0]:
            query_dataset = pd.DataFrame(data=qcolumns).T
            start = time()
            mate_results = mate.MATE(hash_size, query_dataset, query_dataset.columns)
            end = time()
        
            results[qid] = [r[1] for r in mate_results if r[1] != qid]
            timestat.append(round(end - start, 3))
        
        return timestat, results
    
    def update_dict(x, y):
        x.update(y)
        return x
    
    info('4. Apply MATE')
    info(f'\ta. Search candidates with MATE len(work)={len(list(query_columns.items()))}...')
    processes = 2
    with mp.Pool(processes) as pool:
        work = list(query_columns.items())
        
        start_total_mate_time = time()
        timing, mate_results = list(zip(*pool.map(task_mate, chunks(work, max(len(work) // processes, 1)), 1)))
        total_mate_time = round(time() - start_total_mate_time, 5)
        
        mate_results = reduce(update_dict, mate_results)
        
        with open(mate_time_file, 'a') as fa:
            fa.writelines(map(lambda t: str(t) + '\n', timing))
            fa.write(str(total_mate_time) + '\n')

    
    info('\tb. Compute largest overlap for MATE results...')
    with mp.Pool() as pool:
        work = [(qid, mate_results[qid], query_columns[qid], min_h, min_w, max_w) for qid in mate_results.keys()]
        mate_results = dict(pool.map(task_compute_overlaps, work))


    # save the results from both the baseline and MC versions for future analyses
    # no filtering on queries with a non-null results set
    final_results = []
    final_results += [['BSL', qid, *r] for qid, res_list in single_column_results.items() for r in res_list]
    final_results += [['MC', qid, *r] for qid, res_list in multi_columns_results.items() for r in res_list]
    final_results += [['MATE', qid, *r] for qid, res_list in mate_results.items() for r in res_list]

    pd.DataFrame(final_results, columns=['version', 'query_id', 'result_id', 'result_rank', 'columns_overlap']).to_csv(final_results_file, index=False)
 
    info('')
