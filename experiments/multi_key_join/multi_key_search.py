import os
import re
import curses
import binascii
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from tools.josie import JosieDB
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.logging import logging_setup, info
from tools.utils.misc import is_valid_multi_key, is_valid_table, table_columns_to_rows, table_to_tokens, largest_overlap_sloth, table_rows_to_columns
from tools.utils.classes import ResultDatabase
from tools.utils.settings import DefaultPath as dp


def get_result_ids(s): 
    return list(map(int, re.findall(r'\d+', s)[::2])) if s else []
def get_result_overlaps(s): 
    return list(map(int, re.findall(r'\d+', s)[1::2])) if s else []
def parse_results(r): 
    return list(zip(get_result_ids(r), get_result_overlaps(r))) if str(r) != 'nan' else []




def josie_multi_query(queries:list[int:set[int]], k, results_file, dbname, tables_prefix) -> list[tuple[int, list[tuple[int, int]]]]:
    def create_query_table(queries:list[int:set[int]], dbname, tables_prefix):    
        josiedb.open()
        josiedb.clear_query_table()
        for table_id, tokens_ids in queries:
            josiedb._dbconn.execute(f"INSERT INTO {tables_prefix}_queries VALUES ({table_id}, ARRAY[{','.join(map(str, tokens_ids))}]);")
            
        josiedb._dbconn.commit()
        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        # if cost sampling tables already exist we assume they are correct and won't recreate them
        sample_costs_tables_exist = josiedb.cost_tables_exist()
        josiedb.close()

        if not sample_costs_tables_exist:
            os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                        --pg-database={dbname} \
                        --test_tag={tables_prefix} \
                        --pg-table-queries={tables_prefix}_queries')
    def query(results_file, k, results_directory, dbname, tables_prefix):
        # we are not considering the query preparation steps, since in some cases this will 
        # include also the cost sampling phase and in other cases it won't
        token_table_on_memory = False
        
        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        x = 'true' if token_table_on_memory else 'false'

        os.system(f'go run {josie_cmd_dir}/topk/main.go \
                    --pg-database={dbname} \
                    --test_tag={tables_prefix} \
                    --outputDir={results_directory} \
                    --resultsFile={results_file} \
                    --useMemTokenTable={x} \
                    --k={k} \
                    --verbose=false')
    results_directory = os.path.dirname(results_file)
    create_query_table(queries, dbname, tables_prefix)
    query(results_file, k, results_directory, dbname, tables_prefix)
    df = pd.read_csv(results_file)
    return [[row[0], parse_results(row[1])] for row in df[['query_id', 'results']].itertuples(index=False)]


def compute_table_largest_overlap(id1, id2, dlh:SimpleDataLakeHelper, resultsdb:ResultDatabase, **sloth_args):
    id1, id2 = (id1, id2) if id1 <= id2 else (id2, id1)
    if not (o := resultsdb.lookup(id1, id2)):
        tobj1 = dlh.get_table_by_numeric_id(id1)
        tobj2 = dlh.get_table_by_numeric_id(id2)
        o = largest_overlap_sloth(tobj1['content'], tobj2['content'], tobj1['numeric_columns'], tobj2['numeric_columns'], **sloth_args)[0]
        id1, id2 = (id1, id2) if id1 <= id2 else (id2, id1)
        resultsdb.insert_results([[id1, id2, o]])
    return o


def compute_columns_largest_overlap(rid, dlh:SimpleDataLakeHelper, query_cols, **sloth_args):
    # sim = lambda s, b: 1 - abs(s - b) / max(s, b)
    rtobj = dlh.get_table_by_numeric_id(rid)
    columns_largest_overlap = largest_overlap_sloth(table_columns_to_rows(query_cols), rtobj['content'], [0] * len(query_cols), rtobj['numeric_columns'], **sloth_args)[0]
    return columns_largest_overlap


def task_compute_overlaps(data):
    qid, res_list, query_cols, min_h, min_w, max_w = data
    dlh = SimpleDataLakeHelper(datalake_location, dataset, size)
    resultsdb = ResultDatabase(dbname, results_table)
    resultsdb.open()
    x = []
    for rank, (rid, bag_overlap) in tqdm(enumerate(res_list), leave=False, disable=False if os.getpid() % N == 0 else True):
        assert type(rid) == int and type(bag_overlap) == int
        # table_ov = compute_table_largest_overlap(qid, rid, dlh, resultsdb, min_w=min_w, min_h=min_h)
        columns_ov = compute_columns_largest_overlap(rid, dlh, query_cols, min_w=min_w, min_h=min_h)
        # if columns_ov != -1 and table_ov != -1:
            # x.append([rid, rank, bag_overlap, table_ov, columns_ov])
        x.append([rid, rank, bag_overlap, columns_ov])
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

dataset = 'gittables'
test_name, size, mode = 'main', 'standard', 'bag'
multi_key_join_directory =  f"{dp.root_project_path}/experiments/multi_key_join/{dataset}"

dbname = 'nanni'
tables_prefix = f'{test_name}_d{dataset}_m{mode}'
results_table = f'results_d{dataset}_s{size}'

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


match dataset:
    case "wikiturlsnap":
        datalake_location = 'mongodb'
        mapping_id_file = numeric_columns_file = None
        blacklist = [] 
        list_names = [
            ['party', 'district']
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
        raise ValueError(f"Unknown dataset {dataset}")


josiedb = JosieDB(dbname, tables_prefix)
dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)
ntables = dlh.get_number_of_tables()

logfile = f"{multi_key_join_directory}/N{N}_K{K}_T{min_h}.log"

list_names = list(map(lambda names: tuple(s.lower() for s in names), list_names)) # because the names will be used as key of a dict, and tuple are hashables...
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
                    if is_valid_multi_key(names, table, bad_columns, headers):
                        queries_for_name[names].append(table_obj)
                    
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
            fw.writelines(map(lambda x: str(x) + '\n', [q['_id_numeric'] for q in queries_for_name[names]]))


for i, names in enumerate(list_names):
    test_directory =            f"{multi_key_join_directory}/{'-'.join(names)}".replace(' ', '_')
    queries_file =              f"{test_directory}/queries.txt"
    test_directory +=           f"/N{N}_K{K}_T{min_h}"
    josie_single_results_file = f"{test_directory}/results_single_josie.csv"
    josie_multi_results_file =  f"{test_directory}/results_multi_josie.csv"
    final_results_file =        f"{test_directory}/final_results.csv"
    metrics_table_plot =        f"{test_directory}/metrics_table"
    metrics_columns_plot =      f"{test_directory}/metrics_columns"
    
    if SEARCH_QUERIES:
        queries = queries_for_name[names]
    else:
        with open(queries_file) as fr:
            queries = [dlh.get_table_by_numeric_id(int(qid)) for qid in fr.readlines()]
    
    if len(queries) < N * 0.5:
        print(f'Not enough queries for names {names}: only {len(queries)}')
        print()
    
    min_w, max_w = len(names), len(names)
    info(f' Working with names {names} '.center(60, '-'))
    info(f"N = {N}, K = {K}, min_h = {min_h}, min_w = {min_w}, max_w = {max_w}")
    info(f'Using {len(queries)}/{N} initial query tables')
    
    info('1. Take the query columns from the selected tables wrt the names we\'re currently working on')
    query_columns = defaultdict(list)
    for q in queries:
        table, bad_columns = q['content'], q['numeric_columns']
        headers = q['headers'] if 'headers' in q else None
        table = table_rows_to_columns(table, 0, len(table[0]), [0] * len(table[0]))
        idn = q['_id_numeric']
        for i, column in enumerate(table):
            if bad_columns[i] == 1:
                continue
            if not headers and any(token in column for token in names):
                query_columns[q['_id_numeric']].append(column)
            if headers and str(headers[i]).lower().strip() in names:
                query_columns[q['_id_numeric']].append(column)


    info('2. Apply JOSIE with single-column queries (baseline)')
    josiedb.open()
    single_column_bags = {qid: [table_to_tokens([column], mode, [0] * len(column), blacklist=blacklist) # the the column is considered as a row-view table
                                for column in query_columns[qid]] for qid in query_columns.keys()}


    info('\ta. Converting tokens to integer tokens for query sets...')
    queries_set = []
    for qid, qbag_list in tqdm(single_column_bags.items(), leave=False):
        result = josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
        for qbag in qbag_list:
            integer_tokens = set()
            for tid in result:
                raw_token = josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {tid}").fetchone()[0]
                if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                    integer_tokens.add(tid)
            queries_set.append([qid, integer_tokens])


    info('\tb. Compute JOSIE overlaps...')
    d = josie_multi_query(queries_set, K, josie_single_results_file, dbname, tables_prefix)
    josiedb.close()


    info('\tc. Merge distinct results for the same queries')
    single_column_results = defaultdict(lambda: defaultdict(list))
    for qid, josie_res in d:
        for jr in josie_res:
            single_column_results[qid][jr[0]].append(jr[1])

    for qid in single_column_results:
        for jr in single_column_results[qid]:
            single_column_results[qid][jr] = min(single_column_results[qid][jr]) * 2

    single_column_results = {qid: sorted([[rid, ub] for rid, ub in r.items()], key=lambda x: x[1], reverse=True)[:K] for qid, r in single_column_results.items()}
    

    info('\td. Compute largest overlap for single-column query results...')
    with mp.Pool() as pool:
        work = [(qid, single_column_results[qid], query_columns[qid], min_h, min_w, max_w) for qid in single_column_results.keys()]
        single_column_results = {qid: [r for r in res_list] for qid, res_list in pool.map(task_compute_overlaps, work, 1)}
    

    info('\te. Filtering queries where results have all the same largest overlap (no tail)')
    to_drop = set()
    for qid, res_list in single_column_results.items():
        top = res_list[0][1]
        if len(res_list) > K / 2 and len(set(x[0] for x in res_list[1:])) <= 1 or res_list[-1][-1] == top - 1:
            to_drop.add(qid)
    for qid in to_drop:
        del single_column_results[qid]
    info(f'\tDropped {len(queries) - len(single_column_results)} queries')


    info('3. Apply JOSIE with multi-column queries (our)')
    info('\ta. Create multi-column results (only for queries from the previous filtering)')
    josiedb.open()
    multi_column_bags = {qid: table_to_tokens(columns, 'bag', [0] * len(columns[0]), blacklist=blacklist) for qid, columns in query_columns.items() if qid in single_column_results.keys()}
    queries_set = [
        [qid, {
                id 
                for id in josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
                if binascii.unhexlify(
                    josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {id}").fetchone()[0]
                ).decode('utf-8') in qbag
            }
        ]
        for qid, qbag in multi_column_bags.items()
    ]


    info('\tb. Compute JOSIE overlaps...')
    multi_columns_results = dict(josie_multi_query(queries_set, K, josie_multi_results_file, dbname, tables_prefix))
    josiedb.close()


    info('\tc. Compute largest overlap for multi-column query results')
    with mp.Pool() as pool:
        work = [(qid, multi_columns_results[qid], query_columns[qid], min_h, min_w, max_w) for qid in multi_columns_results.keys()]
        multi_columns_results = {qid: [r for r in res_list] for qid, res_list in pool.map(task_compute_overlaps, work, 1)}
    
    # save the results from both the baseline and MC versions for future analyses
    # no filtering on queries with a non-null results set
    final_results = []
    final_results += [['baseline', qid, *r] for qid, res_list in single_column_results.items() for r in res_list]
    final_results += [['MC', qid, *r] for qid, res_list in multi_columns_results.items() for r in res_list]
    pd.DataFrame(final_results, columns=['version', 'query_id', 'result_id', 'result_rank', 'JOSIE_overlap', 'SLOTH_columns_overlap']).to_csv(final_results_file, index=False)
 
    info('')
