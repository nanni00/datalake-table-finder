import os
import re
import curses
import binascii
from time import time
from itertools import chain
from functools import reduce
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from sqlalchemy import select, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.engine import URL

from dltftools.testers.josie.db import JOSIEDBHandler
from dltftools.testers.josie import josie
from dltftools.utils.datalake import DataLakeHandler, DataLakeHandlerFactory
from dltftools.utils.parallel import chunks
from dltftools.utils.loghandler import logging_setup, info
from dltftools.utils.settings import DefaultPath as dp
from dltftools.mate.MATE import MATETableExtraction
from dltftools.utils.misc import largest_overlap_sloth
from dltftools.utils.tables import is_valid_multi_key, is_valid_table, table_columns_to_rows, table_rows_to_columns, table_to_tokens


def get_result_ids(s): 
    return list(map(int, re.findall(r'\d+', s)[::2])) if s else []
def get_result_overlaps(s): 
    return list(map(int, re.findall(r'\d+', s)[1::2])) if s else []
def parse_results(r): 
    return list(zip(get_result_ids(r), get_result_overlaps(r))) if str(r) != 'nan' else []



def josie_query(queries:list[int,set[int]], k, results_file, dbname, tables_prefix) -> list[tuple[int, list[tuple[int, int]]]]:
    def update_query_table(queries:list[int,set[int]], dbname, tables_prefix):
        josiedb.clear_query_table()
        table_ids = [q[0] for q in queries]
        tokens = [q[1] for q in queries]
        josiedb.add_queries(table_ids, tokens)

        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        # if cost sampling tables already exist we assume they are correct and won't recreate them
        sample_costs_tables_exist = josiedb.exist_cost_tables()

        if not sample_costs_tables_exist:
            print('Sampling cost...')
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
    update_query_table(queries, dbname, tables_prefix)
    query(results_file, k, results_directory, dbname, tables_prefix)
    df = pd.read_csv(results_file)
    return [[row[0], parse_results(row[1])] for row in df[['query_id', 'results']].itertuples(index=False)]


def task_mate(data):
    global mate_engine, is_absolute_rate, min_join_ratio, dataset, k, table_name, one_bits, hash_size
    results = {}
    timestat = []
    mate = MATETableExtraction(dataset, None, 
                                k, table_name, ones=one_bits, 
                                min_join_ratio=min_join_ratio,
                                is_min_join_ratio_absolute=is_absolute_rate, 
                                engine=mate_engine)
    
    for qid, qcolumns in tqdm(data[0], disable=os.getpid() % 60 != 0, leave=False):
        query_dataset = pd.DataFrame(data=list(zip(*qcolumns)))
        start = time()
        mate_results = mate.MATE(hash_size, True, query_dataset, query_dataset.columns)
        duration = round(time() - start, 3)
        results[qid] = [r[1] for r in mate_results if r[1] != qid]
        timestat.append(duration)
    return timestat, results


def initializer_mate(_mate_engine, _is_absolute_rate, _min_join_ratio, _dataset, _k, _table_name, _one_bits, _hash_size):
    global mate_engine, is_absolute_rate, min_join_ratio, dataset, k, table_name, one_bits, hash_size
    mate_engine = _mate_engine
    is_absolute_rate = _is_absolute_rate
    min_join_ratio = _min_join_ratio
    dataset = _dataset
    k = _k
    table_name = _table_name
    one_bits = _one_bits 
    hash_size = _hash_size


def update_dict(x, y):
    x.update(y)
    return x


def compute_columns_largest_overlap(rid, dlh:DataLakeHandler, query_cols, **sloth_args):
    rtobj = dlh.get_table_by_numeric_id(rid)
    columns_largest_overlap = largest_overlap_sloth(table_columns_to_rows(query_cols), rtobj['content'], [1] * len(query_cols), rtobj['numeric_columns'], **sloth_args)[0]
    return columns_largest_overlap


def task_compute_overlaps(data):
    global dlhconfig
    qid, res_list, query_cols, min_h_values, min_w, max_w = data
    dlh = DataLakeHandlerFactory.create_handler(dlhconfig)
    x = []
    for rank, rid  in enumerate(res_list):
        rid = int(rid)
        columns_ov = []
        for min_h in min_h_values:
            columns_ov.append(compute_columns_largest_overlap(rid, dlh, query_cols, min_w=min_w, min_h=min_h))
        x.append([rid, rank, *columns_ov])
    return qid, x


def initializer_overlaps(_dlhconfig):
    global dlhconfig
    dlhconfig = _dlhconfig



def multi_key_search(test_name, mode, dbname, num_queries, k, min_h_values, dlhconfig, num_cpu):
    multi_key_join_directory =  f"{dp.root_project_path}/experiments/multi_key_join/{dlhconfig[1]}_mate_final"

    # these will be set accordingly to the list of names used as candidate join columns
    min_w = None
    max_w = None

    SEARCH_QUERIES = True

    # JOSIE parameters
    tables_prefix = f'{test_name}_d{dlhconfig[1]}_m{mode}'
    results_table = f'overlaps'

    josie_db_connection_info = {
        'drivername': 'postgresql',
        'database': dbname, 
        'host': 'localhost',
        'port': 5442
    }

    # MATE parameters
    one_bits = 5
    hash_size = 128
    dataset_name = f'mate_{dlhconfig[1]}'
    table_name = f'mate__wikiturlsnap_table_{hash_size}'
    # MATE allows to directly set a minimum ratio for the join, as a fraction of the height
    # or as a number of tuples in common in the selected join attributes (right?)
    min_join_ratio = 0
    is_absolute_rate = True

    mate_db_connection_info = josie_db_connection_info

    josiedb = JOSIEDBHandler(tables_prefix, **josie_db_connection_info)
    mate_engine = create_engine(URL.create(**mate_db_connection_info)) 

    josie__set_table = josiedb.metadata.tables[f'{tables_prefix}_sets']
    josie__pl_table =  josiedb.metadata.tables[f'{tables_prefix}_inverted_lists']


    match dlhconfig[1]:
        case "wikiturlsnap":
            blacklist = [] 
            list_names = [
                ['party', 'member'], 
                ['director', 'genre'],
                ['party', 'district'], 
            ]
        case "gittables":
            blacklist = {"comment", "story", "{\"$numberDouble\": \"NaN\"}"}
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


    dlh = DataLakeHandlerFactory.create_handler(dlhconfig)
    ntables = dlh.count_tables()

    logfile = f"{multi_key_join_directory}/N{num_queries}_K{k}.log"

    # the names will be used as key of a dict, and tuple are hashables
    list_names = list(map(lambda names: tuple(s.lower() for s in names), list_names)) 
    queries_for_name = {names: [] for names in list_names}


    # create the directory for the results and setup the logging system
    for i, names in enumerate(list_names):
        test_directory = f"{multi_key_join_directory}/{'-'.join(names)}/N{num_queries}_K{k}".replace(' ', '_')
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
                    if sum(token in headers for token in names) == len(names) and len(queries_for_name[names]) < num_queries:
                        if headers_idx := is_valid_multi_key(names, table, bad_columns, headers):
                            queries_for_name[names].append([table_obj, headers_idx])
                        
            if len(queries_for_name) > 0 and all([len(queries) >= num_queries for _, queries in queries_for_name.items()]):
                break
            if i % 100 == 0:
                stdscr.addstr(0, 0, f'Scanned {i} tables')        
                for j, (names, queries) in enumerate(queries_for_name.items()):
                    stdscr.addstr(j+1, 0, f'{len(queries)}/{num_queries} - {names}')
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
        test_directory +=           f"/N{num_queries}_K{k}"
        josie_single_results_file = f"{test_directory}/results_single_josie.csv"
        josie_multi_results_file =  f"{test_directory}/results_multi_josie.csv"
        final_results_file =        f"{test_directory}/final_results.csv"
        mate_time_file =            f"{test_directory}/mate_time.txt"
        
        if SEARCH_QUERIES:
            queries = queries_for_name[names]
        else:
            with open(queries_file) as fr:
                queries = [list(map(int, line.strip().split(' '))) for line in fr.readlines()]
                queries = [[dlh.get_table_by_numeric_id(x[0]), x[1:]] for x in queries]

        min_w, max_w = len(names), len(names)

        info(f' Working with names {names} '.center(60, '-'))
        info(f"N = {num_queries}, K = {k}, min_h = {min_h_values}, min_w = {min_w}, max_w = {max_w}, MATE_min_join_ratio = {min_join_ratio}")
        info(f'Using {len(queries)}/{num_queries} initial query tables')
        

        info('1. Extract the query columns')
        query_columns = defaultdict(list)
        for qtobj, hidx in queries:
            table, bad_columns = qtobj['content'], qtobj['numeric_columns']
            headers = qtobj['headers'] if 'headers' in qtobj else None
            table = table_rows_to_columns(table, 0, len(table[0]), [1] * len(table[0]))
            for i in hidx:
                query_columns[qtobj['_id_numeric']].append(table[i])

        info('2. Apply JOSIE with single-column queries (baseline)')
        single_column_bags = {qid: [table_to_tokens([column], [1] * len(column), mode, blacklist=blacklist) # the column is considered as a row-view table
                                    for column in query_columns[qid]] for qid in query_columns.keys()}

        info('\ta. Converting tokens to integer tokens for query sets...')
        queries_set = []
        with Session(josiedb.engine) as session:
            tableids_tokens = list(session.execute(select(josie__set_table.c.id, josie__set_table.c.tokens).where(josie__set_table.c.id.in_(single_column_bags.keys()))))
            all_token_ids = set(chain(*[row[1] for row in tableids_tokens]))
            all_raw_tokens = session.execute(select(josie__pl_table.c.token, josie__pl_table.c.raw_token).where(josie__pl_table.c.token.in_(all_token_ids)))
            str_tokens = [
                [row[0], row[1], binascii.unhexlify(row[1]).decode('utf-8')]
                for row in all_raw_tokens
            ]
        
        for qid, qbag_list in single_column_bags.items():
            for qbag in qbag_list:
                integer_tokens = set()
                for (tid, raw_token, str_token) in str_tokens:
                    if str_token in qbag:
                        integer_tokens.add(tid)
                queries_set.append([qid, integer_tokens])


        info('\tb. Compute JOSIE overlaps...')
        d = josie_query(queries_set, k, josie_single_results_file, dbname, tables_prefix)


        info('\tc. Merge distinct results for the same queries')
        single_column_results = defaultdict(lambda: defaultdict(list))
        for qid, josie_res in d:
            for jr in josie_res:
                single_column_results[qid][jr[0]].append(jr[1])

        for qid in single_column_results:
            for jr in single_column_results[qid]:
                single_column_results[qid][jr] = min(single_column_results[qid][jr]) * 2

        # take only the top-K results and drop the overlap computed with JOSIE, which won't be used in next steps
        single_column_results = {qid: sorted(list(r.items()), key=lambda x: x[1], reverse=True)[:k] for qid, r in single_column_results.items()}
        single_column_results = {qid: [rid for rid, _ in r] for qid, r in single_column_results.items()}


        info('\td. Compute largest overlap for single-column candidates')
        with mp.Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, single_column_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in single_column_results.keys()]
            single_column_results = dict(pool.map(task_compute_overlaps, work))
        

        info('3. Apply JOSIE with multi-column queries (our)')
        info('\ta. Converting tokens to integer tokens for query sets...')
        multi_column_bags = {qid: table_to_tokens(columns, [0] * len(columns[0]), 'bag', blacklist=blacklist) for qid, columns in query_columns.items()}
        
        with Session(josiedb.engine) as session:
            tableids_tokens = list(session.execute(select(josie__set_table.c.id, josie__set_table.c.tokens).where(josie__set_table.c.id.in_(multi_column_bags.keys()))))
            all_token_ids = set(chain(*[row[1] for row in tableids_tokens]))
            all_raw_tokens = session.execute(select(josie__pl_table.c.token, josie__pl_table.c.raw_token).where(josie__pl_table.c.token.in_(all_token_ids)))
            str_tokens = [
                [row[0], row[1], binascii.unhexlify(row[1]).decode('utf-8')]
                for row in all_raw_tokens
            ]
        
        queries_set = []
        for qid, qbag_list in multi_column_bags.items():
            integer_tokens = set()
            for (tid, _, str_token) in str_tokens:
                if str_token in qbag_list:
                    integer_tokens.add(tid)
            queries_set.append([qid, integer_tokens])


        info('\tb. Compute JOSIE overlaps...')
        multi_columns_results = dict(josie_query(queries_set, k, josie_multi_results_file, dbname, tables_prefix))


        info('\tc. Compute largest overlap for multi-column candidates')
        with mp.Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, [r[0] for r in multi_columns_results[qid]], query_columns[qid], min_h_values, min_w, max_w) for qid in multi_columns_results.keys()]
            multi_columns_results = dict(pool.map(task_compute_overlaps, work))


        info('4. Apply MATE')
        info(f'\ta. Search candidates with MATE len(work)={len(list(query_columns.items()))}...')
        initargs = (mate_engine, is_absolute_rate, min_join_ratio, dataset, k, table_name, one_bits, hash_size)
        with mp.Pool(num_cpu, initializer_mate, initargs) as pool:
            work = list(query_columns.items())
            
            start_total_mate_time = time()
            timing, mate_results = list(zip(*pool.map(task_mate, chunks(work, max(len(work) // num_cpu, 1)), 1)))
            total_mate_time = round(time() - start_total_mate_time, 5)
        
            mate_results = reduce(update_dict, mate_results)        
            with open(mate_time_file, 'a') as fa:
                fa.writelines(map(lambda t: str(t) + '\n', timing))
                fa.write(str(total_mate_time) + '\n')
        
        
        info('\tb. Compute largest overlap for MATE candidates')
        with mp.Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, mate_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in mate_results.keys()]
            mate_results = dict(pool.map(task_compute_overlaps, work))
        mate_engine.dispose()


        # save the results from both the baseline and MC versions for future analyses
        # no filtering on queries with a non-null results set
        final_results = []
        final_results += [['BSL', qid, *r] for qid, res_list in single_column_results.items() for r in res_list]
        final_results += [['MC', qid, *r] for qid, res_list in multi_columns_results.items() for r in res_list]
        final_results += [['MATE', qid, *r] for qid, res_list in mate_results.items() for r in res_list]

        pd.DataFrame(final_results, columns=['version', 'query_id', 'result_id', 'result_rank'] + [f'overlap_{min_h}' for min_h in min_h_values]) \
            .to_csv(final_results_file, index=False)
        info('')


def main_wikitables():
    multi_key_search(
        test_name='base',
        mode='bag',
        dbname='DLTFWikiTables',
        num_queries=100,
        k=20,
        min_h_values=[0.6, 0.9],
        dlhconfig=['mongodb', 'wikitables', ['datasets.wikitables']],
        num_cpu=9
    )


if __name__ == '__main__':
    main_wikitables()


