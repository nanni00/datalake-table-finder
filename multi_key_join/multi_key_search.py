import os
import pickle
import curses
from time import time
from functools import reduce
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from dltf.testers.josie.josie import JOSIETester
from dltf.utils.datalake import DataLakeHandler, DataLakeHandlerFactory
from dltf.utils.misc import chunks
from dltf.utils.loghandler import logging_setup, info
from dltf.utils.settings import DefaultPath as dp
from dltf.mate.MATE import MATETableExtraction
from dltf.utils.misc import largest_overlap_sloth, lowercase_translator, whitespace_translator
from dltf.utils.tables import is_valid_multi_key, is_valid_table, table_columns_to_rows, table_rows_to_columns, table_to_tokens


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


def compute_columns_largest_overlap(rid, dlh:DataLakeHandler, query_cols, **sloth_args):
    rtobj = dlh.get_table_by_numeric_id(rid)
    columns_largest_overlap = largest_overlap_sloth(table_columns_to_rows(query_cols), rtobj['content'], [1] * len(query_cols), rtobj['valid_columns'], **sloth_args)[0]
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



def multi_key_search(dlhconfig,
                     num_cpu, 
                     blacklist,
                     db_connection_info,

                     # query parameters
                     search_queries,
                     list_names, 
                     num_queries,
                     k, 
                     min_h_values, 

                     # JOSIE parameters
                     mode, 
                     token_table_on_memory,
                     token_bidict_path,

                     # MATE parameters 
                     hash_size, 
                     one_bits, 
                     min_join_ratio, 
                     is_absolute_rate):
    multi_key_join_directory =  f"{dp.root_project_path}/experiments/multi_key_join/{dlhconfig[1]}_mate"

    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
    
    josie = JOSIETester(mode=mode, 
                        blacklist=blacklist, 
                        datalake_handler=dlh, 
                        string_translators=[lowercase_translator, whitespace_translator], 
                        dbstatfile=None, 
                        spark_config=None,
                        josie_db_connection_info=db_connection_info)
    josie.db.load_tables()

    mate_engine = create_engine(URL.create(**db_connection_info)) 

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
    if search_queries:
        stdscr = curses.initscr()
        for i, table_obj in enumerate(dlh.scan_tables()):
            table, headers, valid_columns = table_obj['content'], table_obj['headers'], table_obj['valid_columns']
            if is_valid_table(table, valid_columns):
                headers = [str(h).lower().strip() for h in headers]
                for names in list_names:
                    if sum(token in headers for token in names) == len(names) and len(queries_for_name[names]) < num_queries:
                        if headers_idx := is_valid_multi_key(names, table, valid_columns, headers):
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
        
        if search_queries:
            queries = queries_for_name[names]
        else:
            with open(queries_file) as fr:
                queries = [list(map(int, line.strip().split(' '))) for line in fr.readlines()]
                queries = [[dlh.get_table_by_numeric_id(x[0]), x[1:]] for x in queries]

        min_w, max_w = len(names), len(names)

        info(f' Working with names {names} '.center(60, '-'))
        info(f"{num_queries=}, {k=}, {min_h_values=}, {min_w=}, {max_w=}, {min_join_ratio=}")
        info(f'Using {len(queries)}/{num_queries} initial query tables')
        

        info('1. Extract the query columns')
        query_columns = defaultdict(list)
        for qtobj, hidx in queries:
            table, valid_columns = qtobj['content'], qtobj['valid_columns']
            headers = qtobj['headers'] if 'headers' in qtobj else None
            table = table_rows_to_columns(table, 0, len(table[0]), [1] * len(table[0]))
            for i in hidx:
                query_columns[qtobj['_id_numeric']].append(table[i])

        info('2. Apply JOSIE with single-column queries (baseline)')
        single_column_bags = {qid: [table_to_tokens([column], [1] * len(column), mode, blacklist=blacklist) # the column is considered as a row-view table
                                    for column in query_columns[qid]] for qid in query_columns.keys()}

        info('\ta. Converting tokens to integer tokens for query sets...')
        queries_set = []
        with open(token_bidict_path, 'rb') as fr:
            tokens_bidict = pickle.load(fr)

        str_tokens = [[qid, [[tokens_bidict[token] for token in qbag] for qbag in qbag_list]] for qid, qbag_list in single_column_bags.items()]

        queries_set = [
            [qid, {tid for tid, str_token in str_tokens if str_token in qbag}]
            for qid, qbag_list in single_column_bags.items() for qbag in qbag_list
        ]

        print(len(queries_set))

        # TODO JOSIE when update the query table doesn't allow duplicates (set_id is Primary Key)
        # thus we need to add some more flexible way to run the query
        info('\tb. Compute JOSIE overlaps...')
        d = josie.query(josie_single_results_file, k, [q[0] for q in queries_set], token_table_on_memory=token_table_on_memory, results_directory=test_directory)
        d = pd.read_csv(josie_single_results_file)[['query_id', 'result_ids']].itertuples(index=False)
        
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
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, single_column_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in single_column_results.keys()]
            single_column_results = dict(pool.map(task_compute_overlaps, work))
        

        info('3. Apply JOSIE with multi-column queries (our)')
        info('\ta. Converting tokens to integer tokens for query sets...')
        multi_column_bags = {qid: table_to_tokens(columns, [0] * len(columns[0]), 'bag', blacklist=blacklist) for qid, columns in query_columns.items()}
        str_tokens = [[qid, [[tokens_bidict[token] for token in qbag] for qbag in qbag_list]] for qid, qbag_list in multi_column_bags.items()]
        
        queries_set = [
            [qid, {tid for tid, str_token in str_tokens if str_token in qbag_list}]
            for qid, qbag_list in multi_column_bags.items()
        ]


        info('\tb. Compute JOSIE overlaps...')
        d = josie.query(josie_multi_results_file, k, [q[0] for q in queries_set], token_table_on_memory=token_table_on_memory, results_directory=test_directory)
        multi_columns_results = dict(pd.read_csv(josie_single_results_file)[['query_id', 'result_ids']].itertuples(index=False))


        info('\tc. Compute largest overlap for multi-column candidates')
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, [r[0] for r in multi_columns_results[qid]], query_columns[qid], min_h_values, min_w, max_w) for qid in multi_columns_results.keys()]
            multi_columns_results = dict(pool.map(task_compute_overlaps, work))


        info('4. Apply MATE')
        info(f'\ta. Search candidates with MATE, len(work)={len(list(query_columns.items()))}...')
        initargs = (mate_engine, is_absolute_rate, min_join_ratio, dataset, k, table_name, one_bits, hash_size)
        with mp.get_context('spawn').Pool(num_cpu, initializer_mate, initargs) as pool:
            work = list(query_columns.items())
            
            start_total_mate_time = time()
            timing, mate_results = list(zip(*pool.map(task_mate, chunks(work, max(len(work) // num_cpu, 1)), 1)))
            total_mate_time = round(time() - start_total_mate_time, 5)
            mate_results = reduce(lambda x, y: x | y, mate_results)        
        
            with open(mate_time_file, 'a') as fa:
                fa.writelines(map(lambda t: str(t) + '\n', timing))
                fa.write(str(total_mate_time) + '\n')
        
        
        info('\tb. Compute largest overlap for MATE candidates')
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
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



def main_gittables():

    # JOSIE parameters
    db_connection_info = {
        'drivername': 'postgresql',
        'database': 'DLTFGitTables', 
        'host': 'localhost',
        'port': 5442
    }

    blacklist = {"comment", "story", "{\"$numberDouble\": \"NaN\"}", "{\"$numberdouble\": \"nan\"}"}
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


    multi_key_search(
        # general parameters
        num_cpu=64,
        blacklist=blacklist,
        dlhconfig=['mongodb', 'gittables', ['sloth.gittables']],
        
        # query parameters
        k=10,
        num_queries=30,
        search_queries=False,
        list_names=list_names,
        min_h_values=[0.6, 0.9],
        
        # JOSIE parameters
        mode='bag',
        token_table_on_memory=False,

        # MATE parameters
        one_bits=5,
        hash_size=128,
        min_join_ratio=0,
        is_absolute_rate=True,
        db_connection_info=db_connection_info
    )



if __name__ == '__main__':
    main_gittables()


