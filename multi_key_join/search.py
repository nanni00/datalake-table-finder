import os
import curses
from time import time
from functools import reduce
from itertools import groupby, chain
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from dltf.gsa.josie.josie import JOSIEGS
from dltf.utils.datalake import DataLakeHandlerFactory
from dltf.utils.misc import chunks
from dltf.utils.loghandler import logging_setup, info
from dltf.utils.settings import DefaultPath as dp
from dltf.mate.MATE import MATETableExtraction
from dltf.utils.misc import largest_overlap_sloth
from dltf.utils.tables import is_valid_multi_key, is_valid_table, table_columns_to_rows, table_rows_to_columns


def task_mate(data):
    global k, dataset, db_connection_info, is_absolute_rate, min_join_ratio, one_bits, hash_size, string_blacklist, string_translators, string_patterns
    results = {}
    timestat = []
    
    # Init MATE algorithm
    mate = MATETableExtraction(dataset_name=dataset, 
                               mate_cache_path=None, 
                               t_k=k, 
                               ones=one_bits, 
                               min_join_ratio=min_join_ratio,
                               is_min_join_ratio_absolute=is_absolute_rate,
                               string_blacklist=string_patterns,
                               string_translators=string_translators,
                               string_patterns=string_patterns,
                               db_connection_info=db_connection_info)
    
    # Create a table from each sets of columns
    for qid, qcolumns in tqdm(data[0], disable=os.getpid() % 60 != 0, leave=False):
        query_dataset = pd.DataFrame(data=list(zip(*qcolumns)))
        start = time()
        mate_results = mate.MATE(hash_size, True, query_dataset, query_dataset.columns)
        duration = round(time() - start, 3)
        results[qid] = [r[1] for r in mate_results if r[1] != qid]
        timestat.append(duration)
    return timestat, results


def initializer_mate(_k, _dataset, _db_connection_info, _is_absolute_rate, _min_join_ratio, _one_bits, _hash_size, _string_blacklist, _string_translators, _string_patterns):
    global db_connection_info, is_absolute_rate, min_join_ratio, dataset, k, one_bits, hash_size, string_blacklist, string_translators, string_patterns
    k                       = _k
    dataset                 = _dataset
    db_connection_info      = _db_connection_info
    is_absolute_rate        = _is_absolute_rate
    min_join_ratio          = _min_join_ratio
    one_bits                = _one_bits 
    hash_size               = _hash_size
    string_blacklist        = _string_blacklist
    string_translators      = _string_translators
    string_patterns         = _string_patterns


def task_compute_overlaps(data):
    global dlhconfig
    qid, res_list, query_cols, min_h_values, min_w, max_w = data
    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
    x = []
    for rank, rid  in enumerate(res_list):
        rid = int(rid)
        columns_ov = []
        for min_h in min_h_values:
            rtobj = dlh.get_table_by_numeric_id(rid)
            overlap, _ = largest_overlap_sloth(
                table_columns_to_rows(query_cols), 
                rtobj['content'], 
                [1] * len(query_cols), 
                rtobj['valid_columns'], 
                min_w=min_w, 
                min_h=min_h,
                verbose=False
            )
            columns_ov.append(overlap)
        x.append([rid, rank, *columns_ov])
    return qid, x


def initializer_overlaps(_dlhconfig):
    global dlhconfig
    dlhconfig = _dlhconfig



def multi_key_search(dlhconfig,
                     num_cpu, 
                     db_connection_info,
                     string_blacklist,
                     string_translators,
                     string_patterns,
                     strict_valid_multi_key,

                     # query parameters
                     search_queries,
                     list_names, 
                     num_queries,
                     k, 
                     min_h_values, 

                     # JOSIE parameters
                     mode, 
                     token_bidict_path,

                     # MATE parameters 
                     hash_size, 
                     one_bits, 
                     min_join_ratio, 
                     is_absolute_rate):
    
    multi_key_join_directory =  f"{os.path.dirname(__file__)}/results/{dlhconfig[1]}_mate"

    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)

    josie = JOSIEGS(mode=mode, 
                    datalake_handler=dlh, 
                    string_blacklist=string_blacklist, 
                    string_translators=string_translators, 
                    string_patterns=string_patterns,
                    dbstatfile=None, 
                    tokens_bidict_file=token_bidict_path,
                    josie_db_connection_info=db_connection_info,
                    spark_config=None)
    
    josie.db.load_tables()

    # mate_engine = create_engine(URL.create(**db_connection_info))
    mate_inv_idx_table_name = 'mate_inv_idx'

    logfile = f"{multi_key_join_directory}/N{num_queries}_K{k}.log"

    # the names will be used as key of a dict, and tuple are hashables
    list_names = list(map(lambda names: tuple(s.lower() for s in names), list_names)) 
    queries_for_name = {names: [] for names in list_names}

    # create the directory for the results and setup the logging system
    for i, names in enumerate(list_names):
        names_test_directory = f"{multi_key_join_directory}/{'-'.join(names)}/N{num_queries}_K{k}".replace(' ', '_')
        if not os.path.exists(names_test_directory): os.makedirs(names_test_directory)
        if i == 0: logging_setup()


    # start the search: for each composite key, scann the tables and find the first N valid tables
    # valid means that have at least 5 rows and 2 columns, not considering numeric or null ones
    # and in case the attributes are in the headers they actual form a valid key (i.e. no FD)
    if search_queries:
        # stdscr = curses.initscr()
        for i, table_obj in enumerate(dlh.scan_tables()):
            table, headers, valid_columns = table_obj['content'], table_obj['headers'], table_obj['valid_columns']
            # for wikitables there could be num_header_rows>=2, 
            # but for simplicity we use only those tables with num_header_rows==1
            if not table_obj['headers'] or isinstance(table_obj['headers'][0], list) and len(table_obj['headers']) > 1:
                continue
            if is_valid_table(table, valid_columns):
                headers = [str(h).lower().strip() for h in headers[0]]
                for names in list_names:
                    if sum(token in headers for token in names) == len(names) and len(queries_for_name[names]) < num_queries:
                        if strict_valid_multi_key and (headers_idx := is_valid_multi_key(names, table, valid_columns, headers[0])) \
                            or not strict_valid_multi_key and (headers_idx := [i for i, token in enumerate(headers) if token in names and token not in headers[:i]]):
                            queries_for_name[names].append([table_obj, headers_idx])
                        
            if len(queries_for_name) > 0 and all([len(queries) >= num_queries for _, queries in queries_for_name.items()]):
                break
        #     if i % 100 == 0:
        #         stdscr.addstr(0, 0, f'Scastr_blacklistnned {i} tables')        
        #         for j, (names, queries) in enumerate(queries_for_name.items()):
        #             stdscr.addstr(j+1, 0, f'{len(queries)}/{num_queries} - {names}')
        #         stdscr.refresh()
        # curses.endwin()
        
        print('Saving queries IDs...')
        for i, names in enumerate(list_names):
            queries_file = f"{multi_key_join_directory}/{'-'.join(names)}/queries.txt".replace(' ', '_')
            with open(queries_file, 'w') as fw:
                fw.writelines(map(lambda x: ' '.join(map(str, x)) + '\n', [[q['_id_numeric'], *hidx] for q, hidx in queries_for_name[names]]))


    # For each list of names (i.e., columns names), execute the search
    # with both JOSIE and MATE
    for i, names in enumerate(list_names):
        names_test_directory        = f"{multi_key_join_directory}/{'-'.join(names)}".replace(' ', '_')
        queries_file                = f"{names_test_directory}/queries.txt"
        final_results_file          = f"{names_test_directory}/N{num_queries}_K{k}/final_results.csv"
        mate_time_file              = f"{names_test_directory}/N{num_queries}_K{k}/mate_time.txt"
        
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
        
        # For each query table, we already have its query columns 
        # (those involved into the selected multi key)
        # but we need distinct IDs
        single_queries = {
            f'{qid}_{i}': [qcolumn]
            for qid, qcolumns in query_columns.items()
            for i, qcolumn in enumerate(qcolumns)
        }

        info('\t- Compute JOSIE overlaps...')
        _, josie_results = josie.query(single_queries, k)
        
        # Get only the true query ID, removing the suffix used previously
        josie_results = sorted([[int(qid.split('_')[0]), results] for qid, results in josie_results])
        
        # Group by the query ID
        josie_results = groupby(josie_results, key=lambda g: g[0])
        
        # For each group relative to a query, group all the results relative to a single result ID
        # and take the minimum overlap found, then multiply it for the number of columns required
        single_josie_results = {}
        for r in josie_results:
            qid, res = r[0], list(chain(*[rr[1] for rr in r[1]]))
            min_values = defaultdict(list)
            for rid, ov in res:
                if rid == qid:
                    continue
                min_values[rid].append(ov)
            # sort the results
            single_josie_results[qid] = sorted([(rid, min(overlaps) * len(names)) for rid, overlaps in min_values.items()], key=lambda x: x[1], reverse=True)
            # take only the first k result IDs, the overlap won't be used next
            single_josie_results[qid] = [rid for rid, _ in single_josie_results[qid]][:k]

        info('\t- Compute largest overlap for single-column candidates')
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, single_josie_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in single_josie_results.keys()]
            single_josie_results = dict(pool.map(task_compute_overlaps, work))
        


        info('3. Apply JOSIE with multi-column queries')
        multi_queries = {
            qid: qcolumns
            for qid, qcolumns in query_columns.items()
        }
        
        info('\t- Compute JOSIE overlaps...')
        _, josie_results = josie.query(multi_queries, k)
        multi_josie_results = {qid: [x[0] for x in sorted(res, key=lambda y: y[1], reverse=True)][:k] for qid, res in josie_results}
        
        info('\t- Compute largest overlap for multi-column candidates')
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, multi_josie_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in multi_josie_results.keys()]
            multi_josie_results = dict(pool.map(task_compute_overlaps, work))


        info('4. Apply MATE')
        info(f'\t- Search candidates with MATE, len(work)={len(list(query_columns.items()))}...')
        initargs = (k, dlhconfig[1], db_connection_info, is_absolute_rate, min_join_ratio, one_bits, hash_size, string_blacklist, string_translators, string_patterns)
        with mp.get_context('spawn').Pool(num_cpu, initializer_mate, initargs) as pool:
            work = list(query_columns.items())
            
            start_total_mate_time = time()
            timing, mate_results = list(zip(*pool.map(task_mate, chunks(work, max(len(work) // num_cpu, 1)), 1)))
            total_mate_time = round(time() - start_total_mate_time, 5)
            mate_results = reduce(lambda x, y: x | y, mate_results)        
        
            with open(mate_time_file, 'a') as fa:
                fa.writelines(map(lambda t: str(t) + '\n', timing))
                fa.write(str(total_mate_time) + '\n')
        
        info('\t- Compute largest overlap for MATE candidates')
        with mp.get_context('spawn').Pool(num_cpu, initializer_overlaps, (dlhconfig, )) as pool:
            work = [(qid, mate_results[qid], query_columns[qid], min_h_values, min_w, max_w) for qid in mate_results.keys()]
            mate_results = dict(pool.map(task_compute_overlaps, work))

        # save the results from both the baseline and MC versions for future analyses
        # no filtering on queries with a non-null results set
        final_results = []
        final_results += [['BSL', qid, *r] for qid, res_list in single_josie_results.items() for r in res_list]
        final_results += [['MC', qid, *r] for qid, res_list in multi_josie_results.items() for r in res_list]
        final_results += [['MATE', qid, *r] for qid, res_list in mate_results.items() for r in res_list]

        pd.DataFrame(final_results, columns=['version', 'query_id', 'result_id', 'result_rank'] + [f'overlap_{min_h}' for min_h in min_h_values]) \
            .to_csv(final_results_file, index=False)
        info('')


def main_demo():
    db_connection_info = {
        'drivername': 'postgresql',
        'database'  : 'DEMODB', 
        'host'      : 'localhost',
        'port'      : 5442,
        'username'  : 'demo',
        'password'  : 'demo'
    }

    list_names = [
        ['country'],
        ['location', 'name']
    ]

    multi_key_search(
        # general parameters
        dlhconfig=['mongodb', 'demo', ['sloth.demo']],
        num_cpu=10,
        db_connection_info=db_connection_info,
        string_blacklist=set(),
        string_translators=['whitespace', 'lowercase', ['"', ' ']],
        string_patterns=[],
        strict_valid_multi_key=False,
        
        # query parameters        
        search_queries=True,
        list_names=list_names,
        num_queries=100,
        k=10,
        min_h_values=[0.0, 0.4],
        
        # JOSIE parameters
        mode='bag',
        token_bidict_path=f'{dp.data_path.tests}/demo/demo/josie/tokens_bidict_file_bag.pickle',

        # MATE parameters
        hash_size=128,
        one_bits=5,
        min_join_ratio=0,
        is_absolute_rate=True,
    )



if __name__ == '__main__':
    main_demo()


