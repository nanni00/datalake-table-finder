import os
import shutil
import argparse
from time import time

import pandas as pd
import pymongo

from tools.utils.settings import DefaultPath as defpath

from tools.utils.utils import (
    get_current_time,
    get_mongodb_collections, 
    get_query_ids_from_query_file, 
    sample_queries
)

from tools.lshforest import (
    _mmh3_hashfunc, 
    get_or_create_forest, 
    query_lsh_forest
)

from tools.josiestuff.db import JosieDB
from tools.josiestuff.functions import *



if __name__ == '__main__':
    # TODO ok argparse and CLI, maybe better a file .py with variables and import them? 
    # TODO LSH Forest analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', 
                        type=str, required=True)
    parser.add_argument('-a', '--algorithm',
                        required=False, default='josie',
                        choices=['josie', 'lshforest'])
    parser.add_argument('-m', '--mode', 
                        required=False, default='set',
                        choices=['set', 'bag'])
    parser.add_argument('-k', 
                        type=int, required=False, default=5,
                        help='the K value for the top-K search')
    parser.add_argument('-t', '--tasks', 
                        required=False, nargs='+',
                        choices=['all', 
                                'data-preparation',
                                'sample-queries', 
                                'query'
                                ], 
                        help='the tasks to do')
    parser.add_argument('--num-query-samples', 
                        type=int, required=False, default=100,
                        help='the number of tables that will be sampled from the collections and that will be used as query id for JOSIE (the actual number) \
                            may be less than the specified one due to thresholds tables parameter')
    
    # JOSIE specific arguments
    parser.add_argument('-d', '--dbname', 
                        required=False, default='user',
                        help='the PostgreSQL database where will be uploaded the data used by JOSIE. It must be already running on the machine')

    # LSH Forest specific arguments
    parser.add_argument('--forest-file', 
                        required=False, type=str, 
                        help='the location of the LSH Forest index file that will be used for querying. If it does not exist, \
                            a new index will be created at that location.')
    parser.add_argument('--num-perm', 
                        required=False, type=int, default=256,
                        help='number of permutations to use for minhashing')
    parser.add_argument('-l', 
                        required=False, type=int, default=8,
                        help='number of prefix trees (see datasketch.LSHForest documentation)')
    
    # other general arguments
    parser.add_argument('--query-file', 
                        required=False, type=str, 
                        help='an absolute path to an existing file containing the queries which will be used for JOSIE tests')
    parser.add_argument('--small', 
                        required=False, action='store_true',
                        help='works on small collection versions (only for testing)')
    parser.add_argument('--clean', 
                        required=False, action='store_true', 
                        help='remove PostgreSQL database tables and other big files')


    args = parser.parse_args()
    test_name =         args.test_name
    algorithm =         args.algorithm
    mode =              args.mode
    tasks =             args.tasks if args.tasks else []
    k =                 args.k
    user_dbname =       args.dbname
    num_perm =          args.num_perm
    l =                 args.l
    small =             args.small
    nsamples =          args.num_query_samples

    # TODO set thresholds as a CLI parameter or somethig else?
    tables_thresholds = {
        'min_rows':     5,
        'min_columns':  2,
        'min_area':     20,
        'max_rows':     999999,
        'max_columns':  999999,
        'max_area':     999999,
    }

    ALL =                   'all' in tasks
    DATA_PREPARATION =      'data-preparation' in tasks
    SAMPLE_QUERIES =        'sample-queries' in tasks
    QUERY =                 'query' in tasks

    any_task = bool(args.tasks)

    CLEAN =             args.clean


    # output files
    ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
    query_file =                ROOT_TEST_DIR + '/query.json' if not args.query_file else args.query_file
    forest_file =               ROOT_TEST_DIR + f'/forest_m{mode}.json' if not args.forest_file else args.forest_file
    results_dir =               ROOT_TEST_DIR + '/results'
    topk_results_file =         results_dir + f'/a{algorithm}_m{mode}_k{k}.csv'

    # statistics stuff
    statistics_dir =            ROOT_TEST_DIR  + '/statistics'
    runtime_stat_file =         statistics_dir + '/runtime.csv'     
    db_stat_file =              statistics_dir + '/db.csv'

    runtime_metrics = []

    # the MongoDB collections where data are stored
    mongoclient, collections = get_mongodb_collections(small)

    forest = None
    josiedb = None
    table_prefix = f'{test_name}_m{mode}'

    # JOSIE database handler
    if algorithm == 'josie' or CLEAN:
        josiedb = JosieDB(user_dbname, table_prefix)
        josiedb.open()


    ############# SET UP #############
    if any_task:
        if os.path.exists(ROOT_TEST_DIR):
            if input(f'Directory {ROOT_TEST_DIR} already exists: delete it (old data will be lost)? (yes/no) ') in ('y', 'yes'):
                shutil.rmtree(ROOT_TEST_DIR)

        if not os.path.exists(ROOT_TEST_DIR): 
            print(f'Creating test directory {ROOT_TEST_DIR}...')
            os.makedirs(ROOT_TEST_DIR)
            print(f'Creating test statistics directory {statistics_dir}...')
            os.makedirs(statistics_dir)
            print(f'Creating results statistics directory {results_dir}...')
            os.makedirs(results_dir)


    ############# DATA PREPARATION #############
    if algorithm == 'josie' and (ALL or DATA_PREPARATION):
        josiedb.drop_tables()
        josiedb.create_tables()

        start = time()
        create_index(mode, tables_thresholds, small, table_prefix)
        runtime_metrics.append(('data_preparation', round(time() - start, 5), get_current_time()))
        
        josiedb.create_inverted_list_index()
        josiedb.create_sets_index()
        
        # database statistics
        append = os.path.exists(db_stat_file)
        pd.DataFrame(josiedb.get_statistics()).to_csv(db_stat_file, index=False, 
                                                        mode='a' if append else 'w', header=False if append else True)

    elif algorithm == 'lshforest' and (ALL or DATA_PREPARATION or QUERY):
        start = time()
        forest = get_or_create_forest(forest_file, num_perm, l, mode, _mmh3_hashfunc, tables_thresholds, *collections)
        runtime_metrics.append(('data_preparation', round(time() - start, 5), get_current_time()))    


    ############# SAMPLING TEST VALUES FOR JOSIE ##############
    if ALL or SAMPLE_QUERIES:
        sample_queries(query_file, nsamples, tables_thresholds, *collections)


    ################## RUNNING JOSIE ##################
    if ALL or QUERY:
        start = time()
        sampled_ids = get_query_ids_from_query_file(query_file)
        if algorithm == 'josie':
            josiedb.clear_query_table()
            josiedb.insert_data_into_query_table(sampled_ids)

            josie_test(user_dbname, table_prefix, results_dir, topk_results_file, k)
        elif algorithm == 'lshforest':
            query_lsh_forest(topk_results_file, forest, sampled_ids, mode, num_perm, k, _mmh3_hashfunc, *collections)
        runtime_metrics.append(('query', round(time() - start, 5), get_current_time()))

    if any_task:
        add_header = not os.path.exists(runtime_stat_file)
        with open(runtime_stat_file, 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,time\n")

            for (t_name, t_time, t_loctime) in runtime_metrics:
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{t_time}\n")
        print('All tasks have been completed.')


    if CLEAN:
        print('Cleaning directories and database...')
        josiedb.drop_tables(all=True)
        print('Cleaning completed.')

    if josiedb:
        josiedb.close()
    elif mongoclient:
        mongoclient.close()
