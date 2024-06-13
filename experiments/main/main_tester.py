import re
import os
import shutil
import argparse
from time import time

import pandas as pd

import fasttext

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
    query
)

from tools.josiestuff.db import JosieDB
from tools.josiestuff.functions import *
from tools import embeddings



if __name__ == '__main__':
    # TODO ok argparse and CLI, maybe better a file .py with variables and import them? 
    # TODO LSH Forest analysis
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', 
                        type=str, required=True)
    parser.add_argument('-a', '--algorithm',
                        required=False, default='josie',
                        choices=['josie', 'lshforest', 'embedding'])
    parser.add_argument('-m', '--mode', 
                        required=False, default='set',
                        choices=['set', 'bag', 'fasttext'],
                        help='the specific version of the algorithm. Note that an algorithm doesn\'t support all the available modes: for example, \
                            if "algorithm"="embedding", the only accepted mode is "fasttext"')
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
    parser.add_argument('-u', '--user-interaction',
                        required=False, action='store_true',
                        help='specify if user should be asked to continue if necessary')
    parser.add_argument('--query-file', 
                        required=False, type=str, 
                        help='an absolute path to an existing file containing the queries which will be used for JOSIE tests')
    parser.add_argument('--small', 
                        required=False, action='store_true',
                        help='works on small collection versions (only for testing)')
    parser.add_argument('--clean', 
                        required=False, action='store_true', 
                        help='remove PostgreSQL database tables and other big files, such as the LSH Forest index file')


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
    user_interaction =  args.user_interaction
    nworkers =          min(os.cpu_count(), 64)

    # TODO set thresholds as a CLI parameter or somethig else?
    table_thresholds = {
        'min_row':     5,
        'min_column':  2,
        'min_area':     50,
        'max_row':     999999,
        'max_column':  999999,
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

    forest_dir =                ROOT_TEST_DIR + f'/lshforest' 
    forest_file =               forest_dir + f'/forest_m{mode}.json' if not args.forest_file else args.forest_file
    
    embeddings_dir =            ROOT_TEST_DIR + '/embeddings'
    clut_file =                 embeddings_dir + '/clut.json'
    cidx_file =                 embeddings_dir + '/cidx.index'

    results_base_dir =          ROOT_TEST_DIR + '/results/base'
    results_extr_dir =          ROOT_TEST_DIR + '/results/extracted'
    topk_results_file =         results_base_dir + f'/a{algorithm}_m{mode}_k{k}.csv'

    # statistics stuff
    statistics_dir =            ROOT_TEST_DIR  + '/statistics'
    runtime_stat_file =         statistics_dir + '/runtime.csv'     
    db_stat_file =              statistics_dir + '/db.csv'
    storage_stat_file =         statistics_dir + '/storage.csv'

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

    if algorithm == 'embedding':
        print('Loading fastText model...')
        start = time()
        model = fasttext.load_model(defpath.model_path.fasttext + '/cc.en.300.bin')
        print('loaded model in ', round(time() - start, 3), 's')

        clut, cidx = None, None

    ############# SET UP #############
    if any_task:
        if os.path.exists(ROOT_TEST_DIR) and user_interaction:
            if input(f'Directory {ROOT_TEST_DIR} already exists: delete it (old data will be lost)? (yes/no) ') in ('y', 'yes'):
                shutil.rmtree(ROOT_TEST_DIR)

        for directory in [ROOT_TEST_DIR, statistics_dir, results_base_dir, results_extr_dir, forest_dir, embeddings_dir]:
            if not os.path.exists(directory): 
                print(f'Creating directory {directory}...')
                os.makedirs(directory)
            

    ############# DATA PREPARATION #############
    if algorithm == 'josie' and (ALL or DATA_PREPARATION):
        josiedb.drop_tables()
        josiedb.create_tables()
        start = time()
        create_index(mode, table_thresholds, small, table_prefix)
        runtime_metrics.append(('data_preparation', round(time() - start, 5), get_current_time()))
        josiedb.create_inverted_list_index()
        josiedb.create_sets_index()
        
        # database statistics
        append = os.path.exists(db_stat_file)
        dbstat = pd.DataFrame(josiedb.get_statistics())
        dbstat.to_csv(db_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)


        def convert_to_giga(x):
            if x.endswith('MB'):
                return int(re.match(r'\d+', x).group()) / 1024
            elif x.endswith('KB'):
                return int(re.match(r'\d+', x).group()) / (1024 ** 2)

        append = os.path.exists(storage_stat_file)
        dbsize = pd.DataFrame([[algorithm, mode, dbstat['total_size'].apply(convert_to_giga).sum()]], columns=['algorithm', 'mode', 'size(GB)'])
        dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

    elif algorithm == 'lshforest' and (ALL or DATA_PREPARATION or QUERY):
        start = time()
        forest = get_or_create_forest(forest_file, nworkers, num_perm, l, mode, _mmh3_hashfunc, table_thresholds, *collections)
        runtime_metrics.append(('data_preparation', round(time() - start, 5), get_current_time()))

        forest_size_gb = os.path.getsize(forest_file) / (1024 ** 3)
        
        append = os.path.exists(storage_stat_file)
        dbsize = pd.DataFrame([[algorithm, mode, forest_size_gb]], columns=['algorithm', 'mode', 'size(GB)'])
        dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

    elif algorithm == 'embedding' and (ALL or DATA_PREPARATION):
        start = time()
        clut, cidx = embeddings.data_preparation(clut_file, cidx_file, model, table_thresholds, *collections)
        runtime_metrics.append(('data_preparation', round(time() - start, 5), get_current_time()))

        clut_size_gb = os.path.getsize(clut_file) / (1024 ** 3)
        cidx_file_gb = os.path.getsize(cidx_file) / (1024 ** 3)
        
        append = os.path.exists(storage_stat_file)
        storage = pd.DataFrame([[algorithm, mode, clut_size_gb + cidx_file_gb]], columns=['algorithm', 'mode', 'size(GB)'])
        storage.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)


    ############# SAMPLING TEST VALUES FOR JOSIE ##############
    if ALL or SAMPLE_QUERIES:
        sample_queries(query_file, nsamples, table_thresholds, *collections)


    ################## RUNNING JOSIE ##################
    if ALL or QUERY:
        start = time()
        query_ids = get_query_ids_from_query_file(query_file)
        if algorithm == 'josie':
            josiedb.clear_query_table()
            josiedb.insert_data_into_query_table(query_ids)

            josie_test(user_dbname, table_prefix, results_base_dir, topk_results_file, k)
        elif algorithm == 'lshforest':
            query(topk_results_file, forest, query_ids, mode, num_perm, k, _mmh3_hashfunc, *collections)
        elif algorithm == 'embedding':
            if not clut or not cidx:
                print('Loading LUT and index...')
                clut, cidx = embeddings.load_lut_index(clut_file, cidx_file)
            print('Querying...')
            embeddings.query(topk_results_file, model, clut, cidx, query_ids, k, *collections)

        runtime_metrics.append(('query', round(time() - start, 5), get_current_time()))

    if any_task:
        add_header = not os.path.exists(runtime_stat_file)
        with open(runtime_stat_file, 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,time\n")

            for (t_name, t_time, t_loctime) in runtime_metrics:
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{t_time}\n")

    if CLEAN:
        print('Cleaning directories and database...')
        josiedb.drop_tables(all=True)
        if os.path.exists(forest_file):
            os.remove(forest_file)
        print('Cleaning completed.')

    if josiedb:
        josiedb.close()
    elif mongoclient:
        mongoclient.close()

print('All tasks have been completed.')
