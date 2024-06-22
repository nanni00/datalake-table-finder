import os
import sys
import shutil
import argparse

import pandas as pd
from numerize_denumerize.numerize import numerize

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import (
    get_local_time,
    get_mongodb_collections, 
    get_query_ids_from_query_file, 
    sample_queries
)

from tools import josie, lshforest #, neo4j_graph, embeddings



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', 
                        type=str, required=True)
    parser.add_argument('-a', '--algorithm',
                        required=False, default='josie',
                        choices=['josie', 'lshforest', 'embedding', 'graph'])
    parser.add_argument('-m', '--mode', 
                        required=False, default='set',
                        choices=['set', 'bag', 'fasttext', 'neo4j'],
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
    parser.add_argument('--sample-costs',
                        required=False, action='store_true')
    parser.add_argument('--token-table-on-memory',
                        required=False, action='store_true')

    # LSH Forest specific arguments
    parser.add_argument('--forest-file', 
                        required=False, type=str, 
                        help='the location of the LSH Forest index file that will be used for querying. If it does not exist, \
                            a new index will be created at that location.')
    parser.add_argument('--num-perm', 
                        required=False, type=int, default=128,
                        help='number of permutations to use for minhashing')
    parser.add_argument('-l', 
                        required=False, type=int, default=8,
                        help='number of prefix trees (see datasketch.LSHForest documentation)')
    
    # Neo4j graph specific arguments
    parser.add_argument('--neo4j-user', 
                        required=False, type=str, default='neo4j')
    parser.add_argument('--neo4j-password', 
                        required=False, type=str, default='12345678')

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

    # JOSIE
    user_dbname =       args.dbname
    sample_costs =      args.sample_costs
    toktable_on_mem =  args.token_table_on_memory

    # LSHForest
    num_perm =          args.num_perm
    l =                 args.l
    
    small =             args.small
    nsamples =          args.num_query_samples
    user_interaction =  args.user_interaction
    neo4j_user =        args.neo4j_user
    neo4j_passwd =      args.neo4j_password
    num_cpu =           min(os.cpu_count(), 64)


    # check configuration
    if (algorithm, mode) not in {
        ('josie', 'set'),       
        ('josie', 'bag'), 
        ('lshforest', 'set'),       
        ('lshforest', 'bag'),
        ('embedding', 'fasttext'), 
        ('graph', 'neo4j')
        }:
        sys.exit(0)

    # filtering only those tables that have very few cells (<10)
    tables_thresholds = {
        'min_row':      5,
        'min_column':   2,
        'min_area':     0,
        'max_row':      999999,
        'max_column':   999999,
        'max_area':     999999,
    }

    DATA_PREPARATION =      'data-preparation' in tasks or 'all' in tasks
    SAMPLE_QUERIES =        'sample-queries' in tasks or 'all' in tasks
    QUERY =                 'query' in tasks or 'all' in tasks

    CLEAN =             args.clean

    # output files and directories
    ROOT_TEST_DIR =             defpath.data_path.tests + f'/{test_name}'
    query_file =                ROOT_TEST_DIR + f'/query_{numerize(nsamples, asint=True)}.json' if not args.query_file else args.query_file

    forest_dir =                ROOT_TEST_DIR + f'/lshforest' 
    forest_file =               forest_dir + f'/forest_m{mode}.json' if not args.forest_file else args.forest_file
    
    embeddings_dir =            ROOT_TEST_DIR + '/embeddings'
    clut_file =                 embeddings_dir + '/clut.json'
    cidx_file =                 embeddings_dir + '/cidx.index'

    # results stuff
    results_base_dir =          ROOT_TEST_DIR + '/results/base'
    results_extr_dir =          ROOT_TEST_DIR + '/results/extracted'
    topk_results_file =         results_base_dir + f'/a{algorithm}_m{mode}_k{k}_q{numerize(nsamples, asint=True)}.csv'

    # statistics stuff
    statistics_dir =            ROOT_TEST_DIR  + '/statistics'
    runtime_stat_file =         statistics_dir + '/runtime.csv'     
    db_stat_file =              statistics_dir + '/db.csv'
    storage_stat_file =         statistics_dir + '/storage.csv'

    runtime_metrics = []

    mongoclient, collections = get_mongodb_collections(small)

    table_prefix = f'{test_name}_m{mode}'

    tester = None

    # selecting the right tester accordingly to the specified algorithm and mode
    if algorithm == 'josie':
        tester = josie.JOSIETester(mode, small, tables_thresholds, num_cpu, user_dbname, table_prefix, db_stat_file)
    elif algorithm == 'lshforest':
        tester = lshforest.LSHForestTester(mode, small, tables_thresholds, num_cpu, forest_file, num_perm, l, collections)
    # elif algorithm == 'embedding':
    #     tester = embeddings.EmbeddingTester(mode, small, tables_thresholds, num_cpu, defpath.model_path.fasttext + '/cc.en.300.bin', clut_file, cidx_file)
    # elif algorithm == 'graph':
    #     if mode == 'neo4j':
    #         tester = neo4j_graph.Neo4jTester(mode, small, tables_thresholds, num_cpu, neo4j_user, neo4j_passwd, os.environ["NEO4J_HOME"] + "/data/databases/neo4j/", collections)


    if DATA_PREPARATION or QUERY or SAMPLE_QUERIES:
        if os.path.exists(ROOT_TEST_DIR) and user_interaction:
            if input(f'Directory {ROOT_TEST_DIR} already exists: delete it (old data will be lost)? (yes/no) ') in ('y', 'yes'):
                shutil.rmtree(ROOT_TEST_DIR)

        for directory in [ROOT_TEST_DIR, statistics_dir, results_base_dir, results_extr_dir, forest_dir, embeddings_dir]:
            if not os.path.exists(directory): 
                print(f'Creating directory {directory}...')
                os.makedirs(directory)
            
        
    if DATA_PREPARATION:
        exec_time, storage_size = tester.data_preparation()

        runtime_metrics.append(('data_preparation', exec_time, get_local_time()))

        append = os.path.exists(storage_stat_file)
        dbsize = pd.DataFrame([[algorithm, mode, storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
        dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

            
    if SAMPLE_QUERIES:
        num_samples = sample_queries(query_file, nsamples, tables_thresholds, *collections)
        print(f'Sampled {num_samples} query tables.')


    if QUERY:
        query_ids = get_query_ids_from_query_file(query_file)
        exec_time = tester.query(topk_results_file, k, query_ids, results_directory=results_base_dir, sample_costs=sample_costs, token_table_on_memory=toktable_on_mem)
        runtime_metrics.append((f'query_{numerize(len(query_ids), asint=True)}', exec_time, get_local_time()))


    if DATA_PREPARATION or QUERY or SAMPLE_QUERIES:
        add_header = not os.path.exists(runtime_stat_file)
        with open(runtime_stat_file, 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,time\n")

            for (t_name, t_time, t_loctime) in runtime_metrics:
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{t_time}\n")

    if CLEAN:
        tester.clean()

    if mongoclient:
        mongoclient.close()

print('All tasks have been completed.')
