
import logging
import os
import sys

import pandas as pd
from numerize_denumerize.numerize import numerize

from tools.utils.settings import DefaultPath as defpath, get_all_paths, make_parser
from tools.utils.basicconfig import tables_thresholds
from tools.utils.misc import (
    get_local_time,
    get_query_ids_from_query_file, 
    sample_queries,
    logging_setup
)
from tools.utils.mongodb_utils import get_mongodb_collections
from tools.utils import basicconfig
from tools import josie, lshforest, embedding



if __name__ == '__main__':
    args = make_parser('test_name', 'algorithm', 'mode', 'k', 'tasks', 'num_query_samples', 'num_cpu', 'size', 'dataset', 
                       'clean', 
                       'dbname', 'token_table_on_memory', 
                       'forest_file', 'num_perm', 'l',
                       'fasttext_model_size')

    test_name =         args.test_name
    algorithm =         args.algorithm
    mode =              args.mode
    tasks =             args.tasks if args.tasks else []
    k =                 args.k
    num_query_samples = args.num_query_samples
    num_cpu =           args.num_cpu
    size =              args.size
    dataset =           args.dataset

    # JOSIE
    user_dbname =       args.dbname
    token_table_on_mem =args.token_table_on_memory

    # LSHForest
    num_perm =          args.num_perm
    l =                 args.l

    # fastText
    fasttext_model_size = args.fasttext_model_size

    # check configuration
    if (algorithm, mode) not in basicconfig.algmodeconfig:
        sys.exit(1)

    test_name = test_name.lower()
    num_query_samples = int(num_query_samples)
    str_num_query_samples = numerize(num_query_samples, asint=True)
    
    # tasks to complete in current run
    DATA_PREPARATION =          'data-preparation' in tasks or 'all' in tasks
    SAMPLE_QUERIES =            'sample-queries' in tasks or 'all' in tasks
    QUERY =                     'query' in tasks or 'all' in tasks
    CLEAN =                     args.clean


    TEST_DATASET_DIR, query_file, logfile, \
        forest_dir, embedding_dir, \
            results_base_dir, results_extr_dir, \
                statistics_dir, runtime_stat_file, storage_stat_file = get_all_paths(test_name, dataset, k, str_num_query_samples)

    forest_file =       f'{forest_dir}/forest_m{mode}.json' if not args.forest_file else args.forest_file
    cidx_file =         f'{embedding_dir}/col_idx_mft.index' if mode in ['ft', 'ftdist'] else f'{embedding_dir}/col_idx_m{mode}.index' 
    topk_results_file = f'{results_base_dir}/a{algorithm}_m{mode}.csv'
    db_stat_file =      statistics_dir + '/db.csv'


    # a set of tokens that will be discarded when working on a specific dataset
    # 'comment' and 'story' are very frequent in GitTables, should they be removed? 
    blacklist = {'{"$numberDouble": "NaN"}', 'comment', 'story'} if dataset == 'gittables' else set()
    # blacklist = {'{"$numberDouble": "NaN"}'} if dataset == 'gittables' else set()

    # a list containing information about timing of each step
    runtime_metrics = []

    # the MongoDB collections where initial tables are stored
    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=size)

    # the prefix used in the PostgreSQL database tables (mainly for JOSIE)
    table_prefix = f'{test_name}_d{dataset}_m{mode}'

    # selecting the right tester accordingly to the specified algorithm and mode
    tester = None
    default_args = (mode, dataset, size, tables_thresholds, num_cpu, blacklist)
    match algorithm:
        case 'josie':
            tester = josie.JOSIETester(*default_args, user_dbname, table_prefix, db_stat_file)
        case 'lshforest':
            tester = lshforest.LSHForestTester(*default_args, forest_file, num_perm, l, collections)
        case 'embedding':
            model_path = f'{defpath.model_path.fasttext}/cc.en.{fasttext_model_size}.bin' if mode.startswith('ft') else f'{defpath.model_path.tabert}/tabert_base_k3/model.bin'
            tester = embedding.EmbeddingTester(*default_args, model_path, cidx_file, collections, fasttext_model_size)


    if DATA_PREPARATION or QUERY or SAMPLE_QUERIES:
        for directory in [TEST_DATASET_DIR, statistics_dir, results_base_dir, results_extr_dir, forest_dir, embedding_dir]:
            if not os.path.exists(directory): 
                logging.info(f'Creating directory {directory}...')
                os.makedirs(directory)
    
    
    logging_setup(logfile=logfile)
        
    if DATA_PREPARATION:
        logging.info(f'{"#" * 10} {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {k} - {dataset.upper()} - {size.upper()} - DATA PREPARATION {"#" * 10}')
        try:    
            exec_time, storage_size = tester.data_preparation()
            runtime_metrics.append((('data_preparation', None), exec_time, get_local_time()))
            append = os.path.exists(storage_stat_file)
            dbsize = pd.DataFrame([[algorithm, mode, storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
            dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)
        except Exception as e:
            logging.error(f"Error on data preparation: exception message {e.args}")
            raise Exception()
            

    if SAMPLE_QUERIES:
        logging.info(f'{"#" * 10} {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {k} - {dataset.upper()} - {size.upper()} - SAMPLING QUERIES {"#" * 10}')
        try:
            if not os.path.exists(query_file):
                num_samples = sample_queries(query_file, num_query_samples, tables_thresholds, *collections)
                logging.info(f'Sampled {num_samples} query tables (required {num_query_samples}).')
            else:
                logging.info(f'Query file for {num_query_samples} queries already present.')
        except Exception as e:
            logging.error(f"Error on sampling queries: n={num_query_samples}, query_file={query_file}, exception message {e.args}")
            raise Exception()


    if QUERY:
        logging.info(f'{"#" * 10} {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {k} - {dataset.upper()} - {size.upper()} - QUERY {"#" * 10}')
        try:
            query_ids = get_query_ids_from_query_file(query_file)
            match mode:
                case 'ft': query_mode = 'naive'
                case 'ftdist': query_mode = 'distance'
                case 'ftlsh': query_mode = 'hamming'

            exec_time = tester.query(topk_results_file, k, query_ids, results_directory=results_base_dir, token_table_on_memory=token_table_on_mem, query_mode=query_mode)
            runtime_metrics.append((('query', numerize(len(query_ids), asint=True)), exec_time, get_local_time()))
        except Exception as e:
            logging.error(f"Error on query: n={num_query_samples}, query_file={query_file}, exception message {e.args}")
            raise Exception()
        
            
    if DATA_PREPARATION or QUERY or SAMPLE_QUERIES:
        add_header = not os.path.exists(runtime_stat_file)
        with open(runtime_stat_file, 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,k,num_queries,time(s)\n")
            for ((t_name, num_queries), t_time, t_loctime) in runtime_metrics:
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{k},{num_queries},{t_time}\n")


    if CLEAN:
        logging.info(f'{"#" * 10} {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {dataset.upper()} - {size.upper()} - CLEANING {"#" * 10}')
        tester.clean()
        
    if mongoclient:
        mongoclient.close()


    logging.info('All tasks have been completed.')
