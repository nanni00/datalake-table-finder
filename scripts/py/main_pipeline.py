"""
To add a new tester, create a class which inherits from tools.utils.classes.AlgorithmTester, implementing the 
methods data_preparation, query and clean (and __init__ obv) with the default interface and then add the specifications
for everything (loading models, modules, paths...) in the code below, indicating the 'algorithm' and 'mode' tag as well,
which will be used to specify the results file, in statistics and other stuff.
N.B. In the extraction and analyses phases there are grouping stages on the 'algorithm' and 'mode' tag, but differences in
parameters are not introduced in naming (e.g. no embedding-ft300 vs embedding-ft128), so if needed is necessary to create a different
test folder
"""
import os
from pprint import pprint

import pandas as pd
from numerize_denumerize.numerize import numerize

from tools.utils.logging import info, logging_setup
from tools.utils.settings import DefaultPath as defpath, get_all_paths, make_parser
from tools.utils.basicconfig import TABLES_THRESHOLDS
from tools.utils.misc import (
    get_local_time,
    get_query_ids_from_query_file, 
    whitespace_translator, punctuation_translator, lowercase_translator
)
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils import basicconfig
from tools import josie, lshforest, embedding



def main_pipeline(test_name, algorithm, mode, tasks:list[str], 
                  k:int=10, num_query_samples:int=1000, num_cpu:int=72, 
                  datalake_location:str=None,
                  dataset:str=None, size:str=None,
                  mapping_id_file:str=None,
                  numeric_columns_file:str=None,

                  # JOSIE specific parameters
                  pg_dbname:str='user', token_table_on_mem:bool=False, 
                  pg_user:str='user', pg_password:str='',
                  spark_local_dir=None,
                  
                  # LSH Forest specific parameters
                  num_perm:int=256, l:int=16, forest_file=None,

                  # fastText-FAISS specific parameters
                  embedding_model_path:str=None,
                  embedding_model_size:int=300,
                  embedding_translators=[whitespace_translator, punctuation_translator, lowercase_translator],

                  blacklist:list[str]=[]):
    
    # check configuration
    if (algorithm, mode) not in basicconfig.ALGORITHM_MODE_CONFIG:
        return
    
    assert int(k) > 0
    assert int(num_cpu) > 0
    assert datalake_location == 'mongodb' or os.path.exists(datalake_location)
    assert not dataset or dataset in basicconfig.DATALAKES
    assert not size or size in basicconfig.DATALAKE_SIZES
    assert not mapping_id_file or os.path.exists(mapping_id_file)
    assert not numeric_columns_file or os.path.exists(numeric_columns_file)
    assert not spark_local_dir or os.path.exists(spark_local_dir)
    assert int(num_perm) > 0
    assert int(l) > 0
    assert int(embedding_model_size) > 0
    assert not embedding_model_path or os.path.exists(embedding_model_path)

    
    test_name = test_name.lower()
    num_query_samples = int(num_query_samples)
    str_num_query_samples = numerize(num_query_samples, asint=True)
    
    # tasks to complete in current run
    DATA_PREPARATION =          'data_preparation' in tasks
    QUERY =                     'query' in tasks
    CLEAN =                     'clean' in tasks


    TEST_DATASET_DIR, query_file, logfile, \
        forest_dir, embedding_dir, \
            results_base_dir, results_extr_dir, \
                statistics_dir, runtime_stat_file, storage_stat_file = get_all_paths(test_name, dataset, k, str_num_query_samples)

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    logging_setup(logfile=logfile)

    info(f" MAIN PIPELINE - {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {dataset.upper()} - {size.upper()} ".center(150, '#'))

    # create folders
    if DATA_PREPARATION or QUERY:
        for directory in [statistics_dir, results_base_dir, results_extr_dir, forest_dir, embedding_dir]:
            if not os.path.exists(directory): 
                info(f'Creating directory {directory}...')
                os.makedirs(directory)
    

    forest_file =       f'{forest_dir}/forest_m{mode}.json' if not forest_file else forest_file
    idx_tag =           'ft' if mode in ['ft', 'ftdist'] else 'cft' if mode in ['cft', 'cftdist'] else ''
    cidx_file =         f'{embedding_dir}/col_idx_m{idx_tag}.index' 
    topk_results_file = f'{results_base_dir}/a{algorithm}_m{mode}.csv'
    db_stat_file =      statistics_dir + '/db.csv'


    # tokens that will be filtered
    blacklist = set(blacklist)
    info(f"Blacklist: {blacklist}")
    
    # a list containing information about timing of each step
    runtime_metrics = []

    datalake_helper = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)

    # the prefix used in the PostgreSQL database tables (mainly for JOSIE)
    table_prefix = f'{test_name}_d{dataset}_m{mode}'

    # selecting the right tester accordingly to the specified algorithm and mode
    tester = None
    default_args = (mode, dataset, size, TABLES_THRESHOLDS, num_cpu, blacklist, datalake_helper)
    match algorithm:
        case 'josie':
            tester = josie.JOSIETester(*default_args, pg_dbname, table_prefix, db_stat_file, pg_user, pg_password, spark_local_dir)
        case 'lshforest':
            tester = lshforest.LSHForestTester(*default_args, forest_file, num_perm, l)
        case 'embedding':
            model_path = f'{defpath.model_path.fasttext}/cc.en.300.bin' if not embedding_model_path else embedding_model_path
            tester = embedding.EmbeddingTester(*default_args, model_path, cidx_file, embedding_model_size, embedding_translators)

    
    if CLEAN:
        info(f' CLEANING '.center(150, '-'))
        tester.clean()


    if DATA_PREPARATION:
        info(f' DATA PREPARATION  '.center(150, '-'))    
        exec_time, storage_size = tester.data_preparation()
        runtime_metrics.append((('data_preparation', None), exec_time, get_local_time()))
        append = os.path.exists(storage_stat_file)
        dbsize = pd.DataFrame([[algorithm, mode, storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
        dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)
            

    if QUERY:
        info(f' QUERY - {k} - {str_num_query_samples} '.center(150, '-'))
        query_ids = get_query_ids_from_query_file(query_file)
        exec_time = tester.query(topk_results_file, k, query_ids, results_directory=results_base_dir, token_table_on_memory=token_table_on_mem)
        runtime_metrics.append((('query', numerize(len(query_ids), asint=True)), exec_time, get_local_time()))


    if DATA_PREPARATION or QUERY:
        add_header = not os.path.exists(runtime_stat_file)
        with open(runtime_stat_file, 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,k,num_queries,time(s)\n")
            for ((t_name, num_queries), t_time, t_loctime) in runtime_metrics:               
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{k if t_name == 'query' else ''},{num_queries if t_name == 'query' else ''},{t_time}\n")

    datalake_helper.close()
    info('All tasks have been completed.')



if __name__ == '__main__':
    args = make_parser('test_name', 'algorithm', 'mode', 'k', 'tasks', 'num_query_samples', 'num_cpu', 'size', 'dataset', 
                       'clean', 
                       'dbname', 'token_table_on_memory', 'pg_user', 'pg_password',
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
    pg_password =       args.pg_password
    pg_user =           args.pg_user

    # LSHForest
    num_perm =          args.num_perm
    l =                 args.l
    forest_file =       args.forest_file

    # fastText
    fasttext_model_size = args.fasttext_model_size

    clean = args.clean

    main_pipeline(test_name, algorithm, mode, 
                  tasks, k, num_query_samples, num_cpu, size, dataset,
                  user_dbname, token_table_on_mem, pg_password, pg_user,
                  num_perm, l, forest_file, fasttext_model_size, clean)