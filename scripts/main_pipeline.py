"""
The main pipeline. For each tester, here you can do the data preparation and query steps
Headers are considered as part of table, to modify that see how each tester loads tables
from the datalake
"""

import os

import pandas as pd

from dltftools.testers.josie import josie
from dltftools.utils.query import read_query_ids
from dltftools.testers import lshforest, embedding
from dltftools.utils.datalake import DataLakeHandlerFactory
from dltftools.utils.loghandler import info, logging_setup

from dltftools.utils.settings import (
    get_all_paths, 
    ALGORITHM_MODE_CONFIG, 
    DATALAKES
)

from dltftools.utils.misc import (
    numerize,
    mmh3_hashfunc,
    get_local_time,
    get_string_translator,
)


def main_pipeline(test_name:str, algorithm:str, mode:str, tasks:list[str], 
                  k:int=10, num_query_samples:int=1000, num_cpu:int=72, 
                  datalake_location:str=None,
                  datalake_name:str=None, 
                  datalake_options:list[str]=None,
                  token_translators=[],
                  blacklist:list[str]=[],

                  # JOSIE specific parameters
                  josie_db_connection_info:dict=None,
                  token_table_on_mem:bool=False, 
                  spark_config:dict=None,
                  
                  # LSH Forest specific parameters
                  num_perm:int=256, l:int=16, hash_func=mmh3_hashfunc,
                  forest_file=None,

                  # fastText-FAISS specific parameters
                  embedding_model_path:str|None=None,
                  embedding_model_size:int=300):
    
    # check configuration
    if (algorithm, mode) not in ALGORITHM_MODE_CONFIG:
        return
    
    assert int(k) > 0
    assert int(num_cpu) > 0
    assert datalake_location == 'mongodb' or os.path.exists(datalake_location)
    assert not datalake_name or datalake_name in DATALAKES
    assert int(num_perm) > 0
    assert int(l) > 0
    assert int(embedding_model_size) > 0
    assert (not embedding_model_path and algorithm != 'embedding') or os.path.exists(embedding_model_path)
    
    test_name = test_name.lower()
    num_query_samples = int(num_query_samples)
    str_num_query_samples = numerize(num_query_samples, asint=True)
    
    # tasks to complete in current run
    DATA_PREPARATION =          'data_preparation' in tasks
    QUERY =                     'query' in tasks
    CLEAN =                     'clean' in tasks


    p = get_all_paths(test_name, datalake_name, k, str_num_query_samples)
    TEST_DATASET_DIR = p['TEST_DATASET_DIR']
    
    (
        statistics_dir, 
        results_base_dir, 
        results_extr_dir, 
        forest_dir, 
        embedding_dir
    ) = (
        p['statistics_dir'], 
        p['results_base_dir'], 
        p['results_extr_dir'], 
        p['forest_dir'], 
        p['embedding_dir']
    )

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    logging_setup(logfile=p['logfile'])

    info(f" MAIN PIPELINE - {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {datalake_name.upper()} ".center(150, '#'))

    # create folders
    if DATA_PREPARATION or QUERY:
        for directory in [statistics_dir, results_base_dir, results_extr_dir, forest_dir, embedding_dir]:
            if not os.path.exists(directory): 
                info(f'Creating directory {directory}...')
                os.makedirs(directory)
    

    forest_file =       f'{forest_dir}/forest_{mode}.json' if not forest_file else forest_file
    cidx_tag =           'ft' if mode in ['ft', 'ftdist'] else 'cft' if mode in ['cft', 'cftdist'] else ''
    cidx_file =         f'{embedding_dir}/col_idx_{cidx_tag}.index' 
    topk_results_file = f'{results_base_dir}/{algorithm}_{mode}.csv'
    db_stat_file =      f'{statistics_dir}/db.csv' 


    # tokens that will be filtered
    blacklist = set(blacklist)
    info(f"Blacklist: {blacklist}")

    token_translators = [get_string_translator(tr) for tr in token_translators]
    
    # a list containing information about timing of each step
    runtime_metrics = []

    # the datalake handler, that provides utilities to access the tables
    dlh_config = [datalake_location, datalake_name, datalake_options]
    dlh = DataLakeHandlerFactory.create_handler(*dlh_config)

    # selecting the right tester accordingly to the specified algorithm and mode
    tester = None
    default_args = (mode, blacklist, dlh, token_translators)
    match algorithm:
        case 'josie':
            tester = josie.JOSIETester(*default_args, db_stat_file, josie_db_connection_info, spark_config)
        case 'lshforest':
            tester = lshforest.LSHForestTester(*default_args, num_cpu, forest_file, num_perm, l, hash_func)
        case 'embedding':
            tester = embedding.EmbeddingTester(*default_args, num_cpu, embedding_model_path, cidx_file, embedding_model_size)

    
    if CLEAN:
        info(f' CLEANING '.center(150, '-'))
        tester.clean()


    if DATA_PREPARATION:
        info(f' DATA PREPARATION  '.center(150, '-'))    
        exec_time, storage_size = tester.data_preparation()
        runtime_metrics.append((('data_preparation', None), exec_time, get_local_time()))
        append = os.path.exists(p['storage_stat_file'])
        dbsize = pd.DataFrame([[algorithm, mode, storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
        dbsize.to_csv(p['storage_stat_file'], index=False, mode='a' if append else 'w', header=False if append else True)
            

    if QUERY:
        info(f' QUERY - {k} - {str_num_query_samples} '.center(150, '-'))
        query_ids = read_query_ids(p['query_file'])
        exec_time = tester.query(topk_results_file, k, query_ids, results_directory=results_base_dir, token_table_on_memory=token_table_on_mem)
        runtime_metrics.append((('query', numerize(len(query_ids), asint=True)), exec_time, get_local_time()))


    if DATA_PREPARATION or QUERY:
        add_header = not os.path.exists(p['runtime_stat_file'])
        with open(p['runtime_stat_file'], 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,k,num_queries,time(s)\n")
            for ((t_name, num_queries), t_time, t_loctime) in runtime_metrics:               
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{k if t_name == 'query' else ''},{num_queries if t_name == 'query' else ''},{t_time}\n")


    dlh.close()
    info('All tasks have been completed.')
