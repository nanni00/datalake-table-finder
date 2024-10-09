import os

import pandas as pd
from numerize_denumerize.numerize import numerize

from thesistools.testers.josie import josie
from thesistools.testers import lshforest, embedding
from thesistools.utils.logging_handler import info, logging_setup
from thesistools.utils.settings import DefaultPath as defpath, get_all_paths
from thesistools.utils.datalake import SimpleDataLakeHelper
from thesistools.utils import basicconfig
from thesistools.utils.misc import (
    mmh3_hashfunc,
    get_local_time,
    get_string_translator,
    get_query_ids_from_query_file
)



def main_pipeline(test_name, algorithm, mode, tasks:list[str], 
                  k:int=10, num_query_samples:int=1000, num_cpu:int=72, 
                  datalake_location:str=None,
                  datalake_name:str=None, 
                  datalake_size:str=None,
                  mapping_id_file:str=None,
                  numeric_columns_file:str=None,
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
                  embedding_model_path:str=f'{defpath.model_path.fasttext}/cc.en.300.bin',
                  embedding_model_size:int=300):
    
    # check configuration
    if (algorithm, mode) not in basicconfig.ALGORITHM_MODE_CONFIG:
        return
    
    assert int(k) > 0
    assert int(num_cpu) > 0
    assert datalake_location == 'mongodb' or os.path.exists(datalake_location)
    assert not datalake_name or datalake_name in basicconfig.DATALAKES
    assert not datalake_size or datalake_size in basicconfig.DATALAKE_SIZES
    assert not mapping_id_file or os.path.exists(mapping_id_file)
    assert not numeric_columns_file or os.path.exists(numeric_columns_file)
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


    p = get_all_paths(test_name, datalake_name, k, str_num_query_samples)
    TEST_DATASET_DIR = p['TEST_DATASET_DIR']
    statistics_dir, results_base_dir, results_extr_dir, forest_dir, embedding_dir = p['statistics_dir'], p['results_base_dir'], p['results_extr_dir'], p['forest_dir'], p['embedding_dir']

    if not os.path.exists(TEST_DATASET_DIR):
        os.makedirs(TEST_DATASET_DIR)

    logging_setup(logfile=p['logfile'])

    info(f" MAIN PIPELINE - {test_name.upper()} - {algorithm.upper()} - {mode.upper()} - {datalake_name.upper()} - {datalake_size.upper()} ".center(150, '#'))

    # create folders
    if DATA_PREPARATION or QUERY:
        for directory in [statistics_dir, results_base_dir, results_extr_dir, forest_dir, embedding_dir]:
            if not os.path.exists(directory): 
                info(f'Creating directory {directory}...')
                os.makedirs(directory)
    

    forest_file =       f'{forest_dir}/forest_m{mode}.json' if not forest_file else forest_file
    idx_tag =           'ft' if mode in ['ft', 'ftdist'] else 'cft' if mode in ['cft', 'cftdist'] else ''
    cidx_file =         f'{embedding_dir}/col_idx_m{idx_tag}.index' 
    topk_results_file = f'{results_base_dir}/{algorithm}_{mode}.csv'
    db_stat_file =      f'{statistics_dir}/db.csv' 


    # tokens that will be filtered
    blacklist = set(blacklist)
    info(f"Blacklist: {blacklist}")

    token_translators = [get_string_translator(tr) for tr in token_translators]
    
    # a list containing information about timing of each step
    runtime_metrics = []

    datalake_helper = SimpleDataLakeHelper(datalake_location, datalake_name, datalake_size, mapping_id_file, numeric_columns_file)

    # the prefix used in the database tables (mainly for JOSIE)
    table_prefix = f'josie__{test_name}__{datalake_name}_{datalake_size}_{mode}'

    # selecting the right tester accordingly to the specified algorithm and mode
    tester = None
    default_args = (mode, blacklist, datalake_helper, token_translators)
    match algorithm:
        case 'josie':
            tester = josie.JOSIETester(*default_args, table_prefix, db_stat_file, josie_db_connection_info, spark_config)
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
        query_ids = get_query_ids_from_query_file(p['query_file'])
        exec_time = tester.query(topk_results_file, k, query_ids, results_directory=results_base_dir, token_table_on_memory=token_table_on_mem)
        runtime_metrics.append((('query', numerize(len(query_ids), asint=True)), exec_time, get_local_time()))


    if DATA_PREPARATION or QUERY:
        add_header = not os.path.exists(p['runtime_stat_file'])
        with open(p['runtime_stat_file'], 'a') as rfw:
            if add_header:
                rfw.write("local_time,algorithm,mode,task,k,num_queries,time(s)\n")
            for ((t_name, num_queries), t_time, t_loctime) in runtime_metrics:               
                rfw.write(f"{t_loctime},{algorithm},{mode},{t_name},{k if t_name == 'query' else ''},{num_queries if t_name == 'query' else ''},{t_time}\n")


    datalake_helper.close()
    info('All tasks have been completed.')
