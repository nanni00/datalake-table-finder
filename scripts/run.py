import os
import json
from itertools import product
from argparse import ArgumentParser

from scripts.analysis import analyses
from scripts.gsa_pipeline import gsa_pipeline
from extract_results import extract_results

from dltf.utils.misc import numerize
from dltf.utils.query import sample_queries
from dltf.utils.settings import get_all_paths 


def run(configuration_file):
    with open(configuration_file, 'r') as fp:
        configurations = json.load(fp)
    
    # general configuration
    g_conf              = configurations['general']

    # datalake configuration
    datalake_location   = g_conf['datalake_config']['location']
    datalake_name       = g_conf['datalake_config']['name']
    datalake_options    = g_conf['datalake_config']['options']
    dlhargs             = [datalake_location, datalake_name, datalake_options]
    del g_conf['datalake_config']

    # which algorithms-modes have to be executed
    algorithms          = g_conf['algorithms']
    modes               = g_conf['modes']
    del g_conf['algorithms']
    del g_conf['modes']

    num_query_samples   = configurations['sample_queries']['num_query_samples']

    if configurations['sample_queries']['exec']:
        str_num_query_samples = numerize(num_query_samples)

        paths = get_all_paths(g_conf['test_name'], datalake_name, num_query_samples=str_num_query_samples)
        query_file = paths['query_file']

        if not os.path.exists(os.path.dirname(query_file)):
            os.makedirs(os.path.dirname(query_file))

        if not os.path.exists(query_file):
            sample_queries(query_file, num_query_samples, g_conf['num_cpu'], *dlhargs)
        else:
            print(f'Query file for {num_query_samples} already exists: {query_file}')

    # the steps configurations
    dp_conf     = configurations['data_preparation']
    q_conf      = configurations['query']
    c_conf      = configurations['clean']

    # 'exec' defines if the relative part has to be executed
    dp_exec     = dp_conf['exec']
    q_exec      = q_conf['exec']
    c_exec      = c_conf['exec']

    # the number of results to return on query
    k           = q_conf['k']

    tasks = []
    tasks += ['data_preparation'] if dp_exec else []
    tasks += ['query'] if q_exec else []
    tasks += ['clean'] if c_exec else []
    
    del dp_conf['exec']
    del q_conf['exec']
    del c_conf['exec']
        

    if any(tasks):
        for algorithm, mode in product(algorithms, modes):
            gsa_pipeline(
                algorithm=algorithm, 
                mode=mode, 
                tasks=tasks,
                datalake_location=datalake_location,
                datalake_name=datalake_name,
                datalake_options=datalake_options, 
                num_query_samples=num_query_samples,
                **g_conf,
                **dp_conf,
                **q_conf,
                **c_conf)
        
    # delete these parameters, they're no longer needed
    if 'embedding_model_path' in g_conf:
        del g_conf['embedding_model_path']
    if 'embedding_model_size' in g_conf:
        del g_conf['embedding_model_size']

    # The extraction step
    if configurations['extract']['exec']:
        clear_results_table = configurations['extract']['clear_results_table']
        connection_info=g_conf['josie_db_connection_info']
        del g_conf['josie_db_connection_info']
        extract_results(datalake_location=datalake_location,
                        datalake_name=datalake_name,
                        datalake_options=datalake_options,
                        clear_results_table=clear_results_table,
                        connection_info=connection_info,
                        k=k,
                        num_query_samples=num_query_samples,
                        **g_conf)

    # The analysis step
    if configurations['analyses']['exec']:
        analyses(datalake_name=datalake_name, 
                 k=k, 
                 k_values=configurations['analyses']['k_values'],
                 num_query_samples=num_query_samples,
                 **g_conf)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('configuration_file', help='path to the test configuration file')
    configuration_file = parser.parse_args().configuration_file

    run(configuration_file=configuration_file)
