import os
import json
from argparse import ArgumentParser

from analysis_pl import analyses
from main_pipeline import main_pipeline
from extract_results import extract_results

from dltftools.utils.misc import numerize
from dltftools.utils.query import sample_queries
from dltftools.utils.settings import get_all_paths 


def run(configuration_file):
    with open(configuration_file, 'r') as fp:
        configuration = json.load(fp)
    
    # general configuration
    g_config = configuration['general']

    # datalake configuration
    # used for data preparation, query and extraction steps
    datalake_location = g_config['datalake_config']['location']
    datalake_name = g_config['datalake_config']['name']
    datalake_options = g_config['datalake_config']['options']
    dlhargs = [datalake_location, datalake_name, datalake_options]
    del g_config['datalake_config']

    if 'data_preparation' in configuration:
        dp_config = configuration['data_preparation']
        if dp_config['exec_now']:
            algorithms = dp_config['algorithms']
            modes = dp_config['modes']
            # drop these to avoid keyword errors in function call
            del dp_config['algorithms']
            del dp_config['modes']
            del dp_config['exec_now']

            for algorithm in algorithms:
                for mode in modes:
                    main_pipeline(algorithm=algorithm, mode=mode, tasks=['data_preparation'],
                                  datalake_location=datalake_location,
                                  datalake_name=datalake_name,
                                  datalake_options=datalake_options, 
                                  **g_config,
                                  **dp_config)
                    

    if 'sample_queries' in configuration:
        sq_config = configuration['sample_queries']
        if sq_config['exec_now']:
            num_query_samples = g_config['num_query_samples']

            str_num_query_samples = numerize(num_query_samples, asint=True)

            paths = get_all_paths(g_config['test_name'], datalake_name, g_config['k'], str_num_query_samples)
            query_file = paths['query_file']

            if not os.path.exists(query_file):
                num_samples = sample_queries(query_file, num_query_samples, g_config['num_cpu'], *dlhargs)
            else:
                print(f'Query file for {num_query_samples} already exists: {query_file}')


    if 'query' in configuration:
        q_config = configuration['query']
        if q_config['exec_now']:
            algorithms = q_config['algorithms']
            modes = q_config['modes']
            del q_config['algorithms']
            del q_config['modes']
            del q_config['exec_now']

            for algorithm in algorithms:
                for mode in modes:
                    main_pipeline(algorithm=algorithm, mode=mode, tasks=['query'],
                                  datalake_location=datalake_location,
                                  datalake_name=datalake_name,
                                  datalake_options=datalake_options, 
                                  **g_config,
                                  **q_config)
                    
    if 'clean' in configuration:
        q_config = configuration['clean']
        if q_config['exec_now']:
            algorithms = q_config['algorithms']
            modes = q_config['modes']
            del q_config['algorithms']
            del q_config['modes']
            del q_config['exec_now']

            for algorithm in algorithms:
                for mode in modes:
                    main_pipeline(algorithm=algorithm, mode=mode, tasks=['clean'],
                                  datalake_location=datalake_location,
                                  datalake_name=datalake_name,
                                  datalake_options=datalake_options,
                                  **g_config,
                                  **q_config)

    
    # delete these parameters, they're no longer needed
    if 'embedding_model_path' in g_config:
        del g_config['embedding_model_path']
    if 'embedding_model_size' in g_config:
        del g_config['embedding_model_size']


    if 'extract' in configuration:
        e_config = configuration['extract']
        if e_config['exec_now']:
            clear_results_table = e_config['clear_results_table']
            connection_info=g_config['josie_db_connection_info']
            del g_config['josie_db_connection_info']
            extract_results(datalake_location=datalake_location,
                            datalake_name=datalake_name,
                            datalake_options=datalake_options,
                            clear_results_table=clear_results_table,
                            connection_info=connection_info, **g_config)


    if 'analyses' in configuration:
        a_config = configuration['analyses']
        if a_config['exec_now']:
            analyses(datalake_name=datalake_name, **g_config, p_values=a_config['p_values'])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('configuration_file', help='path to the test configuration file')
    configuration_file = parser.parse_args().configuration_file

    # configuration_file = 'scripts/configurations/base/santossmall.json'
    
    run(configuration_file=configuration_file)
