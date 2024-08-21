from argparse import ArgumentParser
import json
import os

from analysis_pl import analyses
from extract_results import extract_results
from main_pipeline import main_pipeline

from tools.utils.misc import sample_queries
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.settings import get_all_paths 
from tools.utils.basicconfig import TABLES_THRESHOLDS
from numerize_denumerize.numerize import numerize



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('configuration_file', help='path to the test configuration file')

    configuration_file = parser.parse_args().configuration_file

    with open(configuration_file, 'r') as fp:
        configuration = json.load(fp)
    
    # general configuration
    g_config = configuration['general']

    if 'data_preparation' in configuration:
        dp_config = configuration['data_preparation']
        if dp_config['exec_now']:
            algorithms = dp_config['algorithms']
            del dp_config['algorithms']
            modes = dp_config['modes']
            del dp_config['modes']
            del dp_config['exec_now']

            for algorithm in algorithms:
                for mode in modes:
                    main_pipeline(algorithm=algorithm, mode=mode, tasks=['data_preparation'], 
                                **g_config,
                                **dp_config)
                    

    if 'sample_queries' in configuration:
        sq_config = configuration['sample_queries']
        if sq_config['exec_now']:
            num_query_samples = g_config['num_query_samples']

            dlh = SimpleDataLakeHelper(g_config['datalake_location'], 
                                       g_config['dataset'], g_config['size'], 
                                       g_config['mapping_id_file'], g_config['numeric_columns_file'])
            
            str_num_query_samples = numerize(num_query_samples, asint=True)

            TEST_DATASET_DIR, query_file, \
                _, _, _, _, _, _, _, _ = get_all_paths(g_config['test_name'], g_config['dataset'], g_config['k'], str_num_query_samples)

            if not os.path.exists(query_file):
                num_samples = sample_queries(query_file, num_query_samples, TABLES_THRESHOLDS, dlh)
            else:
                print(f'Query file for {num_query_samples} already exists: {query_file}')
            
            dlh.close()


    if 'query' in configuration:
        q_config = configuration['query']
        if q_config['exec_now']:
            algorithms = q_config['algorithms']
            del q_config['algorithms']
            modes = q_config['modes']
            del q_config['modes']
            del q_config['exec_now']

            for algorithm in algorithms:
                for mode in modes:
                    main_pipeline(algorithm=algorithm, mode=mode, tasks=['query'], 
                                **g_config,
                                **q_config)

    if 'ft_model_path' in g_config:
        del g_config['ft_model_path']

    if 'extract' in configuration:
        e_config = configuration['extract']
        if e_config['exec_now']:
            extract_results(**g_config)


    if 'analyses' in configuration:
        a_config = configuration['analyses']
        if a_config['exec_now']:
            analyses(**g_config, p_values=a_config['p_values'])

