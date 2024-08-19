from argparse import ArgumentParser
import json
import os

from analysis_pl import analyses
from extract_results import extract_results
from main_pipeline import main_pipeline

from tools.utils.misc import sample_queries
from tools.utils.mongodb_utils import get_mongodb_collections
from tools.utils.settings import get_all_paths 
from tools.utils.basicconfig import tables_thresholds
from numerize_denumerize.numerize import numerize



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('configuration_file', help='path to the test configuration file')

    configuration_file = parser.parse_args().configuration_file

    print(configuration_file)

    with open(configuration_file, 'r') as fp:
        configuration = json.load(fp)

    test_name = configuration['test_name']
    dataset = configuration['dataset']
    size = configuration['size']
    num_cpu = configuration['num_cpu']

    blacklist = configuration['blacklist'] if 'blacklist' in configuration else []
    k = configuration['k'] if 'k' in configuration else None
    num_query_samples = configuration['num_query_samples'] if 'num_query_samples' in configuration else None

    pg_dbname = configuration['pg_dbname']

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
                    main_pipeline(test_name, algorithm, mode, ['data_preparation'], 
                                k, num_query_samples, num_cpu,
                                dataset, size,                        
                                blacklist=blacklist,
                                pg_dbname=pg_dbname,
                                **dp_config)
                    

    if 'sample_queries' in configuration:
        sq_config = configuration['sample_queries']
        if sq_config['exec_now']:
            mongoclient, collections = get_mongodb_collections(dataset, size)
            str_num_query_samples = numerize(num_query_samples, asint=True)

            TEST_DATASET_DIR, query_file, \
                _, _, _, _, _, _, _, _ = get_all_paths(test_name, dataset, k, str_num_query_samples)

            if not os.path.exists(query_file):
                num_samples = sample_queries(query_file, num_query_samples, tables_thresholds, *collections)
            else:
                print(f'Query file for {num_query_samples} already exists: {query_file}')
            mongoclient.close()


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
                    main_pipeline(test_name, algorithm, mode, ['data_preparation'], 
                                k, num_query_samples, num_cpu, 
                                dataset, size, 
                                blacklist=blacklist,
                                pg_dbname=pg_dbname,
                                **q_config)


    if 'extract' in configuration:
        e_config = configuration['extract']
        if e_config['exec_now']:
            extract_results(test_name, k, num_query_samples, num_cpu, pg_dbname, dataset, size, blacklist)


    if 'analyses' in configuration:
        a_config = configuration['analyses']
        if a_config['exec_now']:
            analyses(test_name, k, num_query_samples, num_cpu, dataset, size, a_config['p_values'])

