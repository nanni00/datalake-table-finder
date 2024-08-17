import os
import argparse

from tools.utils import basicconfig



# remember to set the env variable TESI_ROOT_PATH 
root_project_path = os.environ['THESIS_PATH']


class _DataPath:
    base =                  root_project_path + '/data'
    tests =                 root_project_path + '/data/tests'


class _ModelPath:
    base =                  root_project_path + '/models'
    fasttext =              root_project_path + '/models/fasttext'
    tabert =                root_project_path + '/models/TaBERT'


class _DBPath:
    base =                  root_project_path + '/db'
    chroma =                root_project_path + '/db/chroma'
    faiss =                 root_project_path + '/db/faiss'


class DefaultPath:
    root_project_path = root_project_path
    model_path = _ModelPath
    data_path = _DataPath
    db_path = _DBPath






def make_parser(*arguments):
    parser = argparse.ArgumentParser()

    for arg in arguments:
        match arg:
            case 'test_name':
                parser.add_argument('--test-name', 
                                    type=str, required=True,
                                    help='the test name. It will be always considered as a lower-case string')
            case 'algorithm':
                parser.add_argument('-a', '--algorithm',
                                    required=False, default='josie',
                                    choices=basicconfig.algorithms)
            case 'mode':
                parser.add_argument('-m', '--mode', 
                                    required=False, default='set',
                                    choices=basicconfig.modes,
                                    help='the specific version of the algorithm. Note that an algorithm doesn\'t support all the available modes')
            case 'k':
                parser.add_argument('-k', 
                                    type=int, required=False, default=5,
                                    help='the K value for the top-K search')
            case 'tasks':
                parser.add_argument('-t', '--tasks', 
                                    required=False, nargs='+',
                                    choices=['all', 
                                            'data-preparation',
                                            'sample-queries', 
                                            'query'], 
                                    help='the tasks to do')
            case 'num_query_samples':
                parser.add_argument('--num-query-samples', 
                                    required=False, default=1000,
                                    help='the number of tables that will be sampled from the collections and that will be used as query id for JOSIE (the actual number) \
                                    may be less than the specified one due to thresholds tables parameter')
            case 'num_cpu':
                parser.add_argument('--num-cpu', 
                                    type=int, required=False, default=min(os.cpu_count(), 96),
                                    help='number of CPUs that will be used in the experiment')
            case 'size':
                parser.add_argument('--size', 
                                    type=str, choices=basicconfig.datasets_size,
                                    required=False, default='standard',
                                    help='works on small collection versions (only for testing)')
            case 'dataset':
                parser.add_argument('--dataset', 
                                    required=True, choices=basicconfig.datasets)

            # Clean task
            case 'clean':            
                parser.add_argument('--clean', 
                                    required=False, action='store_true', 
                                    help='remove PostgreSQL database tables and other big files, such as the LSH Forest index file')
   
            # JOSIE specific arguments
            case 'dbname':
                parser.add_argument('-d', '--dbname', 
                                    required=False, default='user',
                                    help='the PostgreSQL database where will be uploaded the data used by JOSIE. It must be already running on the machine')
            case 'token_table_on_memory':
                parser.add_argument('--token-table-on-memory', required=False, action='store_true')
            case 'pg_user': 
                parser.add_argument('--pg-user', required=False, default='user')
            case 'pg_password': 
                parser.add_argument('--pg-password', required=False, default='')

            # LSH Forest specific arguments
            case 'forest_file':
                parser.add_argument('--forest-file', 
                                    required=False, type=str, 
                                    help='the location of the LSH Forest index file that will be used for querying. If it does not exist, \
                                        a new index will be created at that location.')
            case 'num_perm':
                parser.add_argument('--num-perm', 
                                    required=False, type=int, default=128,
                                    help='number of permutations to use for minhashing')
            case 'l':
                parser.add_argument('-l', 
                                    required=False, type=int, default=8,
                                    help='number of prefix trees (see datasketch.LSHForest documentation)')
                
            # Embedding versions specific arguments
            case 'fasttext_model_size':
                parser.add_argument('--fasttext-model-size',
                                    required=False, type=int, default=300,
                                    help='the size of the fastText model output vectors size (default 300).')

            # Neo4j graph specific arguments
            case 'neo4j_user':
                parser.add_argument('--neo4j-user', 
                                    required=False, type=str, default='neo4j')
            case 'neo4j_password':
                parser.add_argument('--neo4j-password', 
                                    required=False, type=str, default='12345678')

            # Analyses arguments
            case 'p_values':
                parser.add_argument('--p-values',
                                    required=False, nargs='+', type=int,
                                    help='the values used as cut-off for some metrics, such as precision@P and nDCG@P')

            case '_':
                raise ValueError('Unknown input argument: ' + arg)


    return parser.parse_args()




def get_all_paths(test_name, dataset, k, num_query_samples):
    # output files and directories
    ROOT_TEST_DIR =             f'{DefaultPath.data_path.tests}/{test_name}'
    TEST_DATASET_DIR =          f'{ROOT_TEST_DIR}/{dataset}'
    query_file =                f'{TEST_DATASET_DIR}/query_{num_query_samples}.json'
    logfile =                   f'{TEST_DATASET_DIR}/logging.log'

    # LSH-Forest stuff
    forest_dir =                f'{TEST_DATASET_DIR}/lshforest' 
    
    # embedding stuff
    embedding_dir =             f'{TEST_DATASET_DIR}/embedding'

    # results stuff
    results_base_dir =          f'{TEST_DATASET_DIR}/results/base/k{k}_q{num_query_samples}'
    results_extr_dir =          f'{TEST_DATASET_DIR}/results/extracted'

    # statistics stuff
    statistics_dir =            TEST_DATASET_DIR  + '/statistics'
    runtime_stat_file =         statistics_dir + '/runtime.csv'     
    storage_stat_file =         statistics_dir + '/storage.csv'

    return [TEST_DATASET_DIR, query_file, logfile, 
            forest_dir, 
            embedding_dir, 
            results_base_dir, results_extr_dir,
            statistics_dir, runtime_stat_file, storage_stat_file]


