import os
from enum import IntEnum


# remember to set the env variable DLTFPATH 
root_project_path = os.environ['DLTFPATH']


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




ALGORITHMS = [
    'josie',
    'lshforest',
    'embedding'
]


MODES = [
    'set',
    'bag',
    'ft',
    'ftdist',
    'ftlsh'
]

ALGORITHM_MODE_CONFIG = [
    ('josie', 'set'),
    ('josie', 'bag'),
    ('lshforest', 'set'),
    ('lshforest', 'bag'),
    ('embedding', 'ft'),
    ('embedding', 'deepjoin'),
    ('embedding', 'cft')
]


class TablesThresholds(IntEnum):
    MIN_ROWS    = 5
    MAX_ROWS    = 1_000_000_000
    MIN_COLUMNS = 2
    MAX_COLUMNS = 1_000_000_000
    MIN_AREA    = 0
    MAX_AREA    = 1_000_000_000
    


DATALAKES = [
    'gittables',
    'wikitables',
    'wikiturlsnap',
    'santoslarge',
    'santossmall',
    'latsnaptab',
    'demo'
]


def get_all_paths(test_name, datalake_name, k=None, num_query_samples=None):
    p = {}

    # output files and directories
    ROOT_TEST_DIR =             f'{DefaultPath.data_path.tests}/{test_name}'
    p['TEST_DATASET_DIR'] =     TEST_DATASET_DIR = f'{ROOT_TEST_DIR}/{datalake_name}'
    p['query_file'] =           f'{TEST_DATASET_DIR}/query_{num_query_samples}.json'
    p['logfile'] =              f'{TEST_DATASET_DIR}/logging.log'

    p['josie_dir'] =            f'{TEST_DATASET_DIR}/josie'

    # LSH-Forest stuff
    p['forest_dir'] =           f'{TEST_DATASET_DIR}/lshforest' 
    
    # embedding stuff
    p['embedding_dir'] =        f'{TEST_DATASET_DIR}/embedding'
    
    if k:
        # results stuff
        p['results_base_dir'] =     f'{TEST_DATASET_DIR}/results/base/k{k}_q{num_query_samples}'
        p['results_extr_dir'] =     f'{TEST_DATASET_DIR}/results/extracted'

    if num_query_samples:
        # statistics stuff
        p['statistics_dir'] =       statistics_dir = TEST_DATASET_DIR  + '/statistics'
        p['runtime_stat_file'] =    statistics_dir + '/runtime.csv'     
        p['storage_stat_file'] =    statistics_dir + '/storage.csv'

    return p

