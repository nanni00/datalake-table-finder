from enum import Enum, IntEnum


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
    ('embedding', 'ftdist'),
    ('embedding', 'cft'),
    ('embedding', 'cftdist'),
]


# filtering only those tables that have very few cells (<10)
TABLES_THRESHOLDS = {
    'min_row':      5,
    'min_column':   2,
    'min_area':     0,
    'max_row':      999999,
    'max_column':   999999,
    'max_area':     999999,
}


class TablesThresholds(IntEnum):
    MIN_ROWS = 5
    MAX_ROWS = 9999999
    MIN_COLUMNS = 2
    MAX_COLUMNS = 9999999
    MIN_AREA = 0
    MAX_AREA = 9999999
    


DATASETS = [
    'gittables',
    'wikitables',
    'wikiturlsnap',
    'santoslarge'
]

MONGODB_DATASETS = [
    'gittables',
    'wikitables',
    'wikiturlsnap',
]


DATASETS_SIZES = [
    'small',
    'standard'
]