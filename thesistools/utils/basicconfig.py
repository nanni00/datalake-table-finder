from enum import IntEnum


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
    # ('embedding', 'ftdist'),
    # ('embedding', 'cftdist'),
]


class TablesThresholds(IntEnum):
    MIN_ROWS = 5
    MAX_ROWS = 1_000_000_000
    MIN_COLUMNS = 2
    MAX_COLUMNS = 1_000_000_000
    MIN_AREA = 0
    MAX_AREA = 1_000_000_000
    


DATALAKES = [
    'gittables',
    'wikitables',
    'wikiturlsnap',
    'santoslarge'
]

MONGODB_DATALAKES = [
    'gittables',
    'wikitables',
    'wikitables_small',
    'wikiturlsnap',
]


DATALAKE_SIZES = [
    'small',
    'standard'
]