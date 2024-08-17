algorithms = [
    'josie',
    'lshforest',
    'embedding'
]

modes = [
    'set',
    'bag',
    'ft',
    'ftdist',
    'ftlsh'
]

algmodeconfig = [
    ('josie', 'set'),
    ('josie', 'bag'),
    ('lshforest', 'set'),
    ('lshforest', 'bag'),
    ('embedding', 'ft'),
    ('embedding', 'ftdist'),
    ('embedding', 'ftlsh'),
]


# filtering only those tables that have very few cells (<10)
tables_thresholds = {
    'min_row':      5,
    'min_column':   2,
    'min_area':     0,
    'max_row':      999999,
    'max_column':   999999,
    'max_area':     999999,
}


datasets = [
    'gittables',
    'wikitables'
]

datasets_size = [
    'small',
    'standard'
]