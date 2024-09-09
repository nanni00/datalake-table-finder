import re
import sys
import time
import json
import random
import logging
import logging.handlers
from functools import reduce
import multiprocessing as mp
from collections import defaultdict
from string import whitespace, digits, punctuation, ascii_uppercase, ascii_lowercase

import pandas as pd
from datasketch.minhash import MinHash

from tools.sloth.sloth import sloth
from tools.sloth.utils import parse_table
from tools.utils.parallel_worker import chunks
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.basicconfig import TablesThresholds as tab_thresh



_TOKEN_TAG_SEPARATOR = '@#'
whitespace_translator =     str.maketrans(whitespace, ' ' * len(whitespace))
digits_translator =         str.maketrans(digits, ' ' * len(digits))
punctuation_translator =    str.maketrans(punctuation, ' ' * len(punctuation))
lowercase_translator =      str.maketrans(ascii_uppercase, ascii_lowercase)


def jaccard(X:set, Y:set):
    intersection_size = 0
    union_size = 0
    for x in X:
        if x in Y:
            intersection_size += 1
        union_size += 1
    if not intersection_size or not union_size: 
        return 0
    for y in Y:
        if y not in X: union_size += 1
    return intersection_size / union_size if union_size else 0


metrics = {
    'jaccard':      jaccard, # lambda X, Y: len(X.intersection(Y)) / len(X.union(Y)) if len(X.union(Y)) > 0 else 0,
    'containment':  lambda X, Y: len(X.intersection(Y)) / min(len(X), len(Y)) if min(len(X), len(Y)) > 0 else 0
}


def create_minhash(iterable):
    m = MinHash()
    m.update_batch([str(i).encode() for i in iterable])
    return m



def prepare_token(token):
    return str(token).replace('|', ' ').replace('\n', ' ')


def token_to_str(token, *translators):
    return reduce(lambda to, tr: str(to).translate(tr), translators, str(token)).strip()


def tokenize_column(column, *translators):
    """Extract distinct tokens from the column, apply on them the translators and remove empty strings """
    column = [token_to_str(token, *translators) for token in set(column)]
    return [token for token in column if set(token)]


def column_to_text(column, *translators):
    return ' '.join(tokenize_column(column, *translators))


def table_rows_to_columns(table, num_headers, num_cols, bad_columns:list=[]):
    """ Restructure the table as the list of its columns, ignoring the headers """
    return [[row[i] for row in table[num_headers:]] for i in range(num_cols) if bad_columns[i] == 0]


def table_rows_to_rows(table, num_headers, num_cols, bad_columns:list=[]):
    """ Keeps the row-view structure of the table, removing cells from bad columns """
    return [[row[i] for i, x in enumerate(bad_columns) if x == 0] for row in table[num_headers:]]


def are_joinable_columns(c1: list, c2:list, t:float, metric:str):
    return int(metrics[metric](set(c1), set(c2)) >= t)


def print_info(**dec_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'msg_before' in dec_kwargs: logging.getLogger('TestLog').info(dec_kwargs['msg_before'])
            result = func(*args, **kwargs)
            if 'msg_after' in dec_kwargs: logging.getLogger('TestLog').info(dec_kwargs['msg_after'])
            return result
        return wrapper
    return decorator


def get_local_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")


def convert_to_giga(x):
    if x.endswith('GB'):
        return int(re.match(r'\d+', x).group())
    elif x.endswith('MB'):
        return int(re.match(r'\d+', x).group()) / 1024
    elif x.endswith('KB'):
        return int(re.match(r'\d+', x).group()) / (1024 ** 2)


def is_valid_table(table, bad_columns):
    if all(bad_columns):
        return False
    # here is kept a row-view for the table
    # table = [[row[i] for i, x in enumerate(numeric_columns) if x == 0] for row in table]
    table = table_rows_to_rows(table, 0, len(table[0]), bad_columns)
    return  tab_thresh.MIN_ROWS     <= len(table)                   <= tab_thresh.MAX_ROWS and \
            tab_thresh.MIN_COLUMNS  <= len(table[0])                <= tab_thresh.MAX_COLUMNS and \
            tab_thresh.MIN_AREA     <= len(table) * len(table[0])   <= tab_thresh.MAX_AREA
    


def is_number_tryexcept(s):
    try: float(s)
    except (ValueError, TypeError): return False
    return True


def is_bad_column(column:list, tokenize=lambda cell: cell):
    column = map(tokenize, column)
    return sum(map(lambda cell: is_number_tryexcept(cell) or pd.isna(cell), column)) >= len(column) // 2 + 1


def naive_detect_table_numeric_and_null_columns(table: list[list], tokenize=lambda cell: cell) -> list[int]:
    """ 
    :param table: a list of lists representing a table in row view
    :param any_int: if set to True, a column is detected as an numeric column wheter any of its values is 
        a numeric value, else if the majority (i.e. #numeric_cells >= #column_size / 2 + 1) of its values are numerics
    :return a list of int, where the i-th element is set to 1 if the i-th column is detected as numeric or with only null values, 
        0 otherwise
    """
        
    if len(table) == 0 or len(table[0]) == 0:
        return []

    return [
        # int(sum(not cell or is_number_tryexcept(cell) for cell in column) >= len(column) // 2 + 1 or not any(cell for cell in column if not is_number_tryexcept(cell)))
        int(is_bad_column(column, tokenize))
        for column in parse_table(table, len(table[0]), 0)
    ]



def table_to_tokens_set(table, mode, numeric_columns, encode=None, blacklist:set=set()):
    """ Create the token set for the given table 
    :param table: a list of list (row-view) of the table content 
    :param mode: how to create the token set, with "set" or "bag" semantic
    :param numeric_columns: a flag vector, where if the ith element is 1, this means that the 
                            ith column is numeric and its elements are skipped while creating the token set
    :param encode: if set, tokens will be encoded as specified (e.g. 'utf-8')
    :param blacklist: a set of tokens that won't be considered
    """
    if mode == 'set':
        tokens = list({prepare_token(token) for row in table for icol, token in enumerate(row) 
                     if not pd.isna(token) and token and numeric_columns[icol] == 0 and token not in blacklist})
    elif mode == 'bag':
        counter = defaultdict(int)
        
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        
        tokens = [_create_token_tag(prepare_token(token)) for row in table for icol, token in enumerate(row)
                if not pd.isna(token) and token and numeric_columns[icol] == 0 and token not in blacklist]
    else:
        raise Exception('Unknown mode: ' + str(mode))
    return tokens if not encode else [token.encode(encode) for token in tokens]



def apply_sloth(table1, table2, numeric_columns1, numeric_columns2, verbose=False, blacklist=[]) -> tuple[int, float]:
    num_null = 0

    def format_value_for_excluding_nan(t):
        nonlocal num_null
        if not t or pd.isna(t) or t in blacklist:
            num_null += 1
            return f'{t}@{num_null}'
        t = prepare_token(t)
        return t
    
    table1 = [[format_value_for_excluding_nan(row[i]) for row in table1] for i in range(len(table1[0])) if numeric_columns1[i] == 0]
    table2 = [[format_value_for_excluding_nan(row[i]) for row in table2] for i in range(len(table2[0])) if numeric_columns2[i] == 0]

    metrics = []
    start_sloth = time.time()
    try:
        _, metrics = sloth(table1, table2, metrics=metrics, verbose=verbose)
        tot_sloth_time = round(time.time() - start_sloth, 3)
        largest_ov_sloth = metrics[-2]
        return largest_ov_sloth, tot_sloth_time
    except TimeoutError:
        return -1, round(time.time() - start_sloth, 3)



def sample_query_task(data):
    chunk, data_lake_args = data[0], data[1:]
    dlh = SimpleDataLakeHelper(*data_lake_args)
    s = set()
    for table_id in chunk:
        table_obj = dlh.get_table_by_numeric_id(table_id)
        if not is_valid_table(table_obj['content'], table_obj['numeric_columns']):
            continue
        
        s.add(table_id)
    return s


def sample_queries(output_query_json, nsamples, num_cpu, *data_lake_args):
    s = set()
    dlh = SimpleDataLakeHelper(*data_lake_args)
    N = dlh.get_number_of_tables()
    dlh.close()
    
    print(f'Sampling {nsamples} tables from {N} total tables...')

    with mp.Pool(num_cpu) as pool:
        while len(s) < nsamples: 
            work = random.sample(range(N), nsamples - len(s))
            chunk_size = max((nsamples - len(s)) // num_cpu, 1)
            results = pool.map(sample_query_task, chunks(work, chunk_size, *data_lake_args))
            for taskres in results:
                for x in taskres:
                    s.add(int(x))
            print(f'Sampled {len(s)} ({round(len(s) * 100 / nsamples)}%)', end='\r')
    samples = {'_id_numeric': list(s)[:nsamples]}
    
    with open(output_query_json, 'w') as wf:
        json.dump(samples, wf, indent=1)
    return len(samples['_id_numeric'])


def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return json.load(fr)['_id_numeric']


def logging_setup(logfile):
    logger = logging.getLogger('TestLog')
    
    if not logger.handlers:
        handler_sysout = logging.StreamHandler(sys.stdout)
        handler_file = logging.FileHandler(logfile)
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        handler_sysout.setFormatter(formatter)
        handler_file.setFormatter(formatter)

        logger.addHandler(handler_sysout)
        logger.addHandler(handler_file)

        logger.setLevel(logging.INFO)
        logger.propagate = False