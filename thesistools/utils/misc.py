import re
import time
import json
import random
from functools import reduce
import multiprocessing as mp
from collections import defaultdict, Counter
from string import whitespace, digits, punctuation, ascii_uppercase, ascii_lowercase

import mmh3
import pandas as pd
from datasketch.minhash import MinHash

from thesistools.sloth.sloth import sloth
from thesistools.sloth.utils import parse_table
from thesistools.utils.logging_handler import info
from thesistools.utils.parallel import chunks
from thesistools.utils.datalake import DataLakeHandlerFactory
from thesistools.utils.basicconfig import TablesThresholds as tab_thresh



_TOKEN_TAG_SEPARATOR = '@#'

whitespace_translator =     str.maketrans(whitespace, ' ' * len(whitespace))
digits_translator =         str.maketrans(digits, ' ' * len(digits))
punctuation_translator =    str.maketrans(punctuation, ' ' * len(punctuation))
lowercase_translator =      str.maketrans(ascii_uppercase, ascii_lowercase)


def get_string_translator(tr):
    match tr:
        case 'whitespace':  return whitespace_translator
        case 'digits':      return digits_translator
        case 'punctuation': return punctuation_translator
        case 'lowercase':   return lowercase_translator
        case '':            raise ValueError(f'Unknown translator: {tr}')


def jaccard(X:set, Y:set):
    # set.union seems to be quite unefficient, 
    # more than computing in this way the union
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


def mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def create_minhash(iterable):
    m = MinHash()
    m.update_batch([str(i).encode() for i in iterable])
    return m


def clean_string(s, *translators):
    if len(translators) == 0:
        translators = [str.maketrans('\n|', '  ')]
    return reduce(lambda si, tr: str(si).translate(tr), translators, str(s)).strip()


def column_to_text(column, *translators, blacklist=[], max_seq_len=512):
    def most_frequent_tokens(column):
        column = [token for token in column if token not in blacklist]
        if len(column) < max_seq_len: 
            return column
        return [x[0] for x in sorted(list(Counter(column).items()), key=lambda x: x[1], reverse=True)][:max_seq_len]
    return clean_string(' '.join([str(token) for token in most_frequent_tokens(column)]), *translators)


def table_columns_to_rows(columns):
    return [row for row in zip(*columns)]

def table_rows_to_columns(table, num_headers, num_cols, valid_columns:list=[]):
    """ Restructure the table as the list of its columns, ignoring the headers """
    return [[row[i] for row in table[num_headers:]] for i in range(num_cols) if valid_columns[i] == 1]


def table_rows_to_rows(table, num_headers, num_cols, valid_columns:list=[]):
    """ Keeps the row-view structure of the table, removing cells from bad columns """
    return [[row[i] for i, x in enumerate(valid_columns) if x == 1] for row in table[num_headers:]]


def are_joinable_columns(c1: list, c2:list, t:float, metric:str):
    """ Return true if the applied metric between columns c1 and c2 crosses the threshold t"""
    return int(metrics[metric](set(c1), set(c2)) >= t)


def print_info(**dec_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'msg_before' in dec_kwargs: info(dec_kwargs['msg_before'])
            result = func(*args, **kwargs)
            if 'msg_after' in dec_kwargs: info(dec_kwargs['msg_after'])
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


def is_valid_table(table, valid_columns):
    if sum(valid_columns) < tab_thresh.MIN_COLUMNS:
        return False
    # here is kept a row-view for the table
    table = table_rows_to_rows(table, 0, len(table[0]), valid_columns)
    return  tab_thresh.MIN_ROWS     <= len(table)                   <= tab_thresh.MAX_ROWS and \
            tab_thresh.MIN_COLUMNS  <= len(table[0])                <= tab_thresh.MAX_COLUMNS and \
            tab_thresh.MIN_AREA     <= len(table) * len(table[0])   <= tab_thresh.MAX_AREA
    



def is_valid_multi_key(names, table, valid_cols, headers):
    table = table_rows_to_columns(table, 0, len(table[0]), valid_cols)
    
    valid_headers = [h for i, h in enumerate(headers) if valid_cols[i] == 1]
    multi_key_columns = [column for h, column in zip(valid_headers, table) if h in names]
    
    # sometimes the attributes aren't accepted
    if len(multi_key_columns) != len(names):
        return
    
    for column in multi_key_columns:
        if len(set(column)) == len(column):
            return

    #  if there are duplicates in the combinations of the two columns 
    if len(set(zip(*multi_key_columns))) != len(multi_key_columns[0]):
        return
    
    # the second condition is becoue sometimes there are headers like
    # ["name", "name", "surname", "age", ...]
    # and if we want the key <"name", "surname">, usually one of them is an empty column
    idxs = [i for i, h in  enumerate(headers) if h in names if valid_cols[i] == 1]
    return idxs if len(idxs) == len(names) else None




def is_number_tryexcept(s):
    try: float(s)
    except (ValueError, TypeError): return False
    return True


def is_valid_column(column:list, tokenize=lambda cell: cell):
    column = list(map(tokenize, column))
    return sum(map(lambda cell: is_number_tryexcept(cell) or pd.isna(cell) or not cell, column)) <= len(column) // 2 + 1


def naive_detect_valid_columns(table: list[list], tokenize=lambda cell: cell) -> list[int]:
    """ 
    :param table: a list of lists representing a table in row view
    :return a list of int, where the i-th element is set to 1 if the i-th column is detected as valid 0 otherwise
    """
        
    if len(table) == 0 or len(table[0]) == 0:
        return []
    return [int(is_valid_column(column, tokenize)) for column in parse_table(table, len(table[0]), 0)]



def table_to_tokens(table, mode, valid_columns, encode=None, blacklist:set=set(), string_transformers:list|None=None):
    """ Create the token set for the given table 
    :param table: a list of list (row-view) of the table content 
    :param mode: how to create the token set, with "set" or "bag" semantic
    :param valid_columns: a flag vector, where if the ith element is 1, this means that the 
                            ith column is numeric and its elements are skipped while creating the token set
    :param encode: if set, tokens will be encoded as specified (e.g. 'utf-8')
    :param blacklist: a set of tokens that won't be considered
    """
    if mode == 'set':
        tokens = list({clean_string(token, *string_transformers) for row in table for icol, token in enumerate(row) 
                     if not pd.isna(token) and token and valid_columns[icol] == 1 and token not in blacklist})
    elif mode == 'bag':
        counter = defaultdict(int)
        
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        
        tokens = [_create_token_tag(clean_string(token)) for row in table for icol, token in enumerate(row)
                if not pd.isna(token) and token and valid_columns[icol] == 1 and token not in blacklist]
    else:
        raise Exception('Unknown mode: ' + str(mode))
    return tokens if not encode else [token.encode(encode) for token in tokens]



def largest_overlap_sloth(table1, table2, valid_cols1, valid_cols2, verbose=False, blacklist=[], **sloth_args) -> tuple[int, float]:
    num_null = 0

    def format_value_for_excluding_nan(t):
        nonlocal num_null
        if not t or pd.isna(t) or t in blacklist:
            num_null += 1
            return f'{t}@{num_null}'
        t = clean_string(t)
        return t
    
    table1 = [[format_value_for_excluding_nan(row[i]) for row in table1] for i in range(len(table1[0])) if valid_cols1[i] == 1]
    table2 = [[format_value_for_excluding_nan(row[i]) for row in table2] for i in range(len(table2[0])) if valid_cols2[i] == 1]

    metrics = []
    start_sloth = time.time()
    try:
        results, metrics = sloth(table1, table2, metrics=metrics, verbose=verbose, **sloth_args)
        if results == []:
            return -2, round(time.time() - start_sloth, 3)
        largest_ov_sloth = metrics[-2]
        return largest_ov_sloth, round(time.time() - start_sloth, 3)
    except TimeoutError:
        return -1, round(time.time() - start_sloth, 3)
    except IndexError:        
        return -2, round(time.time() - start_sloth, 3)




def sample_query_task(data):
    chunk, data_lake_args = data[0], data[1:]
    dlh = DataLakeHandlerFactory.create_handler(*data_lake_args)
    s = set()
    for table_id in chunk:
        table_obj = dlh.get_table_by_numeric_id(table_id)
        if not is_valid_table(table_obj['content'], table_obj['valid_columns']):
            continue
        
        s.add(table_id)
    dlh.close()
    return s



def sample_queries(output_query_json, nsamples, num_cpu, *data_lake_args):
    s = set()
    start = time.time()
    dlh = DataLakeHandlerFactory.create_handler(*data_lake_args)
    N = dlh.count_tables()
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
            if time.time() - start > 3:
                break
    samples = {'_id_numeric': list(s)[:nsamples]}
    
    with open(output_query_json, 'w') as wf:
        json.dump(samples, wf, indent=1)
    return len(samples['_id_numeric'])


def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return sorted(json.load(fr)['_id_numeric'])
