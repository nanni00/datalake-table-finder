from collections import Counter, defaultdict

import pandas as pd

from dltf.utils.misc import clean_string
from dltf.utils.metrics import metrics
from dltf.utils.settings import TablesThresholds as tab_thresh


_TOKEN_TAG_SEPARATOR = '@#'


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


def table_rows_to_rows(table, num_headers, num_cols, valid_columns):
    """ Keeps the row-view structure of the table, removing cells from bad columns """
    return [[row[i] for i, x in enumerate(valid_columns) if x == 1] for row in table[num_headers:]]


def are_joinable_columns(c1: list, c2:list, t:float, metric:str):
    """ Return true if the applied metric between columns c1 and c2 crosses the threshold t"""
    return int(metrics[metric](set(c1), set(c2)) >= t)


def _is_number(s):
    try: float(s)
    except (ValueError, TypeError): return False
    return True


def _is_valid_column(column:list, tokenize=lambda cell: cell):
    column = list(map(tokenize, column))
    return sum(map(lambda cell: _is_number(cell) or pd.isna(cell) or not cell, column)) <= len(column) // 2 + 1


def naive_detect_valid_columns(table: list[list], tokenize=lambda cell: cell) -> list[int]:
    """ 
    :param table: a list of lists representing a table in row view
    :return a list of int, where the i-th element is set to 1 if the i-th column is detected as valid 0 otherwise
    """
    return [] if len(table) == 0 or len(table[0]) == 0 \
        else [int(_is_valid_column(column, tokenize)) for column in table_rows_to_columns(table, 0, len(table[0]), [1] * len(table[0]))]


def table_to_tokens(table, valid_columns, mode, encode=None, blacklist:set=set(), string_translators:list|None=None):
    """ 
    Create the tokens set for the given table
    :param table: a list of list (row-view) of the table content 
    :param valid_columns: a flag vector, where if the ith element is 1, this means that the 
                            ith column is numeric and its elements are skipped while creating the token set
    :param mode: how to create the token set, with "set" or "bag" semantic
    :param encode: if set, tokens will be encoded as specified (e.g. 'utf-8')
    :param blacklist: a set of tokens that won't be considered
    """
    if mode == 'set':
        tokens = list({clean_string(token, *string_translators) for row in table for icol, token in enumerate(row) 
                     if not pd.isna(token) and token and valid_columns[icol] == 1 and clean_string(token, *string_translators) not in blacklist})
    elif mode == 'bag':
        counter = defaultdict(int)
        
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        
        tokens = [_create_token_tag(clean_string(token, *string_translators)) for row in table for icol, token in enumerate(row)
                if not pd.isna(token) and token and valid_columns[icol] == 1 and clean_string(token, *string_translators) not in blacklist]
    else:
        raise Exception('Unknown mode: ' + str(mode))
    return tokens if not encode else [token.encode(encode) for token in tokens]


def columns_to_tokens(table, valid_columns, blacklist, string_transformers):
    """
    Create the tokens sets from the given table columns
    :param table: a list of list of the table content
    :param valid_columns:
    :param blacklist:
    :param string_transformers:
    :return 
    """
    return [
        [clean_string(token, *string_transformers) for token in column if not pd.isna(token) and token and token not in blacklist]
        for is_valid, column in zip(valid_columns, table)
        if is_valid
    ]


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

