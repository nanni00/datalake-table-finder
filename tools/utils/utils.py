from collections import defaultdict
import json
import re
import time
import numpy as np
import pandas as pd
import pymongo

from tools.sloth.sloth import sloth


def print_info(**dec_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            on_line = 'on_line' in dec_kwargs and dec_kwargs['on_line'] == True
            end_line = '\r' if on_line else '\n'
            
            msg_before = dec_kwargs['msg_before'] if 'msg_before' in dec_kwargs else '' 
            msg_after = dec_kwargs['msg_after'] if 'msg_after' in dec_kwargs else ''
            msg_after = msg_after if not on_line else f'{msg_before} {msg_after}' if msg_before else msg_after 
            
            if msg_before: 
                print(msg_before, end=end_line)
            start = time.time()
            results = func(*args, **kwargs)            
            end = time.time()

            if 'time' in dec_kwargs: 
                print(f'Elapsed time: {round(end - start, 3)}s')
            if msg_after:
                print(msg_after)
            return results
        return wrapper
    return decorator


def round_to(n, precision):
    if n >= 0 or n < 0:
        correction = 0.5 if n >= 0 else -0.5
        return int(n / precision + correction) * precision
    else:
        return n


def round_to_05(n):
    return float(format(round_to(n, 0.05), ".2f"))


def my_tokenizer(s: str, remove_numbers=False):
    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words('english'))

    s = str(s)
    if not remove_numbers:
        return [            
            x for x in re.findall(r'\b([a-zA-Z]+|\d{1}|\d{2}|\d{3}|\d{4})\b', s) 
            if x not in stopwords_set
        ]
    else:
        return [
            x for x in re.findall(r'[a-zA-Z]+', s)
            if x not in stopwords_set
        ]


def cosine_similarity(arr1:np.array, arr2:np.array):
    return np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))


def get_int_from_(s: str):
    return [int(x) for x in re.findall(r'\d+', s)]
    

def get_current_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")


def get_mongodb_collections(small=True):
    mongoclient = pymongo.MongoClient()

    if not small:
        wikitables_coll = mongoclient.optitab.turl_training_set
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables
    else:
        wikitables_coll = mongoclient.optitab.turl_training_set_small
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables_small

    return mongoclient, [wikitables_coll, snapshot_coll]



def get_one_document_from_mongodb_by_key(key, value, *collections):
    for collection in collections:
        document = collection.find_one({key: value})
        if document:
            return document


def check_table_is_in_thresholds(content, table_thresholds):
    return table_thresholds['min_row'] <= len(content) <= table_thresholds['max_row'] and \
        table_thresholds['min_column'] <= len(content[0]) <= table_thresholds['max_column'] and \
        table_thresholds['min_area'] <= len(content) * len(content[0]) <= table_thresholds['max_area']



_TOKEN_TAG_SEPARATOR = '@#'


def _create_token_set(table, mode, numeric_columns, encode=None):
    """ Create the token set for the given table 
    :param table: a list of list (row-view) of the table content 
    :param mode: how to create the token set, with "set" or "bag" semantic
    :param numeric_columns: a flag vector, where if the ith element is 1, this means that the 
                            ith column is numeric and its elements are skipped while creating the token set
    :param encode: if set, tokens will be encoded as specified (e.g. 'utf-8')
    """
    def prepare_token(token):
        return str(token).replace('|', ' ').replace('\n', ' ')

    if mode == 'set':
        tokens = list({prepare_token(token) for row in table for icol, token in enumerate(row) 
                     if not pd.isna(token) and token and numeric_columns[icol] == 0})
    elif mode == 'bag':
        counter = defaultdict(int) # is that better? More space but no sort operation            
        
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        
        tokens = [_create_token_tag(prepare_token(token)) for row in table for icol, token in enumerate(row)
                if not pd.isna(token) and token and numeric_columns[icol] == 0]
    else:
        raise Exception('Unknown mode: ' + str(mode))
    return tokens if not encode else [token.encode('utf-8') for token in tokens]




def apply_sloth(table1, table2, numeric_columns1, numeric_columns2):
    num_null = 0

    def format_value_for_excluding_nan(t):
        nonlocal num_null 
        if not t or pd.isna(t):
            num_null += 1
            return f'{t}@{num_null}'
        return t
    
    table1 = [[format_value_for_excluding_nan(row[i]) for row in table1] for i in range(len(table1[0])) if numeric_columns1[i] == 0]
    table2 = [[format_value_for_excluding_nan(row[i]) for row in table2] for i in range(len(table2[0])) if numeric_columns2[i] == 0]

    metrics = []
    _, metrics = sloth(table1, table2, verbose=False, metrics=metrics)
    largest_ov_sloth = metrics[-2]
    return largest_ov_sloth




def sample_queries(
    output_query_json,
    nsamples,
    tables_thresholds:dict[str, int],
    *collections
    ):

    samples = [collection.aggregate([{"$sample": {"size": nsamples // 2}}]) for collection in collections]

    samples = [
        {'_id': t['_id'], '_id_numeric': t['_id_numeric'], 'numeric_columns': t['numeric_columns']} 
        for s in samples for t in list(s)
        if check_table_is_in_thresholds(t['content'], tables_thresholds) \
        and not all(t['numeric_columns'])
    ]

    print(f'Sampled {len(samples)} tables')
    with open(output_query_json, 'w') as wf:
        json.dump(
            samples,
            wf,
            indent=1
        )
    return len(samples)


def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return [t['_id_numeric'] for t in json.load(fr)]




