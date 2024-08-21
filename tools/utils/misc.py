import os
import random
import re
import sys
import time
import json
import logging
import logging.handlers
from collections import defaultdict

from bidict import bidict
import pymongo
import pyspark
import pyspark.rdd
import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession

from tools.sloth.sloth import sloth
from tools.sloth.utils import parse_table
from tools.utils.datalake import SimpleDataLakeHelper


def print_info(**dec_kwargs):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if 'msg_before' in dec_kwargs: logging.info(dec_kwargs['msg_before'])
            result = func(*args, **kwargs)
            if 'msg_after' in dec_kwargs: logging.info(dec_kwargs['msg_after'])
            return result
        return wrapper
    return decorator


def get_local_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")


def convert_to_giga(x):
    if x.endswith('MB'):
        return int(re.match(r'\d+', x).group()) / 1024
    elif x.endswith('KB'):
        return int(re.match(r'\d+', x).group()) / (1024 ** 2)




def check_table_is_in_thresholds(content, table_thresholds):
    return table_thresholds['min_row'] <= len(content) <= table_thresholds['max_row'] and \
        table_thresholds['min_column'] <= len(content[0]) <= table_thresholds['max_column'] and \
        table_thresholds['min_area'] <= len(content) * len(content[0]) <= table_thresholds['max_area']



def get_tables_thresholds_from(tables_thresholds):
    return (
        0 if 'min_row' not in tables_thresholds else tables_thresholds['min_row'], 
        999999 if 'max_row' not in tables_thresholds else tables_thresholds['max_row'], 
        0 if 'min_column' not in tables_thresholds else tables_thresholds['min_column'],
        999999 if 'max_column' not in tables_thresholds else tables_thresholds['max_column'], 
        0 if 'min_area' not in tables_thresholds else tables_thresholds['min_area'], 
        999999 if 'max_area' not in tables_thresholds else tables_thresholds['max_area']
    )



def naive_detect_table_numeric_and_null_columns(table: list[list], any_int:bool=False) -> list[int]:
    """ 
    :param table: a list of lists representing a table in row view
    :param any_int: if set to True, a column is detected as an numeric column wheter any of its values is 
        a numeric value, else if the majority (i.e. #numeric_cells >= #column_size / 2 + 1) of its values are numerics
    :return a list of int, where the i-th element is set to 1 if the i-th column is detected as numeric or with only null values, 
        0 otherwise
    """
    if len(table) == 0 or len(table[0]) == 0:
        return []
    
    if any_int:
        return [
            int(any(is_number_tryexcept(cell) for cell in column) or not any(cell for cell in column))
            for column in parse_table(table, len(table[0]), 0)
        ]
    else:
        return [
            int(sum(is_number_tryexcept(cell) for cell in column) >= len(column) // 2 + 1 or not any(cell for cell in column))
            for column in parse_table(table, len(table[0]), 0)
        ]


def get_spark_session(num_cpu, spark_local_dir:str, spark_jars_packages=['org.mongodb.spark:mongo-spark-connector_2.12:10.3.0']) -> SparkSession:
    # fine tune of executor/driver.memory?
    builder = SparkSession.Builder()
    spark = (
        builder
        .appName("Big Bang Testing with MongoDB")
        .master(f"local[{num_cpu}]")
        .config('spark.jars.packages', ','.join(spark_jars_packages))
        .config('spark.executor.memory', '100g')
        .config('spark.driver.memory', '10g')
        .config('spark.local.dir', spark_local_dir)
        .config('spark.driver.maxResultSize', '12g')
        .getOrCreate()
    )

    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("ERROR")
    return spark    


def is_number_tryexcept(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False
    except TypeError:
        return False


_TOKEN_TAG_SEPARATOR = '@#'

def prepare_token(token):
    return str(token).replace('|', ' ').replace('\n', ' ')


def create_token_set(table, mode, numeric_columns, encode=None, blacklist:set=set()):
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



# TODO insert tokens blacklist here
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



def sample_queries(output_query_json, nsamples, tables_thresholds:dict[str, int], dlh:SimpleDataLakeHelper):
    s = set()
    print('Sampling tables...')
    while len(s) < nsamples:
        r = random.randint(0, dlh.get_number_of_tables() - 1)
        table = dlh.get_table_by_numeric_id(r)
        if not check_table_is_in_thresholds(table['content'], tables_thresholds) or all(table['numeric_columns']):
            continue
        s.add(r)
        print(f'Sampled {len(s)} ({round(len(s) * 100 / nsamples)}%)', end='\r')
        if len(s) >= nsamples:            
            break
    
    samples = {'_id_numeric': list(s)[:nsamples]}
    
    with open(output_query_json, 'w') as wf:
        json.dump(samples, wf, indent=1)

    return len(samples['_id_numeric'])



def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return json.load(fr)['_id_numeric']




def logging_setup(logfile):
    logger = logging.getLogger('TestLog')
    handler_sysout = logging.StreamHandler(sys.stdout)
    handler_file = logging.FileHandler(logfile)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    handler_sysout.setFormatter(formatter)
    handler_file.setFormatter(formatter)

    logger.addHandler(handler_sysout)
    logger.addHandler(handler_file)

    logger.setLevel(logging.INFO)
