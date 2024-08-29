import re
import sys
import time
import json
import random
import logging
import logging.handlers
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from pyspark.sql import SparkSession

from tools.sloth.sloth import sloth
from tools.sloth.utils import parse_table
from tools.utils.parallel_worker import chunks
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.basicconfig import TablesThresholds as tab_thresh


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


def is_valid_table(table, numeric_columns):
    if all(numeric_columns):
        return False
    table = [[row[i] for i, x in enumerate(numeric_columns) if x == 0] for row in table]
    # return check_table_is_in_thresholds(table, tables_thresholds)
    return  tab_thresh.MIN_ROWS     <= len(table)                   <= tab_thresh.MAX_ROWS and \
            tab_thresh.MIN_COLUMNS  <= len(table[0])                <= tab_thresh.MAX_COLUMNS and \
            tab_thresh.MIN_AREA     <= len(table) * len(table[0])   <= tab_thresh.MAX_AREA
    
    
def naive_detect_table_numeric_and_null_columns(table: list[list], any_int:bool=False) -> list[int]:
    """ 
    :param table: a list of lists representing a table in row view
    :param any_int: if set to True, a column is detected as an numeric column wheter any of its values is 
        a numeric value, else if the majority (i.e. #numeric_cells >= #column_size / 2 + 1) of its values are numerics
    :return a list of int, where the i-th element is set to 1 if the i-th column is detected as numeric or with only null values, 
        0 otherwise
    """

    def is_number_tryexcept(s):
        try: 
            float(s)
            return True
        except ValueError:
            return False
        except TypeError:
            return False
        
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
        .config('spark.driver.memory', '20g')
        .config('spark.local.dir', spark_local_dir)
        .config('spark.driver.maxResultSize', '12g')
        .getOrCreate()
    )

    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("WARN")
    return spark


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



def task(data):
    chunk, data_lake_args = data[0], data[1:]
    dlh = SimpleDataLakeHelper(*data_lake_args)
    s = set()
    for table_id in chunk:
        # while len(s) < nsamples:
        #    r = random.randint(0, dlh.get_number_of_tables() - 1)
        table_obj = dlh.get_table_by_numeric_id(table_id)
        if not is_valid_table(table_obj['content'], table_obj['numeric_columns']):
            continue
        
        s.add(table_id)
        # print(f'Sampled {len(s)} ({round(len(s) * 100 / nsamples)}%)', end='\r')
        # if len(s) >= nsamples:            
        #     break
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
            results = pool.map(task, chunks(work, chunk_size, *data_lake_args))
            for taskres in results:
                for x in taskres:
                    s.add(int(x))
            print(f'Sampled {len(s)} ({round(len(s) * 100 / nsamples)}%)', end='\r')

            # s = {x for taskres in results for x in taskres}
    
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