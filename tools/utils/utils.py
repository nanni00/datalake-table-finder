from abc import ABC, abstractmethod
from collections import defaultdict
import json
import re
import time
import numpy as np
import pandas as pd
import pymongo

import pyspark.rdd
from pyspark.sql import SparkSession

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
            result = func(*args, **kwargs)            
            end = time.time()

            if 'time' in dec_kwargs: 
                print(f'Elapsed time: {round(end - start, 3)}s')
            if msg_after:
                print(msg_after)
            return result
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
    

def get_local_time():
    return time.strftime("%Y/%m/%d %H:%M:%S")


def convert_to_giga(x):
    if x.endswith('MB'):
        return int(re.match(r'\d+', x).group()) / 1024
    elif x.endswith('KB'):
        return int(re.match(r'\d+', x).group()) / (1024 ** 2)




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





class AlgorithmTester(ABC):
    def __init__(self, mode, small, tables_thresholds, num_cpu) -> None:
        self.mode = mode
        self.small = small
        self.num_cpu = num_cpu
        self.tables_thresholds = tables_thresholds

    @abstractmethod
    def data_preparation(self) -> None:
        pass
    
    @abstractmethod
    def query(self, results_file, k, query_ids, *args) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass




def get_tables_thresholds_from(tables_thresholds):
    return tables_thresholds['min_row'], tables_thresholds['max_row'], tables_thresholds['min_column'], \
        tables_thresholds['max_column'], tables_thresholds['min_area'], tables_thresholds['max_area']


def get_initial_spark_rdd(small, num_cpu, spark_jars_packages, tables_thresholds):
        MIN_ROW, MAX_ROW, MIN_COLUMN, MAX_COLUMN, MIN_AREA, MAX_AREA = get_tables_thresholds_from(tables_thresholds)

        # fine tune of executor/driver.memory?
        builder = SparkSession.Builder()
        spark = (
            builder
            .appName("Big Bang Testing with MongoDB")
            .master(f"local[{num_cpu}]")
            .config('spark.jars.packages', ','.join(spark_jars_packages))
            .config('spark.executor.memory', '100g')
            .config('spark.driver.memory', '10g')
            .config('spark.local.dir', '/data4/nanni/spark')
            .getOrCreate()
        )

        # adjusting logging level to error, avoiding warnings
        spark.sparkContext.setLogLevel("ERROR")

        db_collections = zip(['optitab', 'sloth'], 
                             ["turl_training_set_small" if small else "turl_training_set", 
                              "latest_snapshot_tables_small" if small else "latest_snapshot_tables"])

        initial_rdd = spark.sparkContext.emptyRDD()

        for database, collection_name in db_collections:
            initial_rdd = initial_rdd.union(
                spark 
                .read 
                .format("mongodb") 
                .option ("uri", "mongodb://127.0.0.1:27017/") 
                .option("database", database) 
                .option("collection", collection_name) 
                .load() 
                .select('_id_numeric', 'content', 'numeric_columns') 
                .filter(f"""
                    size(content) BETWEEN {MIN_ROW} AND {MAX_ROW}
                    AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN}
                    AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}"""
                )
                .rdd
                .map(list)
            )

        return spark, initial_rdd





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

    s = set()
    while len(s) < nsamples:
        samples = [collection.aggregate([{"$sample": {"size": nsamples}}]) for collection in collections]
        for p in samples:
            for t in list(p):
                if not check_table_is_in_thresholds(t['content'], tables_thresholds) or all(t['numeric_columns']):
                    continue
                s.add(t['_id_numeric'])
                if len(s) >= nsamples:
                    break            
            
        # s = [
        #         t['_id_numeric']
        #         for s in samples for t in list(s)
        #         if  \
        #         and not all(t['numeric_columns'])
        #     ]
        

    samples = {
        '_id_numeric': list(s)[:nsamples]
    }
    
    with open(output_query_json, 'w') as wf:
        json.dump(
            samples,
            wf,
            indent=1
        )
    return len(samples['_id_numeric'])



def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return json.load(fr)['_id_numeric']

