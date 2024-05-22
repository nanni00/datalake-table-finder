import json
import os
import re
import sys
import mmh3
import pymongo.collection
import spacy
import random

import pymongo
import binascii
import jsonlines
import numpy as np
import pandas as pd
import polars as pl
from pandas.api.types import is_numeric_dtype
from typing import Literal
from tqdm import tqdm
from collections import defaultdict

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
    
from tools.utils.settings import DefaultPath
from tools.utils.utils import print_info, my_tokenizer

import multiprocessing as mp
import jenkspy

_TOKEN_TAG_SEPARATOR = '@#'



"""
def _infer_column_type(column: list, check_column_threshold:int=3, nlp=None|spacy.Language) -> str:
    NUMERICAL_NER_TAGS = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}
    # column = set(column)
    # if '' in column: column.remove('') # this causes an exception whenever a NA is in the set, but is this passage really useful or correct?
    idxs = range(0, len(column) - 1) if len(column) <= check_column_threshold else random.sample(range(0, len(column)), check_column_threshold)
    
    parsed_values = nlp.pipe([str(column[i]) for i in idxs])
    ner_tags = {token.ent_type_ for cell in parsed_values for token in cell}
    
    rv = sum(1 if tag in NUMERICAL_NER_TAGS else -1 for tag in ner_tags)
    return 'real' if rv > 0 else 'text'
    

def _create_set_with_inferring_column_dtype(df: pd.DataFrame, check_column_threshold:int=3, nlp=None|spacy.Language):
    tokens_set = set()
    for i in range(len(df.columns)):
        unique_values = df.iloc[:, i].unique()
        if is_numeric_dtype(unique_values) or _infer_column_type(unique_values.tolist(), check_column_threshold, nlp) == 'real':
            continue
        for token in unique_values:
            if not pd.isna(token) and token: # discard NA values and empty strings/values
                tokens_set.add(token)
    return tokens_set


def _create_set_with_my_tokenizer(df: pd.DataFrame):
    tokens_set = set()
    for i in range(len(df.columns)):
        if is_numeric_dtype(df.iloc[:, i]):
            continue
        for token in df.iloc[:, i].unique():
            for t in my_tokenizer(token, remove_numbers=True):
                tokens_set.add(t.lower())

    return tokens_set
"""


def _create_token_set(data, mode):
    if mode == 'set':
        return list({str(token).replace('|', ' ') for column in data for token in column if not pd.isna(token) and token})
    elif mode == 'bag':
        counter = defaultdict(int) # is that better? More space but no sort operation            
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        return [_create_token_tag(str(token).replace('|', ' ')) for column in data for token in column if not pd.isna(token) and token]
    else:
        raise Exception('Unknown mode: ' + str(mode))



@print_info(msg_before="Extracting tables from JSONL file...")
def extract_tables_from_jsonl_to_mongodb(
        input_tables_jsonl_file,
        table_collection:pymongo.collection.Collection,
        milestone=1000,
        ntotal_tables=570171
        ):
    tables = list()  # empty list to store the parsed tables

    with jsonlines.open(input_tables_jsonl_file) as reader:
        for counter, raw_table in tqdm(enumerate(reader, start=1), total=ntotal_tables):
            if counter % milestone == 0:
                # print(counter, end='\r')
                pass
            raw_table_content = raw_table["tableData"]  # load the table content in its raw form
            table = dict()  # empty dictionary to store the parsed table
            table["_id"] = raw_table["_id"]  # unique identifier of the table
            table["headers"] = raw_table["tableHeaders"][0]
            table["context"] = raw_table["pgTitle"] + " | " + raw_table["sectionTitle"] + " | " + raw_table["tableCaption"]  # table context            
            table["content"] = [[cell["text"] for cell in row] for row in raw_table_content]  # table content (only cell text)                
            tables.append(table)  # append the parsed table to the list
        table_collection.insert_many(tables)



def parallel_extract_starting_sets_from_tables(input_tables_csv_dir, 
                                      final_set_file,
                                      id_table_file,
                                      tables_stat_file,
                                      columns_stat_file,
                                      ntables_to_load_as_set=10, 
                                      mode:Literal['mytok', 'infer', 'set', 'bag']='set'):
    set_q, id_metadata_q, table_stat_q, column_stat_q = mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()

    def _job(ids, offset):
        for i, id_table in enumerate(ids, start=offset):
            # read the csv
            table = pd.read_csv(input_tables_csv_dir + f'/{id_table}')
            
            table_set = _create_token_set(table, mode)
                        
            # the table set
            set_q.put(f"{i + offset}|{'|'.join(table_set)}\n")      #### !!! Here we use "|", not comma ","
            
            # original table id - integer id
            id_metadata_q.put(f'{id_table},{i + offset}\n')

            # table statistics            
            table_size = table.shape[0] * table.shape[1]
            num_na = table.isna().sum().sum()
            table_stat_q.put(f"{','.join(map(str, [i + offset, table_size, len(table_set), num_na, round(num_na / table_size, 5)]))}\n")

            # columns statistics
            for id_col in range(len(table.columns)):
                num_distinct = table.iloc[:, id_col].unique().shape[0]
                num_na = table.iloc[:, id_col].isna().sum()
                column_stat_q.put(f"{','.join(map(str, [i + offset, id_col, table.shape[1], num_distinct, num_na, round(num_na / table.shape[0], 5)]))}\n")
        
        # print(f'Worker {os.getpid()} completed')
        set_q.put(f'Process {os.getpid()} completed')
        id_metadata_q.put(f'Process {os.getpid()} completed')
        table_stat_q.put(f'Process {os.getpid()} completed')
        column_stat_q.put(f'Process {os.getpid()} completed')
    
    def _writer_job(fname, queue:mp.Queue):
        with open(fname, 'w') as fhandle:
            okstop = 0
            while True:
                try:
                    rec = queue.get(block=False)
                    if re.match(r'Process \d+ completed', rec):
                        okstop += 1
                        if okstop == os.cpu_count():
                            break
                    else:
                        # no need to batch here, there's already a inner IO buffer           
                        fhandle.write(rec)                    
                except: pass

    id_metadata_q.put('sloth_id,josie_id\n')
    table_stat_q.put('id_table,size,set_size,num_nan,%nan\n')
    column_stat_q.put('id_table,id_column,size,num_distincts,num_nan,%nan\n')

    ids = os.listdir(input_tables_csv_dir)
    chuncksize = ntables_to_load_as_set // os.cpu_count()
    worker_pool = []
    writer_pool = []
    print(f'Table extraction: n={ntables_to_load_as_set}, nproc={os.cpu_count()}, chunksize={chuncksize}')
    for i in range(os.cpu_count()):
        worker_pool.append(mp.Process(name=str(i), target=_job, kwargs={'ids':ids[i * chuncksize:(i + 1) * chuncksize], 'offset':i * chuncksize}))

    for f, q in zip( 
        [final_set_file,    id_table_file, tables_stat_file,    columns_stat_file], 
        [set_q,             id_metadata_q, table_stat_q,        column_stat_q]):
        writer_pool.append(mp.Process(name=str(i), target=_writer_job, kwargs={'fname':f, 'queue':q}))

    print('Starting workers...')
    for p in worker_pool: p.start()
    print('Starting writers...')
    for p in writer_pool: p.start()
    print('Joining workers...')
    for p in worker_pool: p.join()
    print('Joining writers...')
    for p in writer_pool: p.join()

    print('Emptying queues...')
    for q in [set_q, table_stat_q, column_stat_q, id_metadata_q]: 
        while not q.empty():
            q.get()
        q.close()


@print_info(msg_before='Creating raw tokens...', msg_after='Completed.')
def create_raw_tokens(
    input_set_file, 
    output_raw_tokens_file, 
    single_txt=False, 
    spark_context=None):

    if not spark_context:
        conf = pyspark.SparkConf() \
            .setAppName('CreateIndex') \
                # .set('spark.executor.memory', '100g') \
                # .set('spark.driver.memory', '5g')
        spark_context = pyspark.SparkContext(conf=conf)
    
    skip_tokens = set()

    sets = spark_context \
        .textFile(input_set_file) \
            .map(
                lambda line: line.split('|')
            ) \
                .map(
                    lambda line: (
                        int(line[0]),
                        [token for token in line[1:] if token not in skip_tokens]
                    )
                )

    if single_txt:
        sets = sets \
            .flatMap(
                lambda sid_tokens: \
                    [
                        (token, sid_tokens[0]) 
                        for token in sid_tokens[1]
                    ]
                ) \
                    .map(
                        lambda token_sid: f'{token_sid[0]} {token_sid[1]}\n'
                    ).collect()
        with open(output_raw_tokens_file, 'w') as f:
            f.writelines(sets)
    else:
        sets \
            .flatMap(
                lambda sid_tokens: 
                    [
                        (token, sid_tokens[0]) 
                        for token in sid_tokens[1]
                    ]
                ) \
                    .map(
                        lambda token_sid: f'{token_sid[0]} {token_sid[1]}'
                    ).saveAsTextFile(output_raw_tokens_file)



@print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.', time=True)
def create_index(
    mode:str,
    original_sloth_results_file,
    output_id_for_queries_file,
    output_tables_id_file,
    output_integer_set_file, 
    output_inverted_list_file,
    thresholds:dict[str:int],
    ):

    MIN_ROW =     0       if 'min_rows'       not in thresholds else thresholds['min_rows']
    MAX_ROW =     999999  if 'max_rows'       not in thresholds else thresholds['max_rows']
    MIN_COLUMN =  0       if 'min_columns'    not in thresholds else thresholds['min_columns']
    MAX_COLUMN =  999999  if 'max_columns'    not in thresholds else thresholds['max_columns']
    MIN_AREA =    0       if 'min_area'       not in thresholds else thresholds['min_area']
    MAX_AREA =    999999  if 'max_area'       not in thresholds else thresholds['max_area']
    
    all_sloth_results = pl.scan_csv(original_sloth_results_file)
    sloth_tables_ids = set( \
        pl.concat( \
            [all_sloth_results.select('r_id').collect().to_series(), 
            all_sloth_results.select('s_id').collect().to_series()]
            ) \
                .to_list()
        )
        
    @F.udf(returnType=BooleanType())
    def check_is_in_sloth_results(col1):
        return col1 in sloth_tables_ids            

    # fine tune of executor/driver.memory?
    spark = SparkSession \
        .builder \
        .appName("mongodbtest1") \
        .master('local[*]')\
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1') \
        .config('spark.executor.memory', '8g') \
        .config('spark.driver.memory', '5g') \
        .getOrCreate()

    wikitables_df = spark.read.format("mongo") \
        .option ("uri", "mongodb://127.0.0.1:27017/optitab.wikitables") \
        .load() \
        .select('_id', 'content') \
        .filter(f"""
                size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
                AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
                AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""") \
        .filter(check_is_in_sloth_results('_id'))

    print('Saving the filtered original table IDs as CSV...')
    wikitables_df.select('_id').toPandas().to_csv(output_id_for_queries_file, index=False)    

    sets = spark.read\
        .format('mongo')\
        .option( "uri", "mongodb://127.0.0.1:27017/sloth.latest_snapshot_tables") \
        .load() \
        .select('_id', 'content') \
        .rdd \
        .map(list) \
        .union(
            wikitables_df.rdd.map(list)    
        )
    
    wikitables_df.unpersist()   # free memory used by the dataframe (is this really useful?)
    print(f'Total RDD size: {sets.count()}')
    
    sets = sets.zipWithUniqueId()

    print('Saving mapping between JOSIE and wikitables (SLOTH) IDs...')
    sets.map(lambda t: f"{t[1]},{t[0][0]}").saveAsTextFile(output_tables_id_file)

    def prepare_tuple(t, mode):
        return [t[1], _create_token_set(t[0][1], mode)]    
    
    print('Start creating inverted index and integer sets...')
    token_sets = sets.map(lambda t: prepare_tuple(t, mode)) \
        .flatMap(
            lambda sid_tokens: \
                [
                    (token, sid_tokens[0]) 
                    for token in sid_tokens[1]
                ]
        )

    posting_lists_sorted = token_sets \
        .groupByKey() \
            .map(
                # t: (token, setIDs)
                lambda t: (t[0], sorted(list(t[1])))
            ) \
                .map(
                    # t: (token, setIDs)
                    lambda t: \
                        (
                            t[0], 
                            t[1], 
                            mmh3.hash_bytes(np.array(t[1]))
                        ) # is ok?
                ) \
                    .sortBy(
                        # t: (token, setIDs, hash)
                        lambda t: (len(t[1]), t[2], t[1])
                    ) \
                        .zipWithIndex() \
                            .map(
                                # t: ((rawToken, sids, hash), tokenIndex)
                                lambda t: (t[1], (t[0][0], t[0][1], t[0][2]))
                            ) \
                                .persist(pyspark.StorageLevel.MEMORY_ONLY)

    def equal_arrays(a1, a2):
        return len(a1) == len(a2) and all(x1 == x2 for (x1, x2) in zip(a1, a2))

    # create the duplicate groups
    duplicate_group_ids = posting_lists_sorted \
        .map(
            # t: (tokenIndex, (rawToken, sids, hash))
            lambda t: (t[0] + 1, (t[1][1], t[1][2]))
        ) \
            .join(posting_lists_sorted) \
                .map(
                    # if the lower and upper posting lists are different, then the upper posting list
                    # belong to a new group, and the upper token index is the new group's
                    # starting index
                    # (tokenIndexUpper, ((sidsLower, hashLower), (_, sidsUpper, hashUpper)))
                    lambda t: 
                        -1 if t[1][0][1] == t[1][1][2] and equal_arrays(t[1][0][0], t[1][1][1]) else t[0]
                ) \
                    .filter(
                        # add the first group's starting index, which is 0, and then
                        # create the group IDs
                        lambda i: i > 0
                    ) \
                        .union(
                            spark.sparkContext.parallelize([0])
                        ) \
                            .sortBy(
                                lambda i: i
                            ) \
                                .zipWithIndex() \
                                    .map(
                                        # returns a mapping from group ID to the
                                        # starting index of the group
                                        # (startingIndex, GroupID)
                                        lambda t: (t[1], t[0])
                                    )

    # generating all token indexes of each group
    token_group_ids = duplicate_group_ids \
        .join(
            duplicate_group_ids \
                .map(
                    # (GroupID, startingIndexUpper)
                    lambda t: (t[0] - 1, t[1])
                )
            ) \
                .flatMap(
                    # GroupID, (startingIndexLower, startingIndexUpper)
                    lambda t: map( 
                            lambda token_index: (token_index, t[0]), range(t[1][0], t[1][1])
                        )
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)

    # join posting lists with their duplicate group IDs
    posting_lists_with_group_ids = posting_lists_sorted \
        .join(
            token_group_ids
        ) \
            .map(
                # (tokenIndex, ((rawToken, sids, _), gid))
                lambda t: (t[0], (t[1][1], t[1][0][0], t[1][0][1]))
            )

    # STAGE 2: CREATE INTEGER SETS
    # Create sets and replace text tokens with token index
    integer_sets = posting_lists_with_group_ids \
        .flatMap(
            # (tokenIndex, (_, _, sids))
            lambda t: [(sid, t[0]) for sid in t[1][2]]        
        ) \
            .groupByKey() \
                .map(
                    # (sid, tokenIndexes)
                    lambda t: (
                        t[0], 
                        sorted(t[1])
                    )
                )

    def sets_format_string(t):
        sid, indices = t
        return "{}|{}|{}|{{{}}}".format(sid, len(indices), len(indices), ','.join(map(str, indices)))

    def postlist_format_string(t):
        token, raw_token, gid, sets = t
        freq = len(sets)
        set_ids = ','.join([str(s[0]) for s in sets])
        set_sizes = ','.join([str(s[1]) for s in sets])
        set_pos = ','.join([str(s[2]) for s in sets])

        return "{}|{}|{}|{}|{{{}}}|{{{}}}|{{{}}}|{}" \
            .format(
                token, freq, gid, 1, set_ids, set_sizes, set_pos, binascii.hexlify(bytes(str(raw_token), 'utf-8'))
            )
    
    # STAGE 3: CREATE THE FINAL POSTING LISTS
    # Create new posting lists and join the previous inverted
    # lists to obtain the final posting lists with all the information
    posting_lists = integer_sets \
        .flatMap(
            # (sid, tokens)
            lambda t:
                [
                    (token, (t[0], len(t[1]), pos))
                    for pos, token in enumerate(t[1])
                ]
        ) \
            .groupByKey() \
                .map(
                    # (token, sets)
                    lambda t: (
                        t[0], 
                        sorted(t[1], 
                            key=lambda s: s[0]
                            )
                    )
                ) \
                    .join(posting_lists_with_group_ids) \
                        .map(
                            # (token, (sets, (gid, rawToken, _))) -> (token, rawToken, gid, sets)
                            lambda t: (t[0], t[1][1][1], t[1][1][0], t[1][0])
                        )
    
    # STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS
    # print('STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS')
    # to load directly into the database
    integer_sets.map(
        lambda t: sets_format_string(t)
    ).saveAsTextFile(output_integer_set_file)

    posting_lists.map(
        lambda t: postlist_format_string(t)
    ).saveAsTextFile(output_inverted_list_file)


def format_spark_set_file(input_file:str, output_formatted_file:str, on_inverted_index=True):
    """ Used when using JAR with scala code instead of the python version of CreateIndex/CreateRawTokens """
    def sets_format_string(t):
        sid, indices = t[0], t[1:]
        return "{}|{}|{}|{{{}}}\n".format(sid, len(indices), len(indices), ','.join(indices))

    def postlist_format_string(t):
        token, raw_token, gid, sets = t[0], t[1], t[2], t[3:]
        sets = list(map(eval, sets))
        freq = len(sets)
        set_ids =   ','.join([str(s[0]) for s in sets])
        set_sizes = ','.join([str(s[1]) for s in sets])
        set_pos =   ','.join([str(s[2]) for s in sets])

        return "{}|{}|{}|{}|{{{}}}|{{{}}}|{{{}}}|{}\n" \
            .format(
                token, freq, gid, 1, set_ids, set_sizes, set_pos, binascii.hexlify(bytes(str(raw_token), 'utf-8'))
            )

    with open(output_formatted_file, 'w') as writer:
        for fpart in sorted(os.listdir(input_file)):
            if not fpart.startswith('part-'):
                continue

            with open(input_file + '/' + fpart) as reader:
                # print(f'opening file {input_file + "/" + fpart}...')
                if on_inverted_index:
                    writer.writelines(map(lambda line: postlist_format_string(line.strip().split('|')), reader.readlines()))
                else:
                    writer.writelines(map(lambda line: sets_format_string(line.strip().split('|')), reader.readlines()))



def get_tables_statistics_from_mongodb(
        wikitables_coll:pymongo.collection.Collection, 
        table_stat_file):
    data = []
    for document in tqdm(wikitables_coll.find(), total=wikitables_coll.estimated_document_count()):
        _id, content = document.get('_id'), document.get('content')
        rows = len(content)
        columns = len(content[0])
        size = rows * columns
        distincts = len({token for row in content for token in row})
        nan = len([token for row in content for token in row if not token or pd.isna(token)])
        nan_p = nan / size
        
        data.append([_id, rows, columns, size, distincts, nan, nan_p])
    stat = pl.DataFrame(data, schema=['tab_id', 'rows', 'cols', 'size', 'distincts', 'nan', '%nan'])
    stat.write_csv(table_stat_file)


def sample_queries(
    sloth_results_csv_file,
    josie_sloth_id_file,
    table_statistics_file,
    output_query_json
    ):

    sloth_res =     pl.scan_csv(sloth_results_csv_file)
    table_stat =    pl.read_csv(table_statistics_file)
    josloth_ids =   pl.read_csv(
        josie_sloth_id_file, 
        has_header=False,
        dtypes={'column_1': pl.datatypes.Int64, 'column_2': pl.datatypes.String}
        ) \
        .rename({'column_1': 'josie_id', 'column_2': 'sloth_id'})

    n_sample_per_interval = 10
    jsim_intervals = [0, 0.3, 0.7, 1]
    overlap_intervals = [50, 100, 5000]
    nan_threshold = 0.1

    sloth_res_samples = defaultdict(list)
    ids = {}

    def lookup_for_josie_int_id(sloth_str_id:str):
        return josloth_ids.row(by_predicate=pl.col('sloth_id') == sloth_str_id)[0]

    table_stat = table_stat.rename({'tab_id': 'r_id'})
    for jsim_min, jsim_max in zip(jsim_intervals[:-1], jsim_intervals[1:]):
        sample = sloth_res.filter((jsim_min < pl.col('jsim')) & (pl.col('jsim') < jsim_max)).collect().sample(n_sample_per_interval)
        sample = sample.join(table_stat, on='r_id', how='left', suffix='_r')
        
        table_stat = table_stat.rename({'r_id': 's_id'})
        sample = sample.join(table_stat, on='s_id', how='left', suffix='_s')
        table_stat = table_stat.rename({'s_id': 'r_id'})

        sample = sample.rename({'rows': 'rows_r', 'cols': 'cols_r', 'size': 'tabsize_r', 'nan': 'nan_r', 'distincts': 'distincts_r', '%nan': '%nan_r'})
        sample = sample.filter((pl.col('%nan_r') < nan_threshold) & (pl.col('%nan_s') < nan_threshold))
        sample = sample.sort(by='overlap_area')
        sloth_res_samples[(jsim_min, jsim_max)] = sample.to_numpy().tolist()
        ids[(jsim_min, jsim_max)] = list(map(lookup_for_josie_int_id, sample.select(['r_id', 's_id']).to_numpy().flatten().tolist()))

    with open(output_query_json, 'w') as writer:            
        json.dump(
            [
                {
                    "jsim": (jsim_min, jsim_max),
                    "sloth_res": sloth_res_samples[(jsim_min, jsim_max)],
                    "ids": ids[(jsim_min, jsim_max)]
                } 
                for jsim_min, jsim_max in zip(jsim_intervals[:-1], jsim_intervals[1:])
            ],
            writer, 
            indent=4
            )
        
    totsamples = [i for _, v in ids.items() for i in v]
    print(f"Sampled {len(totsamples)} IDs from {len(jsim_intervals) - 1} intervals of Jaccard similarity.")
    return totsamples



if __name__ == '__main__':
    format_spark_set_file(
        'data/josie-tests/n45673-mset/scala-tables.set-2',
        'data/josie-tests/n45673-mset/scala-tables-formatted.set-2',
        False
    )

    format_spark_set_file(
        'data/josie-tests/n45673-mset/scala-tables.inverted-list',
        'data/josie-tests/n45673-mset/scala-tables-formatted.inverted-list'
    )