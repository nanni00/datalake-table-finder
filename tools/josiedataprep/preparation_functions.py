import os
import re
import mmh3
import time
import spacy
import random
import pyspark
import binascii
import numpy as np
import pandas as pd
import polars as pl
from pandas.api.types import is_numeric_dtype
from typing import Literal
from tqdm import tqdm

from tools.utils.settings import DefaultPath
from tools.utils.utils import print_info, my_tokenizer

import multiprocessing as mp


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


def _create_set_with_all_tokens(df, bag:bool=False):
    if not bag:
        if type(df) == pd.DataFrame:
            return {str(token) for i in range(len(df.columns)) for token in df.iloc[:, i].unique() if not pd.isna(token) and token}
        elif type(df) == pl.DataFrame:
            pass



def parallel_extract_starting_sets_from_tables(input_tables_csv_dir, 
                                      final_set_file,
                                      id_table_file,
                                      tables_stat_file,
                                      columns_stat_file,
                                      ntables_to_load_as_set=10, 
                                      with_:Literal['mytok', 'infer', 'set', 'bag']='set',
                                      **kwargs):
    set_q, id_metadata_q, table_stat_q, column_stat_q = mp.Queue(), mp.Queue(), mp.Queue(), mp.Queue()

    def _job(ids, offset):
        if with_ == 'infer':
            nlp = spacy.load('en_core_web_sm')
        
        for i, id_table in enumerate(ids, start=offset):
            # read the csv
            table = pd.read_csv(input_tables_csv_dir + f'/{id_table}').convert_dtypes()
            
            # extract the token set
            if with_ == 'mytok':
                table_set = _create_set_with_my_tokenizer(table)
            elif with_ == 'infer':
                table_set = _create_set_with_inferring_column_dtype(table, nlp=nlp, **kwargs)
            elif with_ == 'set':
                table_set = _create_set_with_all_tokens(table, bag=False)
            elif with_ == 'bag':
                table_set = _create_set_with_all_tokens(table, bag=True)
            else:
                raise AttributeError("Parameter with_ must be a value in {'mytok', 'infer', 'set', 'bag'}")
            
            # the table set
            set_q.put(f"{i + offset},{','.join(table_set)}\n")
            
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
                column_stat_q.put(f"{','.join(map(str, [i + offset, id_col, table.shape[1], num_distinct, num_na, round(num_na / table.shape[1], 5)]))}\n")
        
        print(f'Worker {os.getpid()} completed')
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
                    fhandle.write(rec)                    
                except: pass

    id_metadata_q.put('sloth_id,josie_id')
    table_stat_q.put('id_table,size,set_size,num_nan,%nan')
    column_stat_q.put('id_table,id_column,size,num_distincts,num_nan,%nan')

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
        
    



@print_info(msg_before='Extracting intitial sets...', msg_after='Completed.')
def extract_starting_sets_from_tables(input_tables_csv_dir, 
                                      final_set_file,
                                      id_table_file,
                                      tables_stat_file,
                                      columns_stat_file,
                                      ntables_to_load_as_set=10, 
                                      with_:Literal['mytok', 'infer', 'set', 'bag']='set',
                                      **kwargs):
    if with_ == 'infer':
        nlp = spacy.load('en_core_web_sm')
    
    # initialise the metadata and statistics dataframes
    id_tables = pd.DataFrame(columns=['josieID', 'slothID'])
    tables_stat = pd.DataFrame(columns=['id_table', 'size', 'set_size', 'num_nan', '%nan'])
    columns_stat = pd.DataFrame(columns=['id_table', 'id_column', 'size', 'num_distincts', 'num_nan', '%nan'])
        
    # read each CSV, that's actually a table, and create its token set
    # meanwhile, keep statistics about number of distincts, NaNs, timing...
    with open(final_set_file, 'w') as set_writer:
        for i, id_table in tqdm(enumerate(os.listdir(input_tables_csv_dir)), total=ntables_to_load_as_set):
            if i >= ntables_to_load_as_set:
                break
            
            # read the csv
            table = pd.read_csv(input_tables_csv_dir + f'/{id_table}').convert_dtypes()
            
            # extract the token set
            if with_ == 'mytok':
                table_set = _create_set_with_my_tokenizer(table)
            elif with_ == 'infer':
                table_set = _create_set_with_inferring_column_dtype(table, nlp=nlp, **kwargs)
            elif with_ == 'set':
                table_set = _create_set_with_all_tokens(table, bag=False)
            elif with_ == 'bag':
                table_set = _create_set_with_all_tokens(table, bag=True)
            else:
                raise AttributeError("Parameter with_ must be a value in {'mytok', 'infer', 'set', 'bag'}")
            
            # store the token set
            set_writer.write(
                str(i) + ',' + ','.join(table_set) + '\n'
            )
            
            # correspondences integer/original table id
            id_tables.loc[len(id_tables)] = [i, id_table]
            
            # table statistics
            table_size = table.shape[0] * table.shape[1]
            num_na = table.isna().sum().sum()
            tables_stat.loc[len(tables_stat)] = [
                id_table,
                table_size,
                len(table_set),
                num_na, # NA values in the whole table
                round(num_na / table_size, 5)
            ]
            
            # columns statistics
            for i in range(len(table.columns)):
                num_distinct = table.iloc[:, i].unique().shape[0]
                num_na = table.iloc[:, i].isna().sum()
                columns_stat.loc[len(columns_stat)] = [
                    id_table,
                    i,
                    table.shape[1],
                    num_distinct,
                    num_na,
                    round(num_na / table.shape[1], 5)                        
                ]
                
    # store metadatas and statistics
    id_tables.to_csv(id_table_file, index=False)
    tables_stat.to_csv(tables_stat_file, index=False)
    columns_stat.to_csv(columns_stat_file, index=False)



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
                lambda line: line.split(',')
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
def create_index(input_set_file, output_integer_set_file, output_inverted_list_file, spark_context=None):
    if not spark_context:
        conf = pyspark.SparkConf() \
            .setAppName('CreateIndex')
                # .set('spark.executor.memory', '100g') \
                # .set('spark.driver.memory', '5g')

        spark_context = pyspark.SparkContext(conf=conf)

    skip_tokens = set()

    # STAGE 1: BUILD TOKEN TABLE
    # print('STAGE 1: BUILD TOKEN TABLE')
    # Load sets and filter out removed token
    sets = spark_context \
        .textFile(input_set_file) \
            .map(
                lambda line: line.split(',')
            ) \
                .map(
                    lambda line: (
                        int(line[0]),
                        [token for token in line[1:] if token not in skip_tokens]
                    )
                )

    token_sets = sets.flatMap(
            lambda sid_tokens: \
                [
                    (token, sid_tokens[0]) 
                    for token in sid_tokens[1]
                ]
        )

    def compare(a:tuple[int,list[int]], b:tuple[int,list[int]]):
        la, lb = len(a[1]), len(b[1])

        if la != lb: 
            return -1 if la < lb else 0 if la == lb else 1
        elif not a or not b:
            return 0
        elif a[0] != b[0]:
            return -1 if a[0] < b[0] else 0 if a[0] == b[0] else 1 
        else:
            for (x, y) in zip(a[1], b[1]):
                if x != y:
                    return -1 if x < y else 0 if x == y else 1
            return 0

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
                        lambda t: (len(t[1]), t[1] == [], t[2], t[1])
                    ) \
                        .zipWithIndex() \
                            .map(
                                # t: (rawToken, sids, hash), tokenIndex
                                lambda t: (t[1], (t[0][0], t[0][1], t[0][2]))
                            ) \
                                .persist(pyspark.StorageLevel.MEMORY_ONLY)

    def equal_arrays(a1, a2):
        return len(a1) == len(a2) and all(x1 == x2 for (x1, x2) in zip(a1, a2))

    duplicate_group_ids = posting_lists_sorted \
        .map(
            # tokenIndex, (rawToken, sids, hash)
            lambda t: (t[0] + 1, (t[1][1], t[1][2]))
        ) \
            .join(posting_lists_sorted) \
                .map(
                    # (tokenIndexUpper, ((sidsLower, hashLower), (_, sidsUpper, hashUpper)))
                    lambda t: 
                        -1 if equal_arrays(t[1][0][0], t[1][1][1]) and t[1][0][1] == t[1][1][2] else t[0]
                ) \
                    .filter(
                        lambda i: i > 0
                    ) \
                        .union(
                            spark_context.parallelize([0])
                        ) \
                            .sortBy(
                                lambda i: i
                            ) \
                                .zipWithIndex() \
                                    .map(
                                        # (startingIndex, GroupID)
                                        lambda t: (t[1], t[0])
                                    )

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
                            lambda token_index: (token_index, t[0]),
                            range(t[1][0], t[1][1])
                        )
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)

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
    # print('STAGE 2: CREATE INTEGER SETS')

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

    # STAGE 3: CREATE THE FINAL POSTING LISTS
    # Create new posting lists and join the previous inverted
    # lists to obtain the final posting lists with all the information
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

    # print('STAGE 3: CREATE THE FINAL POSTING LISTS')

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
                            # (token, (sets, (gid, rawToken, _)))
                            lambda t: (t[0], t[1][1][1], t[1][1][0], t[1][0])
                        ) \
                            .map(
                                lambda t: postlist_format_string(t)
                            )
    
    # STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS
    # print('STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS')

    integer_sets.map(
        lambda t: sets_format_string(t)
    ).saveAsTextFile(output_integer_set_file)

    posting_lists.saveAsTextFile(output_inverted_list_file)

    """
    integer_sets = integer_sets.map(
        # (sid, indices)
        lambda t: sets_format_string(t) #f"{t[0]}, {','.join(map(str, t[1]))}\n"
    ).collect() #.saveAsTextFile(output_integer_set_file)

            
    posting_lists = posting_lists.map(
        # token, rawToken, gid, sets
        # lambda t: f"{t[0]} {t[1]} {t[2]} {' '.join(map(str, t[3]))}\n"
        lambda t: postlist_format_string(t)
    ).collect() # .saveAsTextFile(output_inverted_list_file)


    # since the data now is really small, a single file is ok (no Spark partitioning)

    with open(output_integer_set_file, 'w') as f:
        f.writelines(integer_sets) 

    with open(output_inverted_list_file, 'w') as f:
        f.writelines(posting_lists) 

    """

if __name__ == '__main__':
    ROOT_TEST_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/josie-test/'
    input_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set'
    output_integer_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set-2'
    output_inverted_list_file = ROOT_TEST_DIR + 'sloth_tables_n10.inverted-list'
