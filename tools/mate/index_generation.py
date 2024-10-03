"""
Table schema for main index:
    TABLE_ID, COLUMN_ID, ROW_ID, token, tokenized

where the tuple (TABLE_ID, COLUMN_ID, ROW_ID) is PK

"tokeinzed" should be the tokenized+stemmed version of "token", but in JOSIE
we didn't do any of these passages (except a basic replacement for characters '\n' and '|')
so here we keep the same
"""
import os
import math
from time import time
from functools import reduce
from collections import Counter

import numpy as np

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
from sqlalchemy import (
    Table, Column, 
    create_engine,
    Integer, VARCHAR, BigInteger, 
    MetaData
)

from tools.josie import get_spark_session
from tools.utils.settings import DefaultPath as dp
from tools.utils.misc import (
    clean_string, is_valid_table, 
    lowercase_translator, whitespace_translator, punctuation_translator
)


def XASH(token: str, hash_size: int = 128) -> int:
    """Computes XASH for given token.

    Parameters
    ----------
    token : str
        Token.

    hash_size : int
        Number of bits.

    Returns
    -------
    int
        XASH value.
    """
    number_of_ones = 5
    char = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
            'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]
    length_bit_start = 37 * segment_size
    result = 0
    cnt_dict = Counter(token)
    selected_chars = [y[0] for y in sorted(cnt_dict.items(), key=lambda x: (x[1], x[0]), reverse=False)[:number_of_ones]]
    for c in selected_chars:
        if c not in char:
            continue
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = np.mean(indices)
        token_size = len(token)
        for i in np.arange(segment_size):
            if mean_index <= ((i + 1) * token_size / segment_size):
                location = char.index(c) * segment_size + i
                break
        result = result | int(math.pow(2, location))

    # rotation
    n = int(result)
    d = int((length_bit_start * (len(token) % (hash_size - length_bit_start))) / (
                hash_size - length_bit_start))
    int_bits = int(length_bit_start)
    x = n << d
    y = n >> (int_bits - d)
    r = int(math.pow(2, int_bits))
    result = int((x | y) % r)

    result = int(result) | int(math.pow(2, len(token) % (hash_size - length_bit_start)) * math.pow(2, length_bit_start))

    return int(result)


def mate__table_to_posting_lists(table_id:int, table:list, bad_columns:list[int], blacklist:set, hash_size:int, string_translators=[]):
    def row_xash(row):
        return reduce(lambda a, b: a | b, map(lambda t: XASH(str(t)[:255], hash_size), row), 0)

    return sorted([
        [table_id, column_id, row_id, clean_string(cell, *string_translators)[:255], str(row_xash(row))]
        for row_id, row in enumerate(table)
            for column_id, cell in enumerate(row)
                if not bad_columns[column_id] and clean_string(cell, *string_translators) not in blacklist and cell not in blacklist
        ], key=lambda x: (x[0], x[1], x[2]))


def mate__create_inverted_index_table(posting_lists_tablename, **connection_info):
    engine = create_engine(**connection_info)
    metadata = MetaData(engine)

    # with multiple "primary_key=True" columns, we define a composite key
    # (see https://docs.sqlalchemy.org/en/20/core/metadata.html#module-sqlalchemy.schema)
    # no need to create an index on tableid/rowid/colid or whatever (right ye?)
    _ = Table(
        posting_lists_tablename,
        metadata,
        Column('tableid',   Integer,        primary_key=True),
        Column('colid',     Integer,        primary_key=True),
        Column('rowid',     Integer,        primary_key=True),
        Column('tokenized', VARCHAR(255)),
        Column('superkey',  VARCHAR(255))
    )

    metadata.create_all(engine)
    engine.dispose()


def mate__create_inverted_index(hash_size:int, processes:int, blacklist:set, posting_lists_tablename:str, string_translators:dict|None=None, **connection_info):
    mate__create_inverted_index_table(posting_lists_tablename, **connection_info)
    
    jars = [
        'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0',
        'org.duckdb:duckdb_jdbc:1.0.0',
        'org.postgresql:postgresql:42.7.3'
    ]
    
    url = f"jdbc:{connection_info['url']}"
    properties = {
        # driver": "org.duckdb.DuckDBDriver"
        "user": "nanni",
        "password": "",
        "driver": "org.postgresql.Driver",
        "batchsize": 5000
    }


    _, initial_rdd = get_spark_session(processes, 'mongodb', 'wikiturlsnap', spark_local_dir='/data4/nanni/spark', spark_jars_packages=jars)
    
    schema = StructType(
        [
            StructField('tableid',      IntegerType(),  False), 
            StructField('colid',        IntegerType(),  False),
            StructField('rowid',        IntegerType(),  False),
            StructField('tokenized',    StringType(),   False),
            StructField('superkey',     StringType(),   False)
        ]
    )

    def prepare_tuple(table_id:int, table_rows:list, bad_columns:list):
        nonlocal hash_size, blacklist, string_translators
        return mate__table_to_posting_lists(table_id, table_rows, bad_columns, hash_size=hash_size, blacklist=blacklist, string_translators=string_translators)

    # posting lists creation and storing on DB
    (
        initial_rdd
        .filter(lambda t: is_valid_table(t[1], t[2]))
        .flatMap(lambda t: prepare_tuple(*t))
        .toDF(schema=schema)
        .write
        .jdbc(url, posting_lists_tablename, 'append', properties) # the overwrite mode drops also the index...
    )


if __name__ == '__main__':
    # db_path = f"{dp.data_path.base}/mate_index.db"
    connection_info = {
        # 'url': f'duckdb:///{db_path}'
        'url': f'postgresql://localhost:5442/nanni'
    }

    hash_size = 128
    posting_lists_tablename = f'mate__wikiturlsnap_table_{hash_size}'
    custom_translator = str.maketrans(';$_\n|', '     ')
    start = time()
    mate__create_inverted_index(hash_size, 72, set(), posting_lists_tablename, [custom_translator], **connection_info)
    step1 = time()
    print(f'create init index time: {round(step1 - start, 3)}s')
