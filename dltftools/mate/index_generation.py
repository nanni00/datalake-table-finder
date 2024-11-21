"""
Table schema for main index:
    TABLE_ID, COLUMN_ID, ROW_ID, token, tokenized

where the tuple (TABLE_ID, COLUMN_ID, ROW_ID) is PK

"tokeinzed" should be the tokenized+stemmed version of "token", but in JOSIE
we didn't do any of these passages (except a basic replacement for characters '\n' and '|')
so here we keep the same
"""
import math
from time import time
from functools import reduce
from collections import Counter

import numpy as np

from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sqlalchemy import (
    Table, Column, 
    create_engine,
    Integer, VARCHAR, 
    MetaData
)
from sqlalchemy.engine import URL

from dltftools.utils.tables import is_valid_table
from dltftools.utils.spark import get_spark_session
from dltftools.utils.misc import (
    clean_string, 
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


def table_to_posting_lists(table_id:int, table:list, bad_columns:list[int], blacklist:set, hash_size:int, string_translators=[]):
    def row_xash(row):
        return reduce(lambda a, b: a | b, map(lambda t: XASH(str(t)[:255], hash_size), row), 0)

    return sorted([
        [table_id, column_id, row_id, clean_string(cell, *string_translators)[:255], str(row_xash(row))]
        for row_id, row in enumerate(table)
            for column_id, cell in enumerate(row)
                if not bad_columns[column_id] and clean_string(cell, *string_translators) not in blacklist and cell not in blacklist
        ], key=lambda x: (x[0], x[1], x[2]))


def create_inverted_index_table(posting_lists_tablename, **connection_info):
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


def create_inverted_index(hash_size:int, blacklist:set, posting_lists_tablename:str, string_translators:dict|None, connection_info:dict|None, spark_config:dict|None):
    url = URL.create(**connection_info)
    jdbc_url = f'jdbc:{URL.create(drivername=url.drivername, database=url.database, host=url.host, port=url.port)}'
    jdbc_properties = {
        'user': url.username,
        'password': url.password
    }
    
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
        return table_to_posting_lists(table_id, table_rows, bad_columns, hash_size=hash_size, blacklist=blacklist, string_translators=string_translators)

    create_inverted_index_table(posting_lists_tablename, **connection_info)
    spark, data_rdd = get_spark_session('mongodb', 'wikiturlsnap', **spark_config)

    # posting lists creation and storing on DB
    (
        data_rdd
        .filter(lambda t: is_valid_table(t[1], t[2]))
        .flatMap(lambda t: prepare_tuple(*t))
        .toDF(schema=schema)
        .write
        .jdbc(jdbc_url, posting_lists_tablename, 'append', jdbc_properties) # the overwrite mode drops also the index...
    )

    spark.sparkContext.stop()


if __name__ == '__main__':
    connection_info = {
        'drivername': 'postgresql',
        'database': 'JOSIEDB',
        'username': 'nanni',
        'password': '',
        'port': 5442,
        'host': 'localhost'
    }

    num_cpu = 64
    spark_local_dir = '/data4/nanni/spark'

    # the Spark JAR for JDBC should not be inserted there, since it's a known issue
    # that a JAR passed as package here won't be retrieved as driver class
    spark_jars_packages = [
        'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0',
        'org.duckdb:duckdb_jdbc:1.0.0'
    ]

    # driver_path = '$HOME/.ivy2/jars/org.postgresql:postgresql:42.7.3'
    spark_config = {
        'spark.app.name':               'JOSIE Data Preparation',
        'spark.master':                 f"local[{num_cpu}]",
        'spark.executor.memory':        '100g',
        'spark.driver.memory':          '20g',
        'spark.local.dir':              spark_local_dir,
        'spark.driver.maxResultSize':   '12g',
        'spark.jars.packages':          ','.join(spark_jars_packages),
        'spark.driver.extraClassPath':  '/path/to/driver/jar'
    }

    hash_size = 128
    posting_lists_tablename = f'mate__wikiturlsnap_table_{hash_size}'
    custom_translator = str.maketrans(';$_\n|', '     ')
    
    start = time()
    create_inverted_index(hash_size, set(), posting_lists_tablename, [custom_translator], connection_info, spark_config)
    step1 = time()
    
    print(f'create init index time: {round(step1 - start, 3)}s')
