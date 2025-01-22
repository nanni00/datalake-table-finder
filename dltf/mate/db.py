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
from typing import List
from functools import reduce
from collections import Counter

import numpy as np
import pandas as pd

from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, declarative_base
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from sqlalchemy import Column, create_engine, Integer, VARCHAR, MetaData, func, select

from dltf.utils.misc import clean_string
from dltf.utils.tables import is_valid_table
from dltf.utils.spark import get_spark_session
from dltf.utils.datalake import DataLakeHandlerFactory



Base = declarative_base()


class MATEInvertedLists(Base):
    __tablename__   = 'mate_inverted_lists'
    tableid         = Column(Integer, primary_key=True, index=True)
    colid           = Column(Integer, primary_key=True, index=True)
    rowid           = Column(Integer, primary_key=True, index=True)
    tokenized       = Column(VARCHAR(255))
    superkey        = Column(VARCHAR(255))
    

class MATEDBHandler:
    def __init__(self, mate_cache_path:str, read_only:bool=True, connection_info:dict|None=None):
        self.mate_cache_path = mate_cache_path
        
        self.url = URL.create(**connection_info)
        self.engine = create_engine(self.url, connect_args={'read_only': read_only})

        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)

    def create_index_table(self):
        Base.metadata.create_all(self.engine)

    def get_concatinated_posting_list(self,
                                      dataset_name: str,
                                      query_column_name: str,
                                      value_list: pd.Series,
                                      top_k: int = -1,
                                      database_request: bool = True):
        """
        Fetches posting lists for top-k values.

        Args:
            dataset_name (str): Name of the query dataset.
            query_column_name (str): Name of the query column.
            value_list (list): Values to fetch posting lists for.
            top_k (int): Number of posting lists to fetch. -1 to fetch all.
            database_request (bool): If true, posting lists are fetched from the database. Otherwise, cached files are used (if existing).
        """
        if os.path.isfile(f"{self.mate_cache_path}/{dataset_name}_{query_column_name}_concatenated_posting_list.txt")\
                and top_k == -1 and not database_request:
            pl = []
            with open(f"{self.mate_cache_path}/{dataset_name}_{query_column_name}_concatenated_posting_list.txt", "r") as f:
                for line in f:
                    pl += [line.strip()]
            return pl
        else:
            distinct_clean_values = value_list.unique().tolist()
            with Session(self.engine) as session:
                if top_k != -1:
                    stmt = (
                        select(MATEInvertedLists.tableid, MATEInvertedLists.rowid, MATEInvertedLists.colid, MATEInvertedLists.tokenized, MATEInvertedLists.superkey)
                        .where(
                            func.regexp_replace(
                                func.regexp_replace(MATEInvertedLists.tokenized, '\W+', ' '), ' +', ' '
                            ).in_(distinct_clean_values))
                        .limit(top_k)
                    )                    
                else:
                    stmt = (
                        select(MATEInvertedLists.tableid, MATEInvertedLists.rowid, MATEInvertedLists.colid, MATEInvertedLists.tokenized, MATEInvertedLists.superkey)
                        .distinct()
                        .where(MATEInvertedLists.tokenized.in_(distinct_clean_values))
                    )
                
            pl = session.execute(stmt)

            if top_k == -1 and not database_request:
                with open(f"{self.mate_cache_path}/{dataset_name}_{query_column_name}_concatenated_posting_list.txt", "w") as f:
                    for s in pl:
                        f.write(str(s) + "\n")
            return pl

    def get_pl_by_table_and_rows(self, joint_list: List[str]) -> List[List[str]]:
        """
        Fetches posting lists from a set of table_row_ids.

        Args:
            joint_list (list): List of table_row_ids.

        Returns:
            list: Posting lists corresponding to the table_row_ids.
        """
        distinct_clean_values = set(joint_list)
        tables, rows = zip(*map(lambda s: s.split('_'), joint_list))

        query = (
            select(func.concat(MATEInvertedLists.tableid, '_', MATEInvertedLists.rowid), MATEInvertedLists.colid, MATEInvertedLists.tokenized)
            .where(MATEInvertedLists.tableid.in_(tables))
            .where(MATEInvertedLists.rowid.in_(rows))
            .where(func.concat(MATEInvertedLists.tableid, '_', MATEInvertedLists.rowid).in_(distinct_clean_values))
        )

        with Session(self.engine) as session:
            return list(session.execute(query))

    def create_inverted_index(self, hash_size:int, string_blacklist:set, string_translators, string_patterns, dlhconfig, spark_config:dict|None):
        def prepare_tuple(table_id:int, table_rows:list, valid_columns:list):
            nonlocal hash_size, string_blacklist, string_translators
            return table_to_posting_lists(table_id, table_rows, valid_columns, hash_size=hash_size, str_blacklist=string_blacklist, str_translators=string_translators, str_patterns=string_patterns)
        
        # DuckDB concurrency error if the DBHandler keeps the lock 
        # when pyspark attempts to write the database
        self.engine.dispose()

        jdbc_url = f'jdbc:{URL.create(drivername=self.url.drivername, database=self.url.database, host=self.url.host, port=self.url.port)}'
        jdbc_properties = {
            'user'      : self.url.username if self.url.username else '',
            'password'  : self.url.password if self.url.password else ''
        }

        print(f'{jdbc_url=}')
        print(f'{jdbc_properties=}')


        schema = StructType(
            [
                StructField('tableid'   , IntegerType() , False), 
                StructField('colid'     , IntegerType() , False),
                StructField('rowid'     , IntegerType() , False),
                StructField('tokenized' , StringType()  , False),
                StructField('superkey'  , StringType()  , False)
            ]
        )

        dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
        spark, rdd = get_spark_session(dlh, **spark_config)
    
        (
            rdd
            .map(lambda t: [t['_id_numeric'], 
                            t['content'] if 'num_header_rows' not in t else t['content'][t['num_header_rows']:],
                            t['valid_columns']])
            .filter(lambda t: is_valid_table(t[1], t[2]))
            .flatMap(lambda t: prepare_tuple(*t))
            .toDF(schema=schema)
            .write
            .jdbc(
                url=jdbc_url, 
                table=MATEInvertedLists.__tablename__, 
                mode='append', # the overwrite mode drops also the index...
                properties=jdbc_properties
            )
        )

        spark.stop()


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


def table_to_posting_lists(table_id:int, table:list, valid_columns:list[int], hash_size:int, str_blacklist:set, str_translators=[], str_patterns=[]):
    def row_xash(row):
        return reduce(lambda a, b: a | b, map(lambda t: XASH(str(t)[:255], hash_size), row), 0)
    table = [[clean_string(cell, str_translators, str_patterns) for cell in row] for row in table]
    return sorted([
        [table_id, column_id, row_id, cell[:255], str(row_xash(row))]
        for row_id, row in enumerate(table)
            for column_id, cell in enumerate(row)
                if valid_columns[column_id] and cell not in str_blacklist
        ], key=lambda x: (x[0], x[1], x[2]))




def main_demo_postgres():
    connection_info = {
        'drivername':   'postgresql',
        'database':     'DEMODB',
        'username':     'demo',
        'password':     'demo',
        'host':         'localhost',
        'port':         5442
    }

    num_cpu = 10
    dlhconfig = ['mongodb', 'demo', ['sloth.demo']]

    spark_config = {
        'spark.app.name':               'MATE Index Preparation',
        'spark.master':                 f"local[{num_cpu}]",
        'spark.executor.memory':        '100g',
        'spark.driver.memory':          '20g',
        'spark.local.dir':              f'{os.path.dirname(__file__)}/tmp',
        'spark.driver.maxResultSize':   '12g',
        'spark.jars.packages':          'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0',
        'spark.driver.extraClassPath':  f'{os.environ["HOME"]}/.ivy2/jars/org.postgresql_postgresql-42.7.3.jar'
    }

    hash_size = 128
    custom_translator = str.maketrans('"', ' ')
    
    tmp_dir = f'{os.path.dirname(__file__)}/tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    matedb = MATEDBHandler('', connection_info)
    start = time()
    matedb.create_index_table()
    matedb.create_inverted_index(
        hash_size, 
        string_blacklist=set(), 
        string_translators=['lowercase', 'whitespace', custom_translator], 
        string_patterns=[],
        dlhconfig=dlhconfig,
        spark_config=spark_config
    )
    end = time()
    print(f'create init index time: {round(end - start, 3)}s')



def main_demo_duckdb():
    dbpath = f'{os.path.dirname(__file__)}/mate_duck.db'
    
    connection_info = {
        'drivername':   'duckdb',
        'database':     dbpath,
    }

    num_cpu = 10
    dlhconfig = ['mongodb', 'demo', ['sloth.demo']]

    spark_config = {
        'spark.app.name':               'MATE Index Preparation',
        'spark.master':                 f"local[{num_cpu}]",
        'spark.executor.memory':        '100g',
        'spark.driver.memory':          '20g',
        'spark.local.dir':              f'{os.path.dirname(__file__)}/tmp',
        'spark.driver.maxResultSize':   '12g',
        'spark.jars.packages':          'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0',
        'spark.driver.extraClassPath':  f'{os.environ["HOME"]}/.ivy2/jars/org.duckdb-duckdb_jdbc-1.1.3.jar'
    }

    hash_size = 128
    
    tmp_dir = f'{os.path.dirname(__file__)}/tmp'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    matedb = MATEDBHandler('', connection_info)
    start = time()
    matedb.create_index_table()
    matedb.create_inverted_index(
        hash_size, 
        string_blacklist=set(), 
        string_translators=['lowercase', 'whitespace', ['"', ' ']], 
        string_patterns=[],
        dlhconfig=dlhconfig,
        spark_config=spark_config
    )
    end = time()
    print(f'create init index time: {round(end - start, 3)}s')


if __name__ == '__main__':
    main_demo_duckdb()
