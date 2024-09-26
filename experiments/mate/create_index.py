"""
Table schema for main index:
    TABLE_ID, COLUMN_ID, ROW_ID, token, tokenized

where the tuple (TABLE_ID, COLUMN_ID, ROW_ID) is PK

"tokeinzed" should be the tokenized+stemmed version of "token", but in JOSIE
we didn't do any of these passages (except a basic replacement for characters '\n' and '|')
so here we should keep the same
"""

import os
import sqlalchemy
from sqlalchemy.engine import create_engine
from sqlalchemy import Table, Column, Integer, VARCHAR

from pyspark.sql import SparkSession

from tools.utils.misc import clean_string, is_valid_table


def mate__table_to_posting_lists(table_id:int, table:list, bad_columns:list[int], blacklist:set):
    return sorted([
        # dict(zip(['table_id', 'column_id', 'row_id', 'tokenized'], [table_id, column_id, row_id, cell, clean_string(cell)]))
        [table_id, column_id, row_id, clean_string(cell), 0]
        for row_id, row in enumerate(table)
            for column_id, cell in enumerate(row)
                if not bad_columns[column_id] and clean_string(cell) not in blacklist and cell not in blacklist
    # ], key=lambda x: (x['table_id'], x['column_id'], x['row_id']))
    ], key=lambda x: (x[0], x[1], x[2]))






def get_spark_session(num_cpu, spark_local_dir:str, spark_jars_packages=['org.mongodb.spark:mongo-spark-connector_2.12:10.3.0']) -> SparkSession:
    builder = SparkSession.Builder()
    spark = (
        builder
        .appName("MATE inverted index generation")
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




def mate__create_inverted_index_table(posting_lists_tablename, path_duckdb, **connection_parameters):
    if not os.path.exists(os.path.dirname(path_duckdb)):
        raise FileNotFoundError(f"Directory doesn't exist: {os.path.dirname(path_duckdb)}")
    if os.path.exists(path_duckdb):
        # raise FileExistsError(f"DuckDB already exists: {path_duckdb}")
        os.remove(path_duckdb)
    
    engine = create_engine(f'duckdb:///{path_duckdb}', **connection_parameters)
    metadata = sqlalchemy.MetaData(engine)

    # with multiple "primary_key=True" columns, we define a composite key
    # (see https://docs.sqlalchemy.org/en/20/core/metadata.html#module-sqlalchemy.schema)
    posting_lists_table = Table(
        posting_lists_tablename,
        metadata,
        Column('tableid', Integer, primary_key=True),
        Column('colid', Integer, primary_key=True),
        Column('rowid', Integer, primary_key=True),
        Column('tokenized', VARCHAR(255)),
        Column('superkey', Integer)
    )

    metadata.create_all(engine)
    engine.dispose()



def mate__create_inverted_index(blacklist:set, posting_lists_tablename, path_duckdb, **connection_parameters):
    mate__create_inverted_index_table(posting_lists_tablename, path_duckdb, **connection_parameters)

    jars = [
        'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0',
        'org.duckdb:duckdb_jdbc:1.0.0'
    ]

    spark = get_spark_session(10, spark_local_dir='/data4/nanni/spark', spark_jars_packages=jars)
    databases, collections = ['optitab', 'sloth'], ['turl_training_set', 'latest_snapshot_tables']            
    db_collections = zip(databases, collections)
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
            .rdd
            .map(list)
        )



    def prepare_tuple(t):
        nonlocal blacklist
        # t = (_id_numeric, content, numeric_columns) -> posting_list
        return mate__table_to_posting_lists(*t, blacklist=blacklist)
    
    posting_lists = (
        initial_rdd
            .sample(False, 0.001)
            .filter(
                # (_id_numeric, content, numeric_columns)
                # lambda t: is_valid_table(t[1], t[2], tables_thresholds)
                lambda t: is_valid_table(t[1], t[2])
            )
            .flatMap(
                # from MongoDB directly
                # (_id_numeric, content, numeric_columns)
                lambda t: prepare_tuple(t)
            )
    )

    url = f"jdbc:duckdb:///{path_duckdb}"

    properties = {
        "driver": "org.duckdb.DuckDBDriver"
    }
    
    df = (
        posting_lists
        .toDF(schema=['tableid', 'rowid', 'colid', 'tokenized', 'superkey'])
    )

    print(df.head(5))

    (
        df
        .write
        .jdbc(url, posting_lists_tablename, 'overwrite', properties)
    )


if __name__ == '__main__':
    posting_lists_tablename = 'firstTest'
    path_duckdb = f"{os.path.dirname(__file__)}/myduck.db"
    mate__create_inverted_index(set(), posting_lists_tablename, path_duckdb)



