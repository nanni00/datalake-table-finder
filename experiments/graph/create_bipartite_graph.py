from collections import Counter
import os
from pprint import pprint
from time import time
import pandas as pd


import mmh3
from neo4j import GraphDatabase
from pyspark.sql import SparkSession


def get_table_tokens_counter(table, numeric_columns):
    def prepare_token(token):
        return str(token).replace('|', ' ').replace('\n', ' ')

    tokens = [mmh3.hash(prepare_token(token), seed=29122000) for row in table for icol, token in enumerate(row) 
                    if not pd.isna(token) and token and numeric_columns[icol] == 0]

    return Counter(tokens)



small = False

USER = "neo4j"
PASSWD = "12345678"

AUTH = (USER, PASSWD)
DATABASE = "neo4j"
URI = f"bolt://localhost:7687"


MIN_ROW = 5
MAX_ROW = 999999
MIN_COLUMN = 2
MAX_COLUMN = 999999
MIN_AREA = 50
MAX_AREA = 999999


spark_jars_packages = [
    'org.neo4j:neo4j-connector-apache-spark_2.12:4.1.5_for_spark_3',
    'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
]

num_cores = min(os.cpu_count(), 64)

with GraphDatabase.driver(uri=URI, auth=AUTH) as driver:
    with driver.session(database=DATABASE) as session:
        session.run("CREATE INDEX table_id_range_index IF NOT EXISTS FOR (tab:Table) ON (tab.table_id)")
        session.run("CREATE INDEX token_id_range_index IF NOT EXISTS FOR (tok:Token) ON (tok.token_id)")
        session.run("CREATE LOOKUP INDEX node_label_lookup IF NOT EXISTS FOR (n) ON EACH labels(n)")
        # session.run("CREATE CONSTRAINT table_id_constraint IF NOT EXISTS FOR (tab:Table) REQUIRE tab.table_id IS KEY")
        # session.run("CREATE CONSTRAINT token_id_constraint IF NOT EXISTS FOR (tok:Token) REQUIRE tok.token_id IS KEY")


# fine tune of executor/driver.memory?
builder = SparkSession.Builder()
spark = (
    builder
    .appName("Final Test with Neo4j")
    .master(f"local[{num_cores}]")
    .config('spark.jars.packages', ','.join(spark_jars_packages))
    .config('spark.executor.memory', '100g')
    .config('spark.driver.memory', '10g')
    .config('neo4j.url', URI)
    .config('neo4j.authentication.basic.username', USER)
    .config('neo4j.authentication.basic.password', PASSWD)
    # .config("neo4j.database", DATABASE)
    .getOrCreate()
)

# adjusting logging level to error, avoiding warnings
spark.sparkContext.setLogLevel("ERROR")


    
optitab__turl_training_set_df = (
    spark
    .read 
    .format("mongodb")
    .option ("uri", "mongodb://127.0.0.1:27017/")
    .option("database", "optitab")
    .option("collection", "turl_training_set" if not small else "turl_training_set_small")
    .load()
    .select('_id_numeric', 'content', 'numeric_columns')
    # .limit(5000)
    .filter(f"""
            size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
            AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
            AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""")
)

sloth__latest_snapshot_tables_df = (
    spark
    .read
    .format('mongodb')
    .option("uri", "mongodb://127.0.0.1:27017/")
    .option("database", "sloth")
    .option("collection", "latest_snapshot_tables" if not small else "latest_snapshot_tables_small")
    .load()
    .select('_id_numeric', 'content', 'numeric_columns')
    # .limit(5000)
    .filter(f"""
            size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
            AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
            AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""")
)


df = sloth__latest_snapshot_tables_df.union(optitab__turl_training_set_df).rdd.map(list)

# free memory used by the dataframe (is this really useful?)
# optitab__turl_training_set_df.unpersist()   
# sloth__latest_snapshot_tables_df.unpersist()    



def prepare_tuple(t):
    # t = (_id_numeric, content, numeric_columns)
    _id_numeric, content, numeric_columns = t
    token_cnt = get_table_tokens_counter(content, numeric_columns)
    return _id_numeric, [(token, cnt) for token, cnt in token_cnt.items()]


table_token_cnt = (
    df
    .map(
        # (_id, _id_numeric, content, numeric_columns) -> (_id_numeric, [token1, token2, token3, ...])
        lambda t: prepare_tuple(t)
        ) 
        .flatMap(
            # (set_id, [(tok1, cnt_tok1), ...]) -> [(tok1, set_id), (tok2, set_id), ...]
            lambda t:
                [
                    (t[0], *token_cnt) 
                    for token_cnt in t[1]
                ]
        )
)

# drop those tokens that have just one link to a table, since they won't give any
# information about possible overlap (obv in future updates of the graph this
# may be a relevant information loss)
table_token_cnt = (
    table_token_cnt
        .groupBy(lambda x: x[1])
        .filter(lambda x: len(x[1]) > 1)
        .flatMap(
            lambda x: [t for t in x[1]]
        )
        .sortBy(
            lambda tabid_tokid_tokcnt: (tabid_tokid_tokcnt[0], tabid_tokid_tokcnt[1])
        )
)



print(table_token_cnt.count())
table_token_cnt = table_token_cnt.toDF(schema=['table_id', 'token_id', 'token_count'])
pprint(table_token_cnt.head(n=5))

if 0:
    print('saving tables...')
    start = time()
    (
        table_token_cnt 
        .select('table_id')
        .distinct()
        .write
        .format("org.neo4j.spark.DataSource")
        .mode("Append")
        .option("labels", "Table")
        .save()
    )
    print(time() - start)

    print('saving tokens...')
    start = time()
    (
        table_token_cnt 
        .select('token_id')
        .distinct()
        .write
        .format("org.neo4j.spark.DataSource")
        .mode("Append")
        .option("labels", "Token")
        .save()
    )
    print(time() - start)

table_token_cnt = table_token_cnt.coalesce(1)

print('saving relationships...')
start = time()
(
    table_token_cnt
        .write
        .mode("Overwrite")
        .format("org.neo4j.spark.DataSource")
        .option("transaction.retry.timeout", 100)
        .option("relationship", "HAS")
        .option("relationship.save.strategy", "keys")

        .option("relationship.source.save.mode", "Match")
        .option("relationship.source.labels", ":Table")
        .option("relationship.source.node.keys", "table_id")

        .option("relationship.target.save.mode", "Match")
        .option("relationship.target.labels", ":Token")
        .option("relationship.target.node.keys", "token_id")

        .option("schema.optimization.node.keys", "UNIQUE")

        .option("relationship.properties", "token_count")
        .save()
)


# query = """
# MERGE (tab:Table {table_id: event.table_id})
# MERGE (tok:Token {token_id: event.token_id})
# CREATE (tab)-[:HAS {token_count:event.token_count}]->(tok)
# """
# 
# (
#     table_token_cnt
#         .write
#         .format("org.neo4j.spark.DataSource")
#         .mode("Overwrite")
#         .option("query", query)
#         .option("database", DATABASE)
#         .save()
# )
# 
print(f"Completed in {round(time() - start, 3)}s")

