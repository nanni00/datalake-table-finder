import os
from time import time
from collections import Counter

import mmh3
import pandas as pd
from neo4j import GraphDatabase

from tools.utils.utils import AlgorithmTester, get_initial_spark_rdd, get_local_time


def get_table_tokens_counter(table, numeric_columns):
    def prepare_token(token):
        return str(token).replace('|', ' ').replace('\n', ' ')

    tokens = [mmh3.hash(prepare_token(token), seed=29122000) for row in table for icol, token in enumerate(row) 
                    if not pd.isna(token) and token and numeric_columns[icol] == 0]

    return Counter(tokens)


class Neo4jTester(AlgorithmTester):
    def __init__(self, mode, small, tables_thresholds, num_cpu, *args) -> None:
        super().__init__(mode, small, tables_thresholds, num_cpu)
        
        self.user, self.password, self.neo4j_db_dir, self.collections = args

    def data_preparation(self):
        start_data_prep = time()
        AUTH = (self.user, self.password)
        DATABASE = "neo4j"
        URI = f"bolt://localhost:7687"

        spark_jars_packages = [
            'org.neo4j:neo4j-connector-apache-spark_2.12:4.1.5_for_spark_3',
            'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
        ]

        with GraphDatabase.driver(uri=URI, auth=AUTH) as driver:
            with driver.session(database=DATABASE) as session:
                session.run("CREATE CONSTRAINT tab_id_constraint IF NOT EXISTS FOR (tab:Table) REQUIRE tab.table_id IS KEY") # KEY only with Enterprise edition
                session.run("CREATE CONSTRAINT tab_id_constraint IF NOT EXISTS FOR (tok:Token) REQUIRE tok.token_id IS KEY") # KEY only with Enterprise edition

        spark, initial_rdd = get_initial_spark_rdd(self.size, self.num_cpu, self.tables_thresholds, spark_jars_packages)
                         
        spark.conf.set('neo4j.authentication.basic.username', self.user)
        spark.conf.set('neo4j.authentication.basic.password', self.password)
        spark.conf.set('neo4j.url', URI)

        def prepare_tuple(t):
            # t = (_id_numeric, content, numeric_columns)
            _id_numeric, content, numeric_columns = t
            token_cnt = get_table_tokens_counter(content, numeric_columns)
            return _id_numeric, [(token, cnt) for token, cnt in token_cnt.items()]

        table_token_cnt = (
            initial_rdd
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

                # drop those tokens that have just one link to a table, since they won't give any
                # information about possible overlap (obv in future updates of the graph this
                # may be a relevant information loss)
                # .groupBy(lambda x: x[1])
                # .filter(lambda x: len(x[1]) > 1)
                # .flatMap(
                #     lambda x: [t for t in x[1]]
                # )
                .sortBy(
                    lambda tabid_tokid_tokcnt: (tabid_tokid_tokcnt[0], tabid_tokid_tokcnt[1])
                )
        )

        print("Total relationships: ", table_token_cnt.count())
        table_token_cnt = table_token_cnt.toDF(schema=['table_id', 'token_id', 'token_count'])
        # pprint(table_token_cnt.head(n=5))
        batch_size = 100000
        
        print(get_local_time(), ' Saving tables...')
        start = time()
        (
            table_token_cnt 
            .select('table_id')
            .distinct()
            .write
            .format("org.neo4j.spark.DataSource")
            .option("batch.size", batch_size)
            .mode("Overwrite")
            .option("labels", "Table")
            .option("node.keys", "table_id")
            .save()
        )
        print(get_local_time(), 'Completed. ', round(time() - start, 3))

        print(get_local_time(), ' Saving tokens...')
        start = time()
        (
            table_token_cnt 
            .select('token_id')
            .distinct()
            .write
            .format("org.neo4j.spark.DataSource")
            .mode("Overwrite")
            .option("batch.size", batch_size)
            .option("labels", "Token")
            .option("node.keys", "token_id")
            .save()
        )
        print(get_local_time(), 'Completed. ', round(time() - start, 3))

        # to avoid deadlocks (but this results into a very long, long time consuming task with zero parallelism...)
        # see https://neo4j.com/docs/spark/current/write/relationship/
        table_token_cnt = table_token_cnt.coalesce(1)

        # Here is assumed that Table and Token nodes are already created
        print('saving relationships...')
        start = time()
        (
            table_token_cnt
                .write
                .mode("Overwrite")
                .format("org.neo4j.spark.DataSource")
                .option("batch.size", batch_size)

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
        print(get_local_time(), 'Completed. ', round(time() - start, 3))

        return round(time() - start_data_prep, 5), sum(os.path.getsize(self.neo4j_db_dir + dbfile) for dbfile in os.listdir(self.neo4j_db_dir)) / (1024 ** 3)


    def query(self, results_file, query_ids, k, **kwargs):
        start = time()
        AUTH = (self.user, self.password)
        DATABASE = "neo4j"
        URI = f"bolt://localhost:7687"
            
        query = """
            UNWIND $query_ids AS _query_id
            CALL {
                WITH _query_id
                MATCH (n:Table WHERE n.table_id = _query_id)-[r:HAS]->(t:Token)<-[p:HAS]-(m:Table WHERE m.table_id <> _query_id)
                RETURN n.table_id AS query_id, m.table_id AS result_id,
                    SUM(
                        CASE 
                            WHEN r.token_count >= p.token_count THEN p.token_count 
                            ELSE r.token_count 
                        END
                    ) AS token_count 
                    ORDER BY token_count DESC 
                    LIMIT $k
            }
            RETURN 
                query_id,
                result_id,
                token_count
        """

        results = []
        steps = []
        s = 100
        with GraphDatabase.driver(uri=URI, auth=AUTH) as driver:
            with driver.session(database=DATABASE) as session:            
                for i in range(0, len(query_ids), s):
                    start = time()
                    results += session.run(query=query, parameters={"query_ids": query_ids[i:i+s], "k": k}).values()
                    steps.append(time() - start)

        (
            pd.DataFrame(results, columns=['query_id', 'result_id', 'token_count'])
            .pivot_table(values='token_count', index=['query_id', 'result_id'], aggfunc='sum')
            .reset_index(level=['query_id', 'result_id'])
            .sort_values(by=['query_id', 'token_count'], ascending=[False, False])
            .to_csv(results_file, index=False)
        )
        return round(time() - start, 5)
    

    def clean(self):
        print("Not implemented yet")


