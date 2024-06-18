import os
from time import time
import mmh3
import pandas as pd
import pymongo.collection

import pymongo
import binascii
import jsonlines
import numpy as np
from tqdm import tqdm

import pyspark
from pyspark.sql import SparkSession

import psycopg
import psycopg.sql

from tools.utils.utils import _create_token_set, convert_to_giga, print_info




class JosieDB:
    def __init__(self, dbname, table_prefix) -> None:
        self.dbname = dbname
        self.tprefix = table_prefix

        self._dbconn = None
        self._dbcur = None

        self._SET_TABLE_NAME =               f'{self.tprefix}_sets'
        self._INVERTED_LISTS_TABLE_NAME =    f'{self.tprefix}_inverted_lists'
        self._SET_INDEX_NAME =               f'{self.tprefix}_sets_id_idx'
        self._INVERTED_LISTS_INDEX_NAME =    f'{self.tprefix}_inverted_lists_token_idx'
        self._QUERY_TABLE_NAME =             f'{self.tprefix}_queries'

        self._READ_LIST_COST_SAMPLES_TABLE_NAME = f'{self.tprefix}_read_list_cost_samples'
        self._READ_SET_COST_SAMPLES_TABLE_NAME = f'{self.tprefix}_read_set_cost_samples'

    @print_info(msg_before='Opening connection to the PostgreSQL database...')
    def open(self):
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}")
        self._dbcur = None

    def is_open(self):
        return not self._dbconn._closed

    @print_info(msg_before='Closing connection...')
    def close(self):
        self._dbconn.close()

    def _commit_dec(f):
        def inner(self, *args, **kwargs):
            self._dbcur = self._dbconn.cursor(row_factory=psycopg.rows.dict_row)
            res = f(self, *args, **kwargs)
            self._dbcur.close()
            self._dbconn.commit()
            return res
        return inner

    @_commit_dec
    @print_info(msg_before='Dropping tables...')
    def drop_tables(self, all=False):    
        self._dbcur.execute(
            f"""
            DROP TABLE IF EXISTS {self._INVERTED_LISTS_TABLE_NAME};
            DROP TABLE IF EXISTS {self._SET_TABLE_NAME};
            DROP TABLE IF EXISTS {self._QUERY_TABLE_NAME};
            """        
        )

        if all:
            self._dbcur.execute(
                f"""
                DROP TABLE IF EXISTS {self._READ_LIST_COST_SAMPLES_TABLE_NAME};
                DROP TABLE IF EXISTS {self._READ_SET_COST_SAMPLES_TABLE_NAME};
                """
            )

    @_commit_dec
    @print_info(msg_before='Creating database tables...')
    def create_tables(self):
        self._dbcur.execute(
            f"""              
            CREATE TABLE {self._INVERTED_LISTS_TABLE_NAME} (
                token integer NOT NULL,
                frequency integer NOT NULL,
                duplicate_group_id integer NOT NULL,
                duplicate_group_count integer NOT NULL,
                raw_token bytea NOT NULL,
                set_ids integer[] NOT NULL,
                set_sizes integer[] NOT NULL,
                match_positions integer[] NOT NULL
            );        
            """
        )

        self._dbcur.execute(
            f"""          
                CREATE TABLE {self._SET_TABLE_NAME} (
                    id integer NOT NULL,
                    size integer NOT NULL,
                    num_non_singular_token integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )

        self._dbcur.execute(
            f"""
                CREATE TABLE {self._QUERY_TABLE_NAME} (
                    id integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )

    @_commit_dec
    @print_info(msg_before='Clearing query table...')
    def clear_query_table(self):
        self._dbcur.execute(
            f"""
                TRUNCATE {self._QUERY_TABLE_NAME}
            """
        )
            
    @_commit_dec
    @print_info(msg_before='Inserting queries...')
    def insert_data_into_query_table(self, table_ids:list[int]):
        # maybe is better to translate all in postgresql...
        self._dbcur.execute(
            f"""
            INSERT INTO {self._QUERY_TABLE_NAME} SELECT id, tokens FROM {self._SET_TABLE_NAME} WHERE id in {tuple(table_ids)};
            """
        )

    @_commit_dec
    @print_info(msg_before='Creating table sets index...')
    def create_sets_index(self):
        self._dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._SET_INDEX_NAME}; """
        )
        self._dbcur.execute(
            f"""CREATE INDEX {self._SET_INDEX_NAME} ON {self._SET_TABLE_NAME}(id);"""
        )

    @_commit_dec
    @print_info(msg_before='Creating inverted list index...')
    def create_inverted_list_index(self):
        self._dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._INVERTED_LISTS_INDEX_NAME}; """
        )

        self._dbcur.execute(
            f"""CREATE INDEX {self._INVERTED_LISTS_INDEX_NAME} ON {self._INVERTED_LISTS_TABLE_NAME}(token);"""
        )

    @_commit_dec
    def get_statistics(self):
        q = f"""
            SELECT 
                i.relname 
                "table_name",
                indexrelname "index_name",
                pg_size_pretty(pg_total_relation_size(relid)) as "total_size",
                pg_size_pretty(pg_relation_size(relid)) as "table_size",
                pg_size_pretty(pg_relation_size(indexrelid)) "index_size",
                reltuples::bigint "estimated_table_row_count"
            FROM pg_stat_all_indexes i JOIN pg_class c ON i.relid = c.oid 
            WHERE i.relname LIKE '{self.tprefix}%'
            """
        
        return self._dbcur.execute(q).fetchall()





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


class AlgorithmTester:
    def __init__(self, mode, small, tables_thresholds, num_cpu) -> None:
        self.mode = mode
        self.small = small
        self.num_cpu = num_cpu
        self.tables_thresholds = tables_thresholds


    def data_preparation(self) -> None:
        pass

    def query(self, results_file, k, query_ids, *args) -> None:
        pass

    def clean(self) -> None:
        pass



class JOSIETester(AlgorithmTester):
    def __init__(self, mode, small, tables_thresholds, num_cpu, *args) -> None:
        super().__init__(mode, small, tables_thresholds, num_cpu)        
        self.dbname, self.table_prefix, self.db_stat_file = args

        self.josiedb = JosieDB(self.dbname, self.table_prefix)
        self.josiedb.open()

    # filtering on the IDs from the SLOTH results file should be deprecated, since now we'll work
    # on a silver standard per test
    @print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.')
    def data_preparation(self):

        start = time()
        
        self.josiedb.drop_tables()
        self.josiedb.create_tables()
        self.josiedb.create_inverted_list_index()
        self.josiedb.create_sets_index()

        MIN_ROW =     self.tables_thresholds['min_row']
        MAX_ROW =     self.tables_thresholds['max_row']
        MIN_COLUMN =  self.tables_thresholds['min_column']
        MAX_COLUMN =  self.tables_thresholds['max_column']
        MIN_AREA =    self.tables_thresholds['min_area']
        MAX_AREA =    self.tables_thresholds['max_area']

        # PostgreSQL write parameters
        url = "jdbc:postgresql://localhost:5442/nanni"

        properties = {
            "user": "nanni",
            "password": "",
            "driver": "org.postgresql.Driver"
        }
        
        spark_jars_packages = [
            'org.postgresql:postgresql:42.7.3',
            'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
        ]

        # fine tune of executor/driver.memory?
        builder = SparkSession.Builder()
        spark = (
            builder
            .appName("Big Bang Testing with MongoDB")
            .master(f"local[{self.num_cpu}]")
            .config('spark.jars.packages', ','.join(spark_jars_packages))
            .config('spark.executor.memory', '100g')
            .config('spark.driver.memory', '10g')
            .config('spark.local.dir', '/data4/nanni/spark')
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
            .option("collection", "turl_training_set" if not self.small else "turl_training_set_small") 
            .load() 
            .select('_id_numeric', 'content', 'numeric_columns') 
            .filter(f"""
                    size(content) BETWEEN {MIN_ROW} AND {MAX_ROW}
                    AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN}
                    AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""")
        )

        initial_rdd = (
            spark
            .read 
            .format('mongodb')
            .option("uri", "mongodb://127.0.0.1:27017/") 
            .option("database", "sloth") 
            .option("collection", "latest_snapshot_tables" if not self.small else "latest_snapshot_tables_small") 
            .load() 
            .select('_id_numeric', 'content', 'numeric_columns') 
            .filter(f"""
                    size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
                    AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
                    AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""") 
            .rdd
            .map(list)
            .union(
                optitab__turl_training_set_df.rdd.map(list)    
            )
        )
        
        # free memory used by the dataframe (is this really useful?)
        optitab__turl_training_set_df.unpersist()   

        def prepare_tuple(t):
            # t = (_id_numeric, content, numeric_columns)
            return [t[0], _create_token_set(t[1], self.mode, t[2])]    
        
        token_sets = (
            initial_rdd
                .map(
                    # from MongoDB directly
                    # (_id_numeric, content, numeric_columns) -> (_id_numeric, [token1, token2, token3, ...])
                    lambda t: prepare_tuple(t)
                )
                    .flatMap(
                        # (set_id, [tok1, tok2, tok3, ...]) -> [(tok1, set_id), (tok2, set_id), ...]
                        lambda t: [ (token, t[0]) for token in t[1]]
                    )
        )

        posting_lists_sorted = (
            token_sets
                .groupByKey()
                    .map(
                        # (token, [set_idK, set_idJ, set_idM, ...]) -> (token, [set_id1, set_id2, set_id3, ..., set_idZ]) 
                        lambda t: (t[0], sorted(list(t[1]))))
                        .map(
                            # (token, set_ids) -> (token, set_ids, set_ids_hash)
                            lambda t:
                                (
                                    t[0], 
                                    t[1], 
                                    mmh3.hash_bytes(np.array(t[1]))
                                )
                            )
                            .sortBy(
                                # t: (token, setIDs, hash)
                                lambda t: (len(t[1]), t[2], t[1]))
                                .zipWithIndex()
                                    .map(
                                        # t: ((rawToken, sids, hash), tokenIndex) -> (token_id, (raw_token, set_ids, set_ids_hash))
                                        lambda t: (t[1], t[0]))
                                        .persist(pyspark.StorageLevel.MEMORY_ONLY)
        )

        # create the duplicate groups
        duplicate_group_ids = (
            posting_lists_sorted
                .map(
                    # t: (tokenIndex, (rawToken, sids, hash)) -> (token_index, (sids, hash))
                    lambda t: (t[0] + 1, (t[1][1], t[1][2])))
                    .join(posting_lists_sorted)
                        .map(
                            lambda t: 
                                -1 if t[1][0][1] == t[1][1][2] and t[1][0][0] == t[1][1][1] else t[0]
                        )
                            .filter(
                                lambda i: i > 0)
                                .union(spark.sparkContext.parallelize([0, posting_lists_sorted.count()]))
                                    .sortBy(lambda i: i)
                                        .zipWithIndex()
                                            .map(
                                                # returns a mapping from group ID to the
                                                # starting index of the group
                                                # (startingIndex, GroupID) -> (GroupID, startingIndex)
                                                lambda t: (t[1], t[0]))
        )
            
        # generating all token indexes of each group
        token_group_ids = (
            duplicate_group_ids
                .join( # (GroupIDLower, startingIndexLower) JOIN (GroupIDUpper, startingIndexUpper) 
                    duplicate_group_ids
                        .map(
                            # (GroupID, startingIndexUpper) -> (GroupID, startingIndexUpper)
                            lambda t: (t[0] - 1, t[1])
                        )
                    )
        )

        token_group_ids = (
            token_group_ids
                .flatMap(
                    # GroupID, (startingIndexLower, startingIndexUpper) -> (tokenIndex, groupID)
                    lambda t: [(i, t[0]) for i in range(t[1][0], t[1][1])]
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)
        )

        # join posting lists with their duplicate group IDs
        posting_lists_with_group_ids = posting_lists_sorted \
            .join(token_group_ids) \
                .map(
                    # (tokenIndex, ((rawToken, sids, _), gid)) -> (token_index, (group_id, raw_token, sids))
                    lambda t: (t[0], (t[1][1], t[1][0][0], t[1][0][1]))
                )
        
        # STAGE 2: CREATE INTEGER SETS
        # Create sets and replace text tokens with token index
        integer_sets = (
            posting_lists_with_group_ids
                .flatMap(
                    # (tokenIndex, (_, _, sids))
                    lambda t: [(sid, t[0]) for sid in t[1][2]]        
                )
                    .groupByKey()
                        .map(
                            # (sid, tokenIndexes)
                            lambda t: (
                                t[0], 
                                sorted(t[1])
                            )
                        )
        )

        # STAGE 3: CREATE THE FINAL POSTING LISTS
        # Create new posting lists and join the previous inverted
        # lists to obtain the final posting lists with all the information
        posting_lists = (
            integer_sets
                .flatMap(
                    lambda t:
                        [
                            (token, (t[0], len(t[1]), pos))
                            for pos, token in enumerate(t[1])
                        ]
                )
                    .groupByKey()
                        .map(
                            # (token, sets)
                            lambda t: (
                                t[0], 
                                sorted(t[1], 
                                    key=lambda s: s[0]
                                    )
                            )
                        )
                            .join(posting_lists_with_group_ids)
                                .map(
                                    # (token, (sets, (gid, rawToken, _))) -> (token, rawToken, gid, sets)
                                    lambda t: (t[0], t[1][1][1], t[1][1][0], t[1][0])
                                )
        )

        # STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS
        # to load directly into the database

        def _set_format_psql(t):
            sid, indices = t
            return (sid, len(indices), len(indices), indices)
        
        def _postinglist_format_psql(t):
            token, raw_token, gid, sets = t
            byteatoken = binascii.hexlify(bytes(str(raw_token), 'utf-8'))
            set_ids =  [s[0] for s in sets]
            set_sizes = [s[1] for s in sets]
            set_pos = [s[2] for s in sets]

            return (token, len(sets), gid, 1, byteatoken, set_ids, set_sizes, set_pos)

        integer_sets.map(
            lambda t: _set_format_psql(t)
        ).toDF(schema=['id', 'size', 'num_non_singular_token', 'tokens']).write.jdbc(url, self.tables_prefix + '_sets', 'overwrite', properties)

        posting_lists.map(
            lambda t: _postinglist_format_psql(t)
        ).toDF(schema=[
            'token', 'frequency', 'duplicate_group_id', 'duplicate_group_count', 
            'raw_token', 'set_ids', 'set_sizes', 'match_positions'
            ]
        ).write.jdbc(url, self.tables_prefix + '_inverted_lists', 'overwrite', properties)

        # database statistics
        append = os.path.exists(self.db_stat_file)
        dbstat = pd.DataFrame(self.josiedb.get_statistics())
        dbstat.to_csv(self.db_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

        return round(time() - start, 3), dbstat['total_size'].apply(convert_to_giga).sum()
        


    @print_info(msg_before='Starting JOSIE tests...', msg_after='Completed.')
    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results_directory = kwargs['results_directory']
        self.josiedb.clear_query_table()
        self.josiedb.insert_data_into_query_table(query_ids)
        
        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)

        os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                    --pg-database={self.dbname} \
                    --test_tag={self.tables_prefix} \
                    --pg-table-queries={self.tables_prefix}_queries')

        os.system(f'go run {josie_cmd_dir}/topk/main.go \
                    --pg-database={self.dbname} \
                    --test_tag={self.tables_prefix} \
                    --outputDir={results_directory} \
                    --resultsFile={results_file} \
                    --k={k}')
        
        self.josiedb.close()
        return round(time() - start, 5)

    def clean(self):
        if not self.josiedb.is_open():
            self.josiedb.open()
        self.josiedb.drop_tables(all=True)
        self.josiedb.close()