import os
import re
import logging
import binascii
from time import time

import mmh3
import pandas as pd
import polars as pl
import numpy as np

import pymongo
import pyspark
import pyspark.rdd
import psycopg.rows

import psycopg

from tools.utils.classes import AlgorithmTester
from tools.utils.misc import (
    get_tables_thresholds_from,
    print_info, 
    create_token_set, 
    convert_to_giga, 
    get_spark_session, 
)




class JosieDB:
    def __init__(self, dbname, tables_prefix) -> None:
        self.dbname = dbname
        self.tables_prefix = tables_prefix

        self._dbconn = None

        self._SET_TABLE_NAME =               f'{self.tables_prefix}_sets'
        self._INVERTED_LISTS_TABLE_NAME =    f'{self.tables_prefix}_inverted_lists'
        self._SET_INDEX_NAME =               f'{self.tables_prefix}_sets_id_idx'
        self._INVERTED_LISTS_INDEX_NAME =    f'{self.tables_prefix}_inverted_lists_token_idx'
        self._QUERY_TABLE_NAME =             f'{self.tables_prefix}_queries'

        self._READ_LIST_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_list_cost_samples'
        self._READ_SET_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_set_cost_samples'

    @print_info(msg_before='Opening connection to the PostgreSQL database...')
    def open(self):
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}")

    def is_open(self):
        return not self._dbconn._closed

    @print_info(msg_before='Closing PostgreSQL database connection...', msg_after='PostgreSQL connection closed.')
    def close(self):
        self._dbconn.close()

    def _commit_decorator(f):
        def inner(self, *args, **kwargs):
            res = f(self, *args, **kwargs)
            self._dbconn.commit()
            return res
        return inner

    @_commit_decorator
    @print_info(msg_before='Dropping PostgreSQL database tables...', msg_after='PostgreSQL tables dropped.')
    def drop_tables(self):    
        self._dbconn.execute(
            f"""
            DROP TABLE IF EXISTS {self._INVERTED_LISTS_TABLE_NAME};
            DROP TABLE IF EXISTS {self._SET_TABLE_NAME};
            DROP TABLE IF EXISTS {self._QUERY_TABLE_NAME};

            DROP TABLE IF EXISTS {self._READ_LIST_COST_SAMPLES_TABLE_NAME};
            DROP TABLE IF EXISTS {self._READ_SET_COST_SAMPLES_TABLE_NAME};
            """
        )

    @_commit_decorator
    @print_info(msg_before='Creating PostgreSQL database tables...', msg_after='PostgreSQL tables created.')
    def create_tables(self):
        self._dbconn.execute(
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

            CREATE TABLE {self._SET_TABLE_NAME} (
                id integer NOT NULL,
                size integer NOT NULL,
                num_non_singular_token integer NOT NULL,
                tokens integer[] NOT NULL
            );

            CREATE TABLE {self._QUERY_TABLE_NAME} (
                id integer NOT NULL,
                tokens integer[] NOT NULL
            );
            """
        )

    @_commit_decorator
    @print_info(msg_before='Clearing PostgreSQL query table...', msg_after='PostgreSQL query table cleaned')
    def clear_query_table(self):
        self._dbconn.execute(
            f"""
                TRUNCATE {self._QUERY_TABLE_NAME}
            """
        )
            
    @_commit_decorator
    @print_info(msg_before='Inserting queries into PostgreSQL table...', msg_after='Queries inserted into PostgreSQL table.')
    def insert_data_into_query_table(self, table_ids:list[int]):
        self._dbconn.execute(
            f"""
            INSERT INTO {self._QUERY_TABLE_NAME} SELECT id, tokens FROM {self._SET_TABLE_NAME} WHERE id in {tuple(table_ids)};
            """
        )

    @_commit_decorator
    @print_info(msg_before='Creating PostgreSQL table sets index...', msg_after='Sets table index created.')
    def create_sets_index(self):
        self._dbconn.execute(
            f""" 
            DROP INDEX IF EXISTS {self._SET_INDEX_NAME}; 
            CREATE INDEX {self._SET_INDEX_NAME} ON {self._SET_TABLE_NAME}(id);
            """
        )

    @_commit_decorator
    @print_info(msg_before='Creating PostgreSQL table inverted list index...', msg_after='Inverted list table index created')
    def create_inverted_list_index(self):
        self._dbconn.execute(
            f"""
            DROP INDEX IF EXISTS {self._INVERTED_LISTS_INDEX_NAME};
            CREATE INDEX {self._INVERTED_LISTS_INDEX_NAME} ON {self._INVERTED_LISTS_TABLE_NAME}(token);
            """
        )

    @_commit_decorator
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
            WHERE i.relname LIKE '{self.tables_prefix}%'
            """
        with self._dbconn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            return cur.execute(q).fetchall()

    def cost_tables_exist(self):
        q = f"""
            SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_name   = '{self._READ_LIST_COST_SAMPLES_TABLE_NAME}'
               OR table_name = '{self._READ_SET_COST_SAMPLES_TABLE_NAME}'
            );
        """

        with self._dbconn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            return cur.execute(q).fetchall()[0]['exists'] == True



class JOSIETester(AlgorithmTester):
    def __init__(self, mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper, *args) -> None:
        super().__init__(mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper)
        (
            self.dbname, 
            self.tables_prefix, 
            self.db_stat_file, 
            self.pg_user, self.pg_password, 
            self.spark_local_dir
        ) = args
        
        self.josiedb = JosieDB(self.dbname, self.tables_prefix)
        self.josiedb.open()
        logging.info(f"Status PostgreSQL connection: {self.josiedb.is_open()}")

    @print_info(msg_before='Creating PostegreSQL integer sets and inverted index tables...', msg_after='Completed JOSIE data preparation.')
    def data_preparation(self):
        start = time()
        self.josiedb.drop_tables()
        self.josiedb.create_tables()

        # PostgreSQL write parameters
        url = f"jdbc:postgresql://localhost:5442/{self.dbname}"

        properties = {
            "user": self.pg_user,
            "password": self.pg_password,
            "driver": "org.postgresql.Driver"
        }
        
        spark_jars_packages = [
            'org.postgresql:postgresql:42.7.3',
            'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
        ]
        
        mode, blacklist, dlh = self.mode, self.blacklist, self.datalake_helper
        
        spark = get_spark_session(self.num_cpu, self.spark_local_dir, spark_jars_packages)
        MIN_ROW, MAX_ROW, MIN_COLUMN, MAX_COLUMN, MIN_AREA, MAX_AREA = get_tables_thresholds_from(self.tables_thresholds)

        match self.datalake_helper.datalake_location:
            # in the end we must have a RDD with tuples (table_id, table_content, table_numeric_columns)
            # with table_id as an integer,
            # table_content as a list of list (rows)
            # table_numeric_columns as a list of 0/1 values (0 => column i-th is not numeric, 1 otherwise)
            case 'mongodb':
                    # if the datalake is stored on MongoDB, then through the connector we
                    # can easily access the tables
                    mongoclient = pymongo.MongoClient(directConnection=True)
                    
                    match self.dataset:
                        case 'wikiturlsnap':
                            databases, collections = ['optitab', 'sloth'], ['turl_training_set', 'latest_snapshot_tables']
                        case 'gittables':
                            if 'sloth' in mongoclient.list_database_names():
                                databases = ['sloth']
                            elif 'dataset' in mongoclient.list_database_names():
                                databases = ['datasets']
                            collections = ['gittables']
                        case 'wikitables':
                            databases, collections = ['datasets'], ['wikitables']
                            
                    collections = [c + '_small' if self.size == 'small' else c for c in collections]
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
                            .filter(f"""
                                size(content) BETWEEN {MIN_ROW} AND {MAX_ROW}
                                AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN}
                                AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""")
                            .rdd
                            .map(list)
                        )
            case _:
                    # otherwise, if the datalake is stored on disk as CSV files, then we can access it using
                    # the helper class
                    initial_rdd = spark.sparkContext.parallelize([(id_num, id_name) for id_num, id_name in dlh._mapping_id.items()])

                    initial_rdd = initial_rdd.map(
                        lambda tabid_tabf: (tabid_tabf[0], pl.read_csv(f'{dlh.datalake_location}/{tabid_tabf[1]}.csv', infer_schema_length=0, encoding='latin1').rows())
                    ).filter(
                        lambda tid_tab: (MIN_ROW <= len(tid_tab[1]) <= MAX_ROW) \
                            and (MIN_COLUMN <= len(tid_tab[1][0]) <= MAX_COLUMN) \
                            and (MIN_AREA <= len(tid_tab[1]) * len(tid_tab[1][0]) <= MAX_AREA)            
                    ).map(
                        lambda tid_tab: (tid_tab[0], tid_tab[1], dlh._numeric_columns[tid_tab[0]])
                    )
        
        def prepare_tuple(t):
            nonlocal mode, blacklist
            # t = (_id_numeric, content, numeric_columns)
            return [t[0], create_token_set(t[1], mode, t[2], blacklist=blacklist)]    
        
        token_sets = (
            initial_rdd
                .map(
                    # from MongoDB directly
                    # (_id_numeric, content, numeric_columns) -> (_id_numeric, [token1, token2, token3, ...])
                    lambda t: prepare_tuple(t)
                ).flatMap(
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

        self.josiedb.create_sets_index()
        self.josiedb.create_inverted_list_index()

        # database statistics
        append = os.path.exists(self.db_stat_file)
        dbstat = pd.DataFrame(self.josiedb.get_statistics())
        dbstat.to_csv(self.db_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)
        
        return round(time() - start, 3), dbstat['total_size'].apply(convert_to_giga).sum()
        
    @print_info(msg_before='Starting JOSIE tests...', msg_after='Completed JOSIE tests.')
    def query(self, results_file, k, query_ids, **kwargs):
        results_directory = kwargs['results_directory']
        token_table_on_memory = kwargs['token_table_on_memory']

        self.josiedb.clear_query_table()
        self.josiedb.insert_data_into_query_table(query_ids)

        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)

        # if cost sampling tables already exist we assume they are correct and won't recreate them
        sample_costs_tables_exist = self.josiedb.cost_tables_exist()
        logging.info(f'Sample costs: {not sample_costs_tables_exist}')
        self.josiedb.close()

        if not sample_costs_tables_exist:
            logging.info('Sampling costs...')
            os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                        --pg-database={self.dbname} \
                        --test_tag={self.tables_prefix} \
                        --pg-table-queries={self.tables_prefix}_queries')

        # we are not considering the query preparation steps, since in some cases this will 
        # include also the cost sampling phase and in other cases it won't
        start_query = time()
        logging.info('Running top-K...')
        x = 'true' if token_table_on_memory else 'false'
        logging.info('Using token table on memory: ' + x)
        os.system(f'go run {josie_cmd_dir}/topk/main.go \
                    --pg-database={self.dbname} \
                    --test_tag={self.tables_prefix} \
                    --outputDir={results_directory} \
                    --resultsFile={results_file} \
                    --useMemTokenTable={x} \
                    --k={k}')

        # preparing output for next analyses
        results_df = pl.read_csv(results_file).select(['query_id', 'duration', 'results'])
        os.rename(results_file, results_file + '.raw')
        
        def get_result_ids(s):
            return str(list(map(int, re.findall(r'\d+', s)[::2])))
        
        def get_result_overlaps(s):
            return str(list(map(int, re.findall(r'\d+', s)[1::2])))
        
        results_df = results_df.with_columns((pl.col('duration') / 1000).name.keep())
        (   
            results_df
            .with_columns(
                pl.col('results').map_elements(get_result_ids, return_dtype=pl.String).alias('result_ids'),
                pl.col('results').map_elements(get_result_overlaps, return_dtype=pl.String).alias('result_overlaps'),
            )
            .drop('results')
            .write_csv(results_file)
        )

        return round(time() - start_query, 5)

    def clean(self):
        if not self.josiedb.is_open():
            self.josiedb.open()
        self.josiedb.drop_tables()
        self.josiedb.close()