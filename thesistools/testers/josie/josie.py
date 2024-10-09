import os
import re
import logging
import binascii
from time import time

import mmh3
import pandas as pd
import polars as pl
import numpy as np

from thesistools.utils.logging_handler import info
from thesistools.testers.base_tester import AlgorithmTester
from thesistools.testers.josie.josiedb_handler import JOSIEDBHandler
from thesistools.utils.misc import is_valid_table, table_to_tokens, convert_to_giga
from thesistools.utils.spark import get_spark_session

import pyspark.storagelevel

def get_result_ids(s):
    return str(list(map(int, re.findall(r'\d+', s)[::2])))

def get_result_overlaps(s):
    return str(list(map(int, re.findall(r'\d+', s)[1::2])))




class JOSIETester(AlgorithmTester):
    def __init__(self, mode, blacklist, datalake_helper, token_translators,
                 table_prefix:str,
                 dbstatfile:str,
                 josie_db_connection_info:dict,
                 spark_config:dict) -> None:
        super().__init__(mode, blacklist, datalake_helper, token_translators)
        
        self.tables_prefix = table_prefix
        self.db_stat_file = dbstatfile
        self.josie_db_connection_info = josie_db_connection_info
        self.spark_config = spark_config
        
        self.josiedb = JOSIEDBHandler(self.tables_prefix, **self.josie_db_connection_info)
        
    def data_preparation(self):
        info('Creating integer sets and inverted index tables...')
        start = time()
        self.josiedb.drop_tables()
        self.josiedb.create_tables()
        
        mode, blacklist, dlh = self.mode, self.blacklist, self.dlh
        
        logging.getLogger('TestLog').info('Preparing inverted index and integer set tables...')
        spark, initial_rdd = get_spark_session(
            dlh.datalake_location, dlh.datalake_name, dlh.size,
            dlh.mapping_id, dlh.numeric_columns,
            **self.spark_config)

        def prepare_tuple(t):
            nonlocal mode, blacklist
            # t = (_id_numeric, content, numeric_columns)
            return [t[0], table_to_tokens(t[1], mode, t[2], blacklist=blacklist)]    
        
        token_sets = (
            initial_rdd
            .filter(
                # (_id_numeric, content, numeric_columns)
                lambda t: is_valid_table(t[1], t[2])
            )
            .map(
                # (_id_numeric, content, numeric_columns) -> (_id_numeric, [token1, token2, token3, ...])
                lambda t: prepare_tuple(t)
            ).flatMap(
                    # (set_id, [tok1, tok2, tok3, ...]) -> [(tok1, set_id), (tok2, set_id), ...]
                    lambda t: [(token, t[0]) for token in t[1]]
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
                lambda t: (t[0], t[1], mmh3.hash_bytes(np.array(t[1]))))
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
                lambda t: -1 if t[1][0][1] == t[1][1][2] and t[1][0][0] == t[1][1][1] else t[0])
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
                    lambda t: (t[0] - 1, t[1]))
            )
        )

        token_group_ids = (
            token_group_ids
            .flatMap(
                # GroupID, (startingIndexLower, startingIndexUpper) -> (tokenIndex, groupID)
                lambda t: [(i, t[0]) for i in range(t[1][0], t[1][1])])
            .persist(pyspark.StorageLevel.MEMORY_ONLY)
        )

        # join posting lists with their duplicate group IDs
        posting_lists_with_group_ids = (
            posting_lists_sorted
            .join(token_group_ids)
            .map(
                # (tokenIndex, ((rawToken, sids, _), gid)) -> (token_index, (group_id, raw_token, sids))
                lambda t: (t[0], (t[1][1], t[1][0][0], t[1][0][1])))
        )
        
        # STAGE 2: CREATE INTEGER SETS
        # Create sets and replace text tokens with token index
        integer_sets = (
            posting_lists_with_group_ids
            .flatMap(
                # (tokenIndex, (_, _, sids))
                lambda t: [(sid, t[0]) for sid in t[1][2]])
            .groupByKey()
            .map(
                # (sid, tokenIndexes)
                lambda t: (t[0], sorted(t[1]))
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
                    ])
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

        info(f"Total posting lists: {posting_lists.count()}")
        info(f"Total number of partitions: {posting_lists.getNumPartitions()}")
        
        def _integer_set_format(t):
            sid, indices = t
            return (sid, len(indices), len(indices), indices)
        
        def _postinglist_format(t):
            token, raw_token, gid, sets = t
            byteatoken = binascii.hexlify(bytes(str(raw_token), 'utf-8'))
            set_ids =  [int(s[0]) for s in sets]
            set_sizes = [int(s[1]) for s in sets]
            set_pos = [int(s[2]) for s in sets]
            return (int(token), len(sets), int(gid), 1, byteatoken, set_ids, set_sizes, set_pos)

        url = self.josiedb.url.create(
            self.josiedb.url.drivername, 
            host=self.josiedb.url.host, 
            port=self.josiedb.url.port, 
            database=self.josiedb.url.database).render_as_string()
        
        properties = {
            'user': self.josiedb.url.username,
            'password': self.josiedb.url.password,
        }


        (
            integer_sets
            .map(
                lambda t: _integer_set_format(t))
            .toDF(schema=['id', 'size', 'num_non_singular_token', 'tokens'])
            .write
            .jdbc(
                f'jdbc:{url}',
                f"{self.tables_prefix}_sets",
                'append',
                properties=properties
            )
        )

        (
            posting_lists
            .map(
                lambda t: _postinglist_format(t))
            .toDF(schema=['token', 'frequency', 'duplicate_group_id', 'duplicate_group_count', 'raw_token', 'set_ids', 'set_sizes', 'match_positions'])
            .write
            .jdbc(
                f'jdbc:{url}',
                f"{self.tables_prefix}_inverted_lists", 
                'append', 
                properties=properties
            )
        )

        spark.sparkContext.stop()
                
        # database statistics
        append = os.path.exists(self.db_stat_file)
        dbstat = pd.DataFrame(self.josiedb.get_statistics())
        dbstat.to_csv(self.db_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)
        
        info('Completed JOSIE data preparation.')
        return round(time() - start, 3), dbstat['total_size'].apply(convert_to_giga).sum()
        
    def query(self, results_file, k, query_ids, **kwargs):
        info('Starting JOSIE tests...')
        results_directory = kwargs['results_directory']
        token_table_on_memory = kwargs['token_table_on_memory']

        start_query = time()
        self.josiedb.clear_query_table()
        self.josiedb.add_queries_from_existent_tables(query_ids)

        GOPATH = os.environ['GOPATH']
        josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
        os.chdir(josie_cmd_dir)
        
        info(f'Check if sample tables already exist...')
        # if cost sampling tables already exist we assume they are correct and won't recreate them
        sample_costs_tables_exist = self.josiedb.cost_tables_exist()
        info(f'Sample costs tables exist? {sample_costs_tables_exist}')
        self.josiedb.close()

        if not sample_costs_tables_exist:
            info('Sampling costs...')
            os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                        --pg-database={self.josiedb.url.database} \
                        --test_tag={self.tables_prefix} \
                        --pg-table-queries={self.tables_prefix}_queries')

        # we are not considering the query preparation steps, since in some cases this will 
        # include also the cost sampling phase and in other cases it won't
        x = 'true' if token_table_on_memory else 'false'
        info('Using token table on memory: ' + x)

        info('Running top-K...')
        os.system(f'go run {josie_cmd_dir}/topk/main.go \
                    --pg-database={self.josiedb.url.database} \
                    --test_tag={self.tables_prefix} \
                    --outputDir={results_directory} \
                    --resultsFile={results_file} \
                    --useMemTokenTable={x} \
                    --k={k}')

        (
            pl.read_csv(results_file).select(['query_id', 'duration', 'results'])
            .with_columns((pl.col('duration') / 1000).name.keep())
            .with_columns(pl.col('results').map_elements(get_result_ids, return_dtype=pl.String).alias('result_ids'))
            .drop('results')
            .write_csv(results_file)
        )
        os.rename(results_file, results_file + '.raw')

        info('Completed JOSIE tests.')
        return round(time() - start_query, 5)

    def clean(self):
        self.josiedb.drop_tables()
        



if __name__ == '__main__':
    from thesistools.utils.datalake import SimpleDataLakeHelper
    from thesistools.utils.logging_handler import info, logging_setup
    from thesistools.utils.settings import DefaultPath as dp
    from thesistools.utils.misc import whitespace_translator, punctuation_translator, lowercase_translator

    mode = 'set'
    datalake = 'wikiturlsnap'
    size = 'small'
    blacklist = []
    dlh = SimpleDataLakeHelper('mongodb', datalake)
    num_cpu = 64
    tables_prefix = f"josie__{datalake}_{size}_{mode}"
    token_translators = [whitespace_translator, punctuation_translator, lowercase_translator]
    
    spark_local_dir = '/path/to/tmp/spark'
    # the Spark JAR for JDBC should not be inserted there, since it's a known issue
    # that a JAR passed as package here won't be retrieved as driver class
    spark_jars_packages = ['org.mongodb.spark:mongo-spark-connector_2.12:10.3.0']

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

    josie_db_connection_info = {
        'drivername':   'postgresql',
        'database':     'JOSIEDB',
        'username':     'nanni',
        'password':     '',
        'port':         5442,
        'host':         '127.0.0.1',
    }
    
    test_dir = f"{dp.data_path.tests}/new/{datalake}"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    logfile = f"{test_dir}/.logfile"
    db_stat_file = f"{test_dir}/.dbstat"

    logging_setup(logfile)
    tester = JOSIETester(mode, blacklist, dlh, token_translators,
                         tables_prefix, db_stat_file, josie_db_connection_info, spark_config)
    
    print(tester.data_preparation())