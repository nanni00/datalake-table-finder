import os
import re
import pickle
import binascii
from time import time
from typing import Dict, List

import mmh3
import numpy as np
import polars as pl
from bidict import bidict

import pyspark.storagelevel
from tqdm import tqdm

from dltf.utils.loghandler import error, info
from dltf.utils.misc import convert_to_giga
from dltf.utils.spark import get_spark_session
from dltf.utils.datalake import DataLakeHandler
from dltf.utils.tables import is_valid_table, table_to_tokens

from dltf.testers.base_tester import AbstractGlobalSearchAlgorithm
from dltf.testers.josie.db import JOSIEDBHandler
from dltf.testers.josie.exp import write_all_results
from dltf.testers.josie.josie_io import RawTokenSet
from dltf.testers.josie.josie_core import JOSIE
from dltf.testers.josie.tokentable import TokenTableDisk, TokenTableMem



def get_result_ids(s):
    return str(list(map(int, re.findall(r'\d+', s)[::2])))

def get_result_overlaps(s):
    return str(list(map(int, re.findall(r'\d+', s)[1::2])))




class JOSIEGS(AbstractGlobalSearchAlgorithm):
    def __init__(self, mode, blacklist, datalake_handler:DataLakeHandler, string_translators, string_patterns,
                 dbstatfile:str,
                 tokens_bidict_file:str,
                 josie_db_connection_info:dict,
                 spark_config:dict) -> None:
        super().__init__(mode, blacklist, datalake_handler, string_translators, string_patterns)
        self.db_stat_file = dbstatfile
        self.tokens_bidict_file = tokens_bidict_file
        self.josie_db_connection_info = josie_db_connection_info
        self.spark_config = spark_config
        
        self.db = JOSIEDBHandler(mode=self.mode, **self.josie_db_connection_info)
        
    def data_preparation(self):
        info('Creating integer sets and inverted index tables...')
        start = time()
        self.db.create_tables()
        
        # PySpark cannot access self.arg, like self.mode
        mode, blacklist, dlh, string_translators, string_patterns = self.mode, self.blacklist, self.dlh, self.string_translators, self.string_patterns
        
        info('Preparing inverted index and integer set tables...')
        spark, initial_rdd = get_spark_session(dlh, **self.spark_config)

        # initial_rdd = initial_rdd.filter(lambda t: t['num_header_rows'] == 1)
        # info(f'Tables with num_header_rows=1: {initial_rdd.count()}')
        
        token_sets = (
            initial_rdd
            .map(lambda t: [t['_id_numeric'], 
                            t['content'] if 'num_header_rows' not in t else t['content'][t['num_header_rows']:],
                            t['valid_columns']])
            .filter(
                # (_id_numeric, content, valid_columns)
                lambda t: is_valid_table(t[1], t[2])
            )
            .map(
                # (_id_numeric, content, valid_columns) -> (_id_numeric, [token1, token2, token3, ...])
                lambda t: [t[0], table_to_tokens(t[1], t[2], 
                                                 mode=mode, 
                                                 encode=None, 
                                                 blacklist=blacklist, 
                                                 string_translators=string_translators,
                                                 string_patterns=string_patterns)]
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
                # (tokenIndex, (group_id, raw_token, sids))
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
        n_posting_lists = posting_lists.count()
        n_sets = integer_sets.count()
        info(f"Total posting lists: {n_posting_lists}")
        info(f"Total integer sets: {n_sets}")
        
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

        url = self.db.url.create(
            self.db.url.drivername, 
            host=self.db.url.host, 
            port=self.db.url.port, 
            database=self.db.url.database).render_as_string()
        
        properties = {
            'user': self.db.url.username,
            'password': self.db.url.password,
        }


        (
            integer_sets
            .map(
                lambda t: _integer_set_format(t))
            .toDF(schema=['id', 'size', 'num_non_singular_token', 'tokens'])
            .write
            .jdbc(
                f'jdbc:{url}',
                f"{mode}__sets",
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
                f"{mode}__inverted_lists", 
                'append', 
                properties=properties
            )
        )
        
        spark.stop()

        # TODO this is inefficient, maybe could be improved/there are different ways?
        # create a bidict between the token IDs and the original string
        # store the original strings on the database is expensive, 
        # and also an index on such strings is unfeasible
        info('Creating the tokens bidictionary')
        info('Fetching pairs token-raw token...')
        raw_tokens = self.db.get_raw_tokens_from_to(0, n_posting_lists)
        info('Start creating the dictionary...')
        tokens_bidict = bidict({token: binascii.unhexlify(raw_token).decode('utf-8') for token, raw_token in raw_tokens})
        info(f'Bidictionary size: {len(tokens_bidict)}')
        info(f'Saving the dictionary to {self.tokens_bidict_file}...')
        with open(self.tokens_bidict_file, 'wb') as fw:
            pickle.dump(tokens_bidict, fw, protocol=pickle.HIGHEST_PROTOCOL)
        info('Done.')

        # save database statistics
        dbstat = pl.DataFrame(self.db.get_statistics(), orient='row')
        append = os.path.exists(self.db_stat_file)
        with open(self.db_stat_file, mode='a' if append else 'w') as fw:
            dbstat.write_csv(fw, include_header=False if append else True)
        
        info('Completed JOSIE data preparation.')
        return round(time() - start, 3), dbstat['total_size'].map_elements(convert_to_giga, return_dtype=pl.Float64).sum()
        
    def query(self, results_file, k, queries:List[int]|Dict[int,List[int]], reset_cost_function_parameters=False, force_sampling_cost=False, token_table_on_memory=False, verbose=False):
        """Run the query with the passed queries. If it is a list of integer, it assumes that 
        it's a list of table IDs already present in the index. Otherwise, it assumes that it's a 
        dictionary with pairs <QueryID, QueryTokenIDs>, where the token IDs are relative to tokens
        in the index, and they are already sorted. """
        
        if verbose: info('Starting JOSIE tests...')
        self.db.load_tables()
        
        start_query = time()
        if isinstance(queries, list):
            # insert the queries into the Queries table
            self.db.clear_query_table()
            self.db.add_queries_from_existing_tables(queries)
            # get the queries as tuples 
            # <query/set_id, list[token_ids], list[raw_tokens]>
            queries = [RawTokenSet(*row) for row in self.db.get_query_sets()]
        else:
            self.db.clear_query_table()
            self.db.add_queries(*zip(*queries.items()))
            queries = [RawTokenSet(qid, qtokens, self.db.get_raw_tokens(qtokens)) for qid, qtokens in queries.items()]

        if not self.db.are_costs_sampled() or force_sampling_cost:
            if verbose: info('Deleting old cost tables values...')
            self.db.delete_cost_tables()
            if verbose: info('Sampling costs...')
            self.db.sample_costs()

        # reset the cost function parameters used by JOSIE 
        # for the cost estimation
        if reset_cost_function_parameters:
            if verbose: info(f'Resetting cost function parameters...')
            self.db.reset_cost_function_parameters(verbose)

        if verbose: info(f"Number of sets: {self.db.count_sets()}")

        # create the token table, on memory or on disk
        if verbose: info(f"Creating token table on {'memory' if token_table_on_memory else 'disk'}...")
        tb = TokenTableMem(self.db, True) if token_table_on_memory else TokenTableDisk(self.db, True)

        if verbose: info(f"Begin experiment for {k=}...")
        
        perfs = []
        start = time()

        # execute the JOSIE algorithm for each query
        for q in tqdm(queries, disable=not verbose):
            try:
                perfs.append(JOSIE(self.db, tb, q, k, ignore_self=True)[1])
            except Exception as e:
                error(f'JOSIE error with query={q.set_id}: {e}')


        if verbose: info(f"Finished experiment for {k=} in {round((time() - start) / 60, 3)} minutes")
        
        # rewrite the results in a common format, i.e.
        # tuples <query_id, duration (s), list[result_table_ids]>
        write_all_results(perfs, f'{results_file}.raw')        
        (
            pl.read_csv(f'{results_file}.raw').select(['query_id', 'duration', 'results'])
            .with_columns((pl.col('duration') / 1000).name.keep())
            .with_columns(pl.col('results').map_elements(get_result_ids, return_dtype=pl.String).alias('result_ids'))
            .drop('results')
            .write_csv(results_file)
        )

        if verbose: info('Completed JOSIE tests.')
        return round(time() - start_query, 5)

    def clean(self):
        os.remove(self.tokens_bidict_file)
        self.db.drop_tables()

