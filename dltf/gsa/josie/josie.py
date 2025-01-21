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

from dltf.gsa.base_tester import AbstractGlobalSearchAlgorithm
from dltf.gsa.josie.db import JOSIEDBHandler
from dltf.gsa.josie.exp import write_all_results
from dltf.gsa.josie.josie_io import RawTokenSet
from dltf.gsa.josie.josie_core import JOSIE
from dltf.gsa.josie.tokentable import TokenTableDisk, TokenTableMem

__all__ = ['JOSIEGS']

def get_result_ids(s):
    return str(list(map(int, re.findall(r'\d+', s)[::2])))

def get_result_overlaps(s):
    return str(list(map(int, re.findall(r'\d+', s)[1::2])))




class JOSIEGS(AbstractGlobalSearchAlgorithm):
    def __init__(self, mode, datalake_handler:DataLakeHandler, 
                 string_blacklist, string_translators, string_patterns,
                 dbstatfile:str,
                 tokens_bidict_file:str,
                 josie_db_connection_info:dict,
                 spark_config:dict) -> None:
        super().__init__(mode, datalake_handler, string_blacklist, string_translators, string_patterns)
        self.db_stat_file = dbstatfile
        self.tokens_bidict_file = tokens_bidict_file
        self.josie_db_connection_info = josie_db_connection_info
        self.spark_config = spark_config
        
        # Create the database handler
        self.db = JOSIEDBHandler(mode=self.mode, **self.josie_db_connection_info)

        self.tokens_bidict = None
        
    def data_preparation(self):
        info('Creating integer sets and inverted index tables...')
        start = time()
        self.db.create_tables()
        
        # PySpark cannot access self.arg
        mode, dlh, string_blacklist, string_translators, string_patterns = (
            self.mode, self.dlh, self.string_blacklist, self.string_translators, self.string_patterns
        )
        
        info('Preparing inverted index and integer set tables...')
        spark, initial_rdd = get_spark_session(dlh, **self.spark_config)

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
                                                 string_blacklist=string_blacklist, 
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
        
    def query(self, queries:List[int]|Dict[int,List], k:int, results_file:str=None, 
              reset_cost_function_parameters:bool=False, force_sampling_cost:bool=False, token_table_on_memory:bool=False, verbose:bool=False):
        info('Starting JOSIE tests...')

        info('Loading database tables...')
        self.db.load_tables()

        if not self.tokens_bidict:
            info('Loading tokens bi-dictionary...')
            with open(self.tokens_bidict_file, 'rb') as fr:
                self.tokens_bidict = pickle.load(fr)
            
        if isinstance(queries, list):
            self.db.clear_query_table()
            self.db.add_queries_from_existing_tables(queries)
            queries = [RawTokenSet(*row) for row in self.db.get_query_sets()]
        elif isinstance(queries, dict):
            queries = {
                qid: list(table_to_tokens(table, [1] * len(table[0]), self.mode, None, 
                                     self.string_blacklist, self.string_translators, self.string_patterns))
                for qid, table in queries.items()
            }
            try:
                queries = {
                    qid: [self.tokens_bidict.inverse[raw_token] for raw_token in qset]
                    for qid, qset in queries.items()
                }
                set_error = False
            except KeyError:
                for qid, qset in queries.items():
                    if '{{ken}}@#38' in qset:
                        print(f'>>> {qid}')
                        set_error = True
            if set_error:
                raise Exception()
            self.db.clear_query_table()
            self.db.add_queries(*zip(*queries.items()))
            queries = [RawTokenSet(qid, qtokens, self.db.get_raw_tokens(qtokens)) for qid, qtokens in queries.items()]

        if not self.db.are_costs_sampled() or force_sampling_cost:
            info('Deleting old cost tables values...')
            self.db.delete_cost_tables()
            info('Sampling costs...')
            self.db.sample_costs()

        # reset the cost function parameters used by JOSIE 
        # for the cost estimation
        if reset_cost_function_parameters:
            info(f'Resetting cost function parameters...')
            self.db.reset_cost_function_parameters(verbose)

        info(f"Number of sets: {self.db.count_sets()}")

        # create the token table, on memory or on disk
        info(f"Creating token table on {'memory' if token_table_on_memory else 'disk'}...")
        tb = TokenTableMem(self.db, True) if token_table_on_memory else TokenTableDisk(self.db, True)

        info(f"Begin experiment for {k=}...")
        
        perfs = []
        start = time()

        start_query = time()
        # execute the JOSIE algorithm for each query
        for q in tqdm(queries):
            try:
                perfs.append(JOSIE(self.db, tb, q, k, ignore_self=True))
            except Exception as e:
                error(f'JOSIE error with query={q.set_id}: {e}')

        info(f"Finished experiment for {k=} in {round((time() - start) / 60, 3)} minutes")
        
        # rewrite the results in a common format, i.e.
        # tuples <query_id, duration (s), list[result_table_ids]>
        if results_file is not None:
            write_all_results([p[1] for p in perfs], f'{results_file}.raw')        
            (
                pl.read_csv(f'{results_file}.raw').select(['query_id', 'duration', 'results'])
                .with_columns((pl.col('duration') / 1000).name.keep())
                .with_columns(pl.col('results').map_elements(get_result_ids, return_dtype=pl.String).alias('result_ids'))
                .drop('results')
                .write_csv(results_file)
            )

        results = [
            [p[1].query_id, list(zip(eval(get_result_ids(p[1].results)), eval(get_result_overlaps(p[1].results))))]
            for p in perfs
        ]

        return round(time() - start_query, 5), results

    def clean(self):
        os.remove(self.tokens_bidict_file)
        self.db.drop_tables()

