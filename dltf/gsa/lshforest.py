import os
import pickle
from time import time
from typing import Dict, List

import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from dltf.utils.loghandler import info
from dltf.utils.spark import get_spark_session
from dltf.utils.tables import is_valid_table, table_to_tokens
from dltf.gsa.base_tester import AbstractGlobalSearchAlgorithm


class LSHForestGS(AbstractGlobalSearchAlgorithm):
    def __init__(self, mode, dlh, string_blacklist, string_translators, string_patterns, forest_file, num_perm, l, hash_func, spark_config) -> None:
        super().__init__(mode, dlh, string_blacklist, string_translators, string_patterns)
        self.forest_file    = forest_file
        self.num_perm       = num_perm
        self.l              = l
        self.hash_func      = hash_func
        self.spark_config   = spark_config
        self.forest         = None
    
    def data_preparation(self):
        mode, string_blacklist, string_translators, string_patterns, num_perm, hash_func = \
            self.mode, self.string_blacklist, self.string_translators, self.string_patterns, self.num_perm, self.hash_func
        
        def create_minhash(table, valid_columns):
            nonlocal mode, string_blacklist, string_translators, num_perm, hash_func
            token_set = table_to_tokens(table, valid_columns, mode, 'utf-8', string_blacklist, string_translators, string_patterns)
            m = MinHash(num_perm, hashfunc=hash_func)
            m.update_batch(token_set)
            return m
        
        self.forest = MinHashLSHForest(num_perm=self.num_perm, l=self.l)
        spark, rdd = get_spark_session(self.dlh, **self.spark_config)
            
        info(f"Start processing tables...")
        start = time()
        minhashes = (
            rdd
            .map(lambda t: [t['_id_numeric'], 
                            t['content'] if 'num_header_rows' not in t else t['content'][t['num_header_rows']:],
                            t['valid_columns']])
            .filter(lambda t: is_valid_table(t[1], t[2]))
            .map(lambda t: (t[0], create_minhash(t[1], t[2])))
            .collect()
        )

        for qid, minhash in tqdm(minhashes, leave=False):
            self.forest.add(qid, minhash)

        spark.sparkContext.stop()
        
        info("Indexing forest...")
        self.forest.index()
        print(len(self.forest.keys))
        info("Saving forest...")
        with open(self.forest_file, 'wb') as fwriter:
            pickle.dump(self.forest, fwriter, protocol=pickle.HIGHEST_PROTOCOL)
        
        info("Completed LSH-Forest data preparation.")
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)
    

    def query(self, queries:List[int]|Dict[int,List], k:int, results_file:str=None):
        start = time()
        results = []

        if not self.forest:
            info('Loading forest...')
            with open(self.forest_file, 'rb') as freader:
                self.forest = pickle.load(freader)

        format_queries = []
        if isinstance(queries, list):
            for qid in queries:
                if qid in self.forest.keys:
                    format_queries.append([qid, MinHash(num_perm=self.num_perm, hashfunc=self.hash_func, hashvalues=self.forest.get_minhash_hashvalues(qid))])
                else:
                    table_obj = self.dlh.get_table_by_numeric_id(qid)
                    valid_columns, table = table_obj['valid_columns'], table_obj['content']
                    tokens = table_to_tokens(table, valid_columns, self.mode, 'utf-8', 
                                             self.string_blacklist, self.string_translators, self.string_patterns)
                    minhash = MinHash(num_perm=self.num_perm, hashfunc=self.hash_func)
                    minhash.update_batch(tokens)
                    format_queries.append([qid, minhash])
        elif isinstance(queries, dict):
            for qid, table in queries.items():
                tokens = table_to_tokens(table, [1] * len(table[0]), self.mode, 'utf-8',
                                         self.string_blacklist, self.string_translators, self.string_patterns)
                minhash = MinHash(num_perm=self.num_perm, hashfunc=self.hash_func)
                minhash.update_batch(tokens)
                format_queries.append([qid, minhash])
        
        info('Running top-K...')
        for qid, minhash in tqdm(format_queries):    
            start_query = time()
            # the "k+1" because often LSHForest returns 
            # the query table itself if it's already in the index
            topk = self.forest.query(minhash, k + 1) 
            end_query = time()
            
            if qid in topk:
                topk.remove(qid)
            results.append([qid, round(end_query - start_query, 3), topk[:k]])

        if results_file:
            info(f' Saving results to {results_file}...')
            pl.DataFrame([[qid, t, str(r)] for qid, t, r in results], schema=['query_id', 'duration', 'results_id'], orient='row').write_csv(results_file)
        return round(time() - start, 5), [[qid, r] for qid, _, r in results]

    def clean(self):
        if os.path.exists(self.forest_file):
            os.remove(self.forest_file)
