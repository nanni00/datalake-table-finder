import os
import pickle
from time import time

import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from dltf.utils.spark import get_spark_session
from dltf.utils.tables import is_valid_table, table_to_tokens
from dltf.utils.loghandler import info
from dltf.testers.base_tester import AlgorithmTester


class LSHForestTester(AlgorithmTester):
    def __init__(self, mode, blacklist, dlh, token_translators, num_cpu, forest_file, num_perm, l, hash_func, spark_config) -> None:
        super().__init__(mode, blacklist, dlh, token_translators)
        self.forest_file = forest_file
        self.num_perm = num_perm
        self.l = l
        self.hash_func = hash_func
        self.spark_config = spark_config

        self.forest = None
    
    def data_preparation(self):
        spark, rdd = get_spark_session(self.dlh, **self.spark_config)
        mode, blacklist, token_translators, num_perm, hash_func = self.mode, self.blacklist, self.token_translators, self.num_perm, self.hash_func
        forest = MinHashLSHForest(num_perm=self.num_perm, l=self.l)

        def create_minhash(table, valid_columns):
            nonlocal mode, blacklist, token_translators, num_perm, hash_func
            token_set = table_to_tokens(table, valid_columns, mode, 'utf-8', blacklist, token_translators)
            m = MinHash(num_perm, hashfunc=hash_func)
            m.update_batch(token_set)
            return m
            
        info(f"Start processing tables...")
        start = time()
        rdd = (
            rdd
            .map(lambda t: [t['_id_numeric'], 
                            t['content'] if 'num_header_rows' not in t else t['content'][t['num_header_rows']:],
                            t['valid_columns']])
            .filter(lambda t: is_valid_table(t[1], t[2]))
            .map(lambda t: (t[0], create_minhash(t[1], t[2])))
        )
        
        def add_to_forest(t):
            nonlocal forest
            forest.add(*t)
        rdd.foreach(add_to_forest)

        spark.sparkContext.stop()
        self.forest = forest

        info("Indexing forest...")
        self.forest.index()
        
        info("Saving forest...")
        with open(self.forest_file, 'wb') as fwriter:
            pickle.dump(self.forest, fwriter)
        
        info("Completed LSH-Forest data preparation.")
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)
    

    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []

        if not self.forest:
            info('Loading forest...')
            with open(self.forest_file, 'rb') as freader:
                self.forest = pickle.load(freader)

        info('Running top-K...')
        for query_id in tqdm(query_ids):
            try:
                hashvalues_q = self.forest.get_minhash_hashvalues(query_id)
                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=self.hash_func, hashvalues=hashvalues_q)
            except KeyError:
                table_q = self.dlh.get_table_by_numeric_id(query_id)
                valid_columns_q, content_q = table_q['valid_columns'], table_q['content']
                
                token_set_q = table_to_tokens(content_q, valid_columns_q, self.mode, blacklist=self.blacklist, string_transformers=self.token_translators)

                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=self.hash_func)
                minhash_q.update_batch(token_set_q)
                
            start_query = time()
            # the "k+1" because often LSHForest returns 
            # the query table itself if it's already in the index
            topk_res = self.forest.query(minhash_q, k + 1) 
            end_query = time()
            
            if query_id in topk_res:
                topk_res.remove(query_id)
            
            results.append([query_id, round(end_query - start_query, 3), str(topk_res[:k])])

        info(f' Saving results to {results_file}...')
        pl.DataFrame(results, schema=['query_id', 'duration', 'results_id'], orient='row').write_csv(results_file)
        return round(time() - start, 5)

    def clean(self):
        if os.path.exists(self.forest_file):
            os.remove(self.forest_file)
