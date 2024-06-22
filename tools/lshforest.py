import os
import pickle
from time import time
import multiprocessing as mp

import mmh3
import pandas as pd
import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from tools.utils.utils import _create_token_set, check_table_is_in_thresholds, get_initial_spark_rdd, AlgorithmTester, get_one_document_from_mongodb_by_key


def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def create_table_minhash(table, mode, numeric_columns, num_perm):
    m = MinHash(num_perm=num_perm, hashfunc=_mmh3_hashfunc)
    token_set = _create_token_set(table, mode, numeric_columns, encode='utf-8')
    m.update_batch(token_set)
    return m


def _worker(input):
    tables_thresholds, mode, num_perm, document = input
    _id_numeric, content, numeric_columns = document['_id_numeric'], document['content'], document['numeric_columns']

    if check_table_is_in_thresholds(content, tables_thresholds) and not all(numeric_columns):
        token_set = _create_token_set(content, mode, numeric_columns, 'utf-8')
        m = MinHash(num_perm, hashfunc=_mmh3_hashfunc)
        m.update_batch(token_set)
        return _id_numeric, m
    return _id_numeric, None


class LSHForestTester(AlgorithmTester):
    def __init__(self, mode, small, tables_thresholds, num_cpu, *args) -> None:
        super().__init__(mode, small, tables_thresholds, num_cpu)
        self.forest_file, self.num_perm, self.l, self.collections = args

        self.forest = None
    
    def _forest_handler(self, task):
        if task == 'save':
            with open(self.forest_file, 'wb') as fwriter:
                pickle.dump(self.forest, fwriter)
                return
        elif task == 'load':
            with open(self.forest_file, 'rb') as freader:
                self.forest = pickle.load(freader)


    def data_preparation(self):
        start = time()

        self.forest = MinHashLSHForest(self.num_perm, self.l)

        total = sum(c.count_documents({}) for c in self.collections)

        with mp.Pool(processes=self.num_cpu) as pool:
            for i, result in enumerate(pool.imap(_worker, 
                                            (
                                                (self.tables_thresholds, self.mode, self.num_perm, document)
                                                for collection in self.collections
                                                for document in collection.find({}, projection={"_id_numeric": 1, "content": 1, "numeric_columns": 1})
                                            ), chunksize=500)):
                if i % 10000:
                    it_per_sec = round(i / (time() - start), 3)
                    print(f"{round(100 * i / total, 3)}%\t\t{it_per_sec}it/s      eta:{round((total - i) / it_per_sec, 3)}", end='\r')
                if result[1] == None:
                    continue
                self.forest.add(result[0], result[1])
        print(end='\r')
        print("Indexing forest...")
        self.forest.index()

        print("Saving forest...")
        self._forest_handler('save')
        
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)



    def __data_preparation(self):
        # TODO why is spark so slow compared to the multiprocessing version? ~20min vs ~5min???
        start = time()

        spark_jars_packages = [
            'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
        ]

        _, initial_rdd = get_initial_spark_rdd(self.small, self.num_cpu, self.tables_thresholds, spark_jars_packages)

        self.forest = MinHashLSHForest(self.num_perm, self.l)

        mode = self.mode
        num_perm = self.num_perm
        initial_rdd = initial_rdd.map(lambda t: (t[0], create_table_minhash(t[1], mode, t[2], num_perm)))
        total = initial_rdd.count()


        # reducing number of partitions it seems that time goes down, but boh
        for t in tqdm(initial_rdd.cache().toLocalIterator(prefetchPartitions=True), total=total, position=2):
            self.forest.add(t[0], t[1])
        
        print("Indexing forest...")
        self.forest.index()

        print("Saving forest...")
        self._forest_handler('save')
        
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)
        

    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []

        if not self.forest:
            print('Loading forest...')
            self._forest_handler('load')

        for query_id in tqdm(query_ids):
            try:
                hashvalues_q = self.forest.get_minhash_hashvalues(query_id)
                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=_mmh3_hashfunc, hashvalues=hashvalues_q)
            except KeyError:
                docq = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *self.collections)
                
                numeric_columns_q, content_q = docq['numeric_columns'], docq['content']
                token_set_q = _create_token_set(content_q, self.mode, numeric_columns_q)

                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=_mmh3_hashfunc)
                minhash_q.update_batch(token_set_q)
                
            start_query = time()
            topk_res = self.forest.query(minhash_q, k + 1) # the "k+1" because often LSHForest returns the query table itself if it's already in the index
            end_query = time()
            
            if query_id in topk_res:
                topk_res.remove(query_id)
            
            results.append([query_id, round(end_query - start_query, 3), str(topk_res[:k]), str([])])

        print(f'Saving results to {results_file}...')
        pl.DataFrame(results, schema=['query_id', 'duration', 'results_id', 'results_overlap']).write_csv(results_file)
        return round(time() - start, 5)

    def clean(self):
        if os.path.exists(self.forest_file):
            os.remove(self.forest_file)

