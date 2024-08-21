import logging
import os
import pickle
from time import time
import multiprocessing as mp

import mmh3
import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from tools.utils.classes import AlgorithmTester
from tools.utils.misc import create_token_set, check_table_is_in_thresholds



def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def create_table_minhash(table, mode, numeric_columns, num_perm, blacklist):
    m = MinHash(num_perm=num_perm, hashfunc=_mmh3_hashfunc)
    token_set = create_token_set(table, mode, numeric_columns, encode='utf-8', blacklist=blacklist)
    m.update_batch(token_set)
    return m


def worker_lshforest_data_preparation(input):
    tables_thresholds, mode, num_perm, blacklist, table = input
    if table == None:
        return None, None
    _id_numeric, content, numeric_columns = table['_id_numeric'], table['content'], table['numeric_columns']

    if check_table_is_in_thresholds(content, tables_thresholds) and not all(numeric_columns):
        token_set = create_token_set(content, mode, numeric_columns, 'utf-8', blacklist)
        m = MinHash(num_perm, hashfunc=_mmh3_hashfunc)
        m.update_batch(token_set)
        return _id_numeric, m
    return _id_numeric, None


class LSHForestTester(AlgorithmTester):
    def __init__(self, mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper, *args) -> None:
        super().__init__(mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper)
        self.forest_file, self.num_perm, self.l = args

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

        total = self.datalake_helper.get_number_of_tables()
        logging.getLogger('TestLog').info(f"Start processing {total} tables...")
        with mp.Pool(processes=self.num_cpu) as pool:
            for i, result in enumerate(pool.imap(worker_lshforest_data_preparation, 
                                            (
                                                (self.tables_thresholds, self.mode, self.num_perm, self.blacklist, table)
                                                for table in self.datalake_helper.scan_tables()
                                            ), chunksize=100)):
                if i % 1000:
                    it_per_sec = round(i / (time() - start), 3)
                    print(f"{round(100 * i / total, 1)}%\t\t{it_per_sec}it/s      eta:{round((total - i) / it_per_sec, 3)}", end='\r')
                if result[1] == None:
                    continue
                self.forest.add(result[0], result[1])
        print(end='\r')
        logging.getLogger('TestLog').info("Indexing forest...")
        self.forest.index()

        logging.getLogger('TestLog').info("Saving forest...")
        self._forest_handler('save')
        
        logging.getLogger('TestLog').info("Completed LSH-Forest data preparation.")
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)



    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []

        if not self.forest:
            logging.getLogger('TestLog').info('Loading forest...')
            self._forest_handler('load')
            logging.getLogger('TestLog').info('Forest loaded.')
        for query_id in tqdm(query_ids):
            try:
                hashvalues_q = self.forest.get_minhash_hashvalues(query_id)
                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=_mmh3_hashfunc, hashvalues=hashvalues_q)
            except KeyError:
                table_q = self.datalake_helper.get_table_by_numeric_id(query_id)
                numeric_columns_q, content_q = table_q['numeric_columns'], table_q['content']
                
                token_set_q = create_token_set(content_q, self.mode, numeric_columns_q, blacklist=self.blacklist)

                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=_mmh3_hashfunc)
                minhash_q.update_batch(token_set_q)
                
            start_query = time()
            topk_res = self.forest.query(minhash_q, k + 1) # the "k+1" because often LSHForest returns the query table itself if it's already in the index
            end_query = time()
            
            if query_id in topk_res:
                topk_res.remove(query_id)
            
            results.append([query_id, round(end_query - start_query, 3), str(topk_res[:k]), str([])])

        logging.getLogger('TestLog').info(f' Saving results to {results_file}...')
        pl.DataFrame(results, schema=['query_id', 'duration', 'results_id', 'results_overlap'], orient='row').write_csv(results_file)
        return round(time() - start, 5)

    def clean(self):
        if os.path.exists(self.forest_file):
            os.remove(self.forest_file)

