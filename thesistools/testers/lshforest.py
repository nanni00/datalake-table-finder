import os
import pickle
from time import time
import multiprocessing as mp

import polars as pl
from tqdm import tqdm
from datasketch import MinHashLSHForest, MinHash

from thesistools.testers.base_tester import AlgorithmTester
from thesistools.utils.misc import table_to_tokens, is_valid_table
from thesistools.utils.logging_handler import info
from thesistools.utils.parallel import chunks
from thesistools.utils.datalake import SimpleDataLakeHelper



def worker_lshforest_data_preparation(data):
    global datalake_location, datalake, size, mapping_id_file, numeric_columns_file, \
        mode, blacklist, token_translators, num_perm, hash_func
    
    id_tables_range = data[0]
    results = []

    dlh = SimpleDataLakeHelper(datalake_location, datalake, size, mapping_id_file, numeric_columns_file)
    for i in id_tables_range:
        table_obj = dlh.get_table_by_numeric_id(i)
        _id_numeric, table, numeric_columns = table_obj['_id_numeric'], table_obj['content'], table_obj['numeric_columns']

        if is_valid_table(table, numeric_columns):
            token_set = table_to_tokens(table, mode, numeric_columns, 'utf-8', blacklist, *token_translators)
            m = MinHash(num_perm, hashfunc=hash_func)
            m.update_batch(token_set)
            results.append([_id_numeric, m])
    return results





def initializer(_datalake_location, _datalake, _size, _mapping_id_file, _numeric_columns_file, 
                _mode, _num_cpu, _blacklist, _token_translators, _num_perm, _hash_func, _emb_model_path):
    global datalake_location, datalake, size, mapping_id_file, numeric_columns_file, \
        mode, num_cpu, blacklist, token_translators, num_perm, hash_func, emb_model_path
    
    datalake_location = _datalake_location
    datalake =          _datalake
    size =              _size
    mapping_id_file =   _mapping_id_file
    numeric_columns_file = _numeric_columns_file

    mode =              _mode
    num_cpu =           _num_cpu
    blacklist =         _blacklist
    token_translators = _token_translators

    num_perm =          _num_perm
    hash_func =         _hash_func
    emb_model_path =    _emb_model_path

    
class LSHForestTester(AlgorithmTester):
    def __init__(self, mode, blacklist, datalake_helper, token_translators, num_cpu, forest_file, num_perm, l, hash_func) -> None:
        super().__init__(mode, blacklist, datalake_helper, token_translators)
        self.num_cpu = num_cpu
        self.forest_file = forest_file
        self.num_perm = num_perm
        self.l = l
        self.hash_func = hash_func

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

        initargs = (
            self.dlh.datalake_location, self.dlh.datalake_name, self.dlh.size, 
            self.dlh.mapping_id_path, self.dlh.numeric_columns_path,
            self.mode, self.num_cpu, self.blacklist, self.token_translators, self.num_perm, self.hash_func, None)
        work = range(self.dlh.get_number_of_tables())

        info(f"Start processing tables...")
        with mp.Pool(processes=self.num_cpu, initializer=initializer, initargs=initargs) as pool:
            r = pool.map(worker_lshforest_data_preparation, chunks(work, max(len(work) // self.num_cpu, 1)))
            for process_results in r:
                for result in process_results:
                    self.forest.add(result[0], result[1])

        info("Indexing forest...")
        self.forest.index()
        
        info("Saving forest...")
        self._forest_handler('save')
        
        info("Completed LSH-Forest data preparation.")
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)



    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []

        if not self.forest:
            info('Loading forest...')
            self._forest_handler('load')

        info('Running top-K...')
        for query_id in tqdm(query_ids):
            try:
                hashvalues_q = self.forest.get_minhash_hashvalues(query_id)
                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=self.hash_func, hashvalues=hashvalues_q)
            except KeyError:
                table_q = self.dlh.get_table_by_numeric_id(query_id)
                numeric_columns_q, content_q = table_q['numeric_columns'], table_q['content']
                
                token_set_q = table_to_tokens(content_q, self.mode, numeric_columns_q, blacklist=self.blacklist)

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




if __name__ == '__main__':
    from thesistools.utils.logging_handler import logging_setup
    from thesistools.utils.settings import DefaultPath as dp
    from thesistools.utils.misc import mmh3_hashfunc, whitespace_translator, punctuation_translator, lowercase_translator

    mode = 'set'
    datalake = 'wikiturlsnap'
    size = 'small'
    blacklist = []
    dlh = SimpleDataLakeHelper('mongodb', datalake, size)
    token_translators = [whitespace_translator, punctuation_translator, lowercase_translator]
    num_cpu = 64
    num_perm = 256
    l = 16
    hash_func = mmh3_hashfunc

    test_dir = f"{dp.data_path.tests}/new/{datalake}"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    logfile = f"{test_dir}/.logfile"
    logging_setup(logfile)

    db_stat_file = f"{test_dir}/.dbstat"
    forest_file = f"{test_dir}/{mode}_forest.index"
    results_file = f"{test_dir}/results.csv"

    tester = LSHForestTester(mode, blacklist, dlh, token_translators, 
                             num_cpu, forest_file, num_perm, l, hash_func)
    
    print(tester.data_preparation())

    query_ids = [0, 1, 2, 3, 4, 5]
    print(tester.query(results_file, 10, query_ids))