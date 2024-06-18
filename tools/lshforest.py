import os
import pickle
from time import time
import mmh3
import pandas as pd
import multiprocessing as mp
from datasketch import MinHashLSHForest, MinHash

from tools.josie import AlgorithmTester
from tools.utils.utils import _create_token_set, check_table_is_in_thresholds
from tools.utils.utils import get_one_document_from_mongodb_by_key


def _worker_forest_creation(inp):
    doc, mode, num_perm, hashfunc, table_thresholds = inp
    _id_numeric, numeric_columns, content = doc['_id_numeric'], doc['numeric_columns'], doc['content']     

    if check_table_is_in_thresholds(content, table_thresholds):
        minhash = MinHash(num_perm=num_perm, hashfunc=hashfunc)
        token_set = _create_token_set(content, mode, numeric_columns)
        minhash.update_batch(token_set)
        # for token in token_set:
        #     minhash.update(token)

        return (_id_numeric, minhash)
    return (_id_numeric, None)


def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)




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

    def init_pool(hashfunc):
        hashfunc = hashfunc


    def data_preparation(self) -> MinHashLSHForest:
        start = time()
        if os.path.exists(self.forest_file):
            print('Loading forest...')
            self._forest_handler('load')
        else:
            print('Creating forest...')

            forest = MinHashLSHForest(num_perm=self.num_perm, l=self.l)
    
            with mp.Pool(processes=self.num_cpu) as pool:
                for collection in self.collections:
                    collsize = collection.count_documents({})
        
                    print(f'Starting pool working on {collection.database.name}.{collection.name}...')
                    for i, res in enumerate(pool.imap(
                                                _worker_forest_creation, 
                                                (
                                                    (doc, self.mode, self.num_perm, _mmh3_hashfunc, self.tables_thresholds) 
                                                    for doc in collection.find({}, 
                                                                        projection={"_id_numeric": 1, "numeric_columns": 1, "content": 1}
                                                    )
                                                ), 
                                                chunksize=500)):
                        if i % 1000 == 0:
                            print(round(100 * i / collsize, 3), '%', end='\r')
                        _id_numeric, minhash = res
                        if minhash:
                            forest.add(_id_numeric, minhash)
                    print(f'{collection.database.name}.{collection.name} updated.')        
                
                print('Indexing forest...')
                forest.index()
            print('Saving forest...')
            self._forest_handler('save')
             
        return round(time() - start, 5), os.path.getsize(self.forest_file) / (1024 ** 3)


    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []

        if not self.forest:
            self._forest_handler('load')

        for query_id in query_ids:
            try:
                hashvaluesq = self.forest.get_minhash_hashvalues(query_id)
                minhash_q = MinHash(self.num_perm, hashfunc=_mmh3_hashfunc, hashvalues=hashvaluesq)
            except KeyError:    
                docq = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *self.collections)
                
                numeric_columns_q, content_q = docq['numeric_columns'], docq['content']
                token_set_q = _create_token_set(content_q, self.mode, numeric_columns_q)

                minhash_q = MinHash(num_perm=self.num_perm, hashfunc=_mmh3_hashfunc)
                minhash_q.update_batch(token_set_q)
                
            topk_res = self.forest.query(minhash_q, k)
            if query_id in topk_res:
                topk_res.remove(query_id)
            results.append([query_id, topk_res])

        print(f'Saving results to {results_file}...')
        pd.DataFrame(results, columns=['query_id', 'results']).to_csv(results_file, index=False)
        return round(time() - start, 5)

    def clean(self):
        if os.path.exists(self.forest_file):
            os.remove(self.forest_file)

