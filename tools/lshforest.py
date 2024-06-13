import os
import pickle
from time import time
import mmh3
from tqdm import tqdm
import pymongo
import pandas as pd
import pymongo.collection
import multiprocessing as mp
from datasketch import MinHashLSHForest, MinHash

from tools.utils.utils import _create_token_set, check_table_is_in_thresholds
from tools.utils.utils import get_one_document_from_mongodb_by_key


def _forest_handler(forest_file, task, forest:None|MinHashLSHForest=None, num_perm=None):
    if task == 'save':
        with open(forest_file, 'wb') as fwriter:
            pickle.dump(forest, fwriter)
            return
    elif task == 'load':
        with open(forest_file, 'rb') as freader:
            return pickle.load(freader)
    else:
        raise Exception('BOH!')


def save_forest(forest, forest_file, num_perm):
    _forest_handler(forest_file, 'save', forest, num_perm)


def load_forest(forest_file) -> MinHashLSHForest:
    return _forest_handler(forest_file, 'load')


def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


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


def init_pool(hashfunc):
    hashfunc = hashfunc


def get_or_create_forest(forest_file, nworkers, num_perm, l, mode, hashfunc, table_thresholds, *collections) -> MinHashLSHForest:
    if os.path.exists(forest_file):
        print('Loading forest...')
        forest = load_forest(forest_file)
    else:
        print('Creating forest...')

        forest = MinHashLSHForest(num_perm=num_perm, l=l)
 
        with mp.Pool(processes=nworkers) as pool:
            for collection in collections:
                collsize = collection.count_documents({})
    
                print(f'Starting pool working on {collection.database.name}.{collection.name}...')
                for i, res in enumerate(pool.imap(
                                            _worker_forest_creation, 
                                            (
                                                (doc, mode, num_perm, hashfunc, table_thresholds) 
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
        save_forest(forest, forest_file, num_perm)
    return forest


def query(results_file, forest:MinHashLSHForest, query_ids, mode, num_perm, k, hashfunc, *collections):
    results = []

    for query_id in query_ids:
        try:
            hashvaluesq = forest.get_minhash_hashvalues(query_id)
            minhash_q = MinHash(num_perm, hashfunc=hashfunc, hashvalues=hashvaluesq)
        except KeyError:    
            docq = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
            
            numeric_columns_q, content_q = docq['numeric_columns'], docq['content']
            token_set_q = _create_token_set(content_q, mode, numeric_columns_q)

            minhash_q = MinHash(num_perm=num_perm, hashfunc=hashfunc)
            # minhash_q.update_batch(token_set_q)
            for token in token_set_q:
                minhash_q.update(token)

        topk_res = forest.query(minhash_q, k)
        if query_id in topk_res:
            topk_res.remove(query_id)
        results.append([query_id, topk_res])

    print(f'Saving results to {results_file}...')
    pd.DataFrame(results, columns=['query_id', 'results']).to_csv(results_file, index=False)


def LSHForest_testing(results_file, forest_file, query_ids, k, num_perm, mode, small, hashfunc=None):
    with pymongo.MongoClient() as mongoclient:
        if not small:
            turlcoll = mongoclient.optitab.turl_training_set
            snapcoll = mongoclient.sloth.latest_snapshot_tables
        else:
            turlcoll = mongoclient.optitab.turl_training_set_small
            snapcoll = mongoclient.sloth.latest_snapshot_tables_small

        forest = get_or_create_forest(forest_file, num_perm, mode, small, hashfunc, turlcoll, snapcoll)
        query(results_file, forest, query_ids, k, hashfunc, turlcoll, snapcoll)
    

if __name__ == '__main__':
    query_ids = [77, 88, 99, 111] # get_query_ids_from_query_file()

    forest_file = '/data4/nanni/tesi-magistrale/experiments/josie/forest.json'
    results_file = '/data4/nanni/tesi-magistrale/experiments/josie/results_lsh.csv'
    num_perm = 256
    mode = 'bag'
    k = 10
    LSHForest_testing(results_file, forest_file, query_ids, num_perm, mode, k, True, _mmh3_hashfunc)
    #LSHForest_testing(results_file, forest_file, query_ids, num_perm, mode, k, True, _mmh3_hashfunc)



