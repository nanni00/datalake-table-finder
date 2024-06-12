import os
import json
import mmh3
import base64
from tqdm import tqdm

import orjson
import pymongo
import pandas as pd
import pymongo.collection
import multiprocessing as mp
from datasketch import MinHashLSHForest, MinHash

from tools.utils.utils import _create_token_set
from tools.utils.utils import get_one_document_from_mongodb_by_key



def _work_forest(inp):
    key, data, mode = inp
    encode_hash = lambda h: base64.b64encode(h).decode('utf-8')
    decode_hash = lambda h: base64.b64decode(h.encode('utf-8'))

    if key == 'keys':
        return 0, {k: [encode_hash(hash) if mode == 'save' else decode_hash(hash) for hash in v] for k, v in data.items()}
            
    elif key == 'hashtables':
        return 1, [{encode_hash(k) if mode == 'save' else decode_hash(k): v for k, v in hashtable.items()} for hashtable in data]
    
    elif key == 'sorted_hashtables':
        return 2, [[encode_hash(hash) if mode == 'save' else decode_hash(hash) for hash in hashtable] for hashtable in data]
        


def _forest_handler(forest_file, task, forest:None|MinHashLSHForest=None, num_perm=None):
    if task == 'save':
        with mp.Pool(3) as pool:
            rv = pool.map(_work_forest, [(key, data, task) for key, data in zip(
                                ['keys', 'hashtables', 'sorted_hashtables'], 
                                [forest.keys, forest.hashtables, forest.sorted_hashtables])], chunksize=1)
        rv = sorted(rv, key=lambda x: x[0])

        with open(forest_file, 'wb') as fwriter:
            json_bytes = orjson.dumps(
                {
                    'num_perm':             num_perm,
                    'k':                    forest.k,
                    'l':                    forest.l,
                    'hashranges':           forest.hashranges,
                    'keys':                 str(rv[0][1]),
                    'hashtables':           str(rv[1][1]),
                    'sorted_hashtables':    str(rv[2][1])                
                }    
            )
            fwriter.write(json_bytes)

    elif task == 'load':
        with open(forest_file) as freader:
            forest_data = orjson.loads(freader.read())

        forest = MinHashLSHForest(num_perm=forest_data['num_perm'], l=forest_data['l'])

        with mp.Pool(3) as pool:
            rv = pool.map(_work_forest, [(key, data, task) for key, data in zip(
                                ['keys', 'hashtables', 'sorted_hashtables'], 
                                [eval(forest_data['keys']), eval(forest_data['hashtables']), eval(forest_data['sorted_hashtables'])])], chunksize=1)
        rv = sorted(rv, key=lambda x: x[0])

        forest.k =                  forest_data['k']
        forest.hashranges =         forest_data['hashranges']
        forest.keys =               rv[0][1]
        forest.hashtables =         rv[1][1]
        forest.sorted_hashtables =  rv[2][1]

        return forest


def save_forest(forest, forest_file, num_perm):
    _forest_handler(forest_file, 'save', forest, num_perm)


def load_forest(forest_file) -> MinHashLSHForest:
    return _forest_handler(forest_file, 'load')


def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def _worker_forest_creation(inp):
    doc, mode, num_perm, hashfunc, MIN_ROW, MAX_ROW, MIN_COLUMN, MAX_COLUMN, MIN_AREA, MAX_AREA = inp
    _id_numeric, numeric_columns, content = doc['_id_numeric'], doc['numeric_columns'], doc['content']     

    if MIN_ROW <= len(content) <= MAX_ROW and \
        MIN_COLUMN <= len(content[0]) <= MAX_COLUMN and \
        MIN_AREA <= len(content) * len(content[0]) <= MAX_AREA: 
        minhash = MinHash(num_perm=num_perm, hashfunc=hashfunc)

        token_set = _create_token_set(content, mode, numeric_columns)
        minhash.update_batch(token_set)
        # for token in token_set:
        #     minhash.update(token)

        return (_id_numeric, minhash)
    return (_id_numeric, None)


def init_pool(hashfunc):
    hashfunc = hashfunc


def get_or_create_forest(forest_file, nworkers, num_perm, l, mode, hashfunc, tables_thresholds, *collections) -> MinHashLSHForest:
    if os.path.exists(forest_file):
        print('Loading forest...')
        forest = load_forest(forest_file)
    else:
        print('Creating forest...')                
        MIN_ROW =     tables_thresholds['min_rows']
        MAX_ROW =     tables_thresholds['max_rows']
        MIN_COLUMN =  tables_thresholds['min_columns']
        MAX_COLUMN =  tables_thresholds['max_columns']
        MIN_AREA =    tables_thresholds['min_area']
        MAX_AREA =    tables_thresholds['max_area']

        forest = MinHashLSHForest(num_perm=num_perm, l=l)
 
        with mp.Pool(processes=nworkers) as pool:
            for collection in collections:
                collsize = collection.count_documents({})
    
                print(f'Starting pool working on {collection.database.name}.{collection.name}...')
                for i, res in enumerate(pool.imap(
                                            _worker_forest_creation, 
                                            (
                                                (doc, mode, num_perm, hashfunc, MIN_ROW, MAX_ROW, MIN_COLUMN, MAX_COLUMN, MIN_AREA, MAX_AREA) 
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


def _get_or_create_forest(forest_file, num_perm, l, mode, hashfunc, tables_thresholds, *collections) -> MinHashLSHForest:
    if os.path.exists(forest_file):
        print('Loading forest...')
        forest = load_forest(forest_file)
    else:
        print('Creating forest...')
        forest = MinHashLSHForest(num_perm=num_perm, l=l)
        for collection in collections:
            print(f'Scanning documents from {collection.database.name}.{collection.name}...')
            for doc in tqdm(collection.find({}), total=collection.count_documents({})):
                minhash = MinHash(num_perm=num_perm, hashfunc=hashfunc)
                _id_numeric, numeric_columns, content = doc['_id_numeric'], doc['numeric_columns'], doc['content']
                            
                MIN_ROW =     tables_thresholds['min_rows']
                MAX_ROW =     tables_thresholds['max_rows']
                MIN_COLUMN =  tables_thresholds['min_columns']
                MAX_COLUMN =  tables_thresholds['max_columns']
                MIN_AREA =    tables_thresholds['min_area']
                MAX_AREA =    tables_thresholds['max_area']

                if MIN_ROW <= len(content) <= MAX_ROW and \
                    MIN_COLUMN <= len(content[0]) <= MAX_COLUMN and \
                    MIN_AREA <= len(content) * len(content[0]) <= MAX_AREA: 

                    token_set = _create_token_set(content, mode, numeric_columns)
                    # minhash.update_batch(token_set)
                    for token in token_set:
                        minhash.update(token)
                    forest.add(_id_numeric, minhash)
        
        forest.index()
        print('Saving forest...')
        save_forest(forest, forest_file, num_perm)
    return forest


def query_lsh_forest(results_file, forest:MinHashLSHForest, query_ids, mode, num_perm, k, hashfunc, *collections):
    # results_df = pd.DataFrame(columns=['query_id', 'res_id', 'rank', 'lshf_overlap', 'sloth_overlap', 'difference_lshf_sloth_overlap'])
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

        """
        print(f'Query ID: {query_id}, top-K: {topk_res}')
        for rank, res_id in enumerate(topk_res, start=1):
            doc_res = get_one_document_from_mongodb_by_key('_id_numeric', res_id, *collections)
            numeric_columns_res, content_res = doc_res['numeric_columns'], doc_res['content']

            token_set_res = _create_token_set(content_res, mode, numeric_columns_res)
            actual_set_overlap = len(set(token_set_res).intersection(token_set_res))

            largest_ov_sloth = apply_sloth(content_q, content_res, numeric_columns_q, numeric_columns_res)

            results_df.loc[len(results_df) - 1] = [query_id, res_id, rank, actual_set_overlap, largest_ov_sloth, actual_set_overlap - largest_ov_sloth]
        """

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
        query_lsh_forest(results_file, forest, query_ids, k, hashfunc, turlcoll, snapcoll)
    

if __name__ == '__main__':
    query_ids = [77, 88, 99, 111] # get_query_ids_from_query_file()

    forest_file = '/data4/nanni/tesi-magistrale/experiments/josie/forest.json'
    results_file = '/data4/nanni/tesi-magistrale/experiments/josie/results_lsh.csv'
    num_perm = 256
    mode = 'bag'
    k = 10
    LSHForest_testing(results_file, forest_file, query_ids, num_perm, mode, k, True, _mmh3_hashfunc)
    #LSHForest_testing(results_file, forest_file, query_ids, num_perm, mode, k, True, _mmh3_hashfunc)



