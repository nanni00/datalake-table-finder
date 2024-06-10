import mmh3
import base64
import json
import pandas as pd
import pymongo
import os
from datasketch import MinHashLSHForest, MinHash
import pymongo.collection
from tqdm import tqdm

from tools.utils.utils import _create_token_set
from tools.utils.utils import get_one_document_from_mongodb_by_key

 

def save_forest(forest, forest_file, num_perm):
    encode_hash = lambda h: base64.b64encode(h).decode('utf-8')
    with open(forest_file, 'w') as fwriter:
        json.dump(
            {
                'num_perm':             num_perm,
                'k':                    forest.k,
                'l':                    forest.l,
                'hashranges':           forest.hashranges,
                'keys':                 {k: [encode_hash(hash) for hash in v] for k, v in forest.keys.items()},
                'hashtables':           [{encode_hash(k): v for k, v in hashtable.items()} for hashtable in forest.hashtables],
                'sorted_hashtables':    [[encode_hash(hash) for hash in hashtable] for hashtable in forest.sorted_hashtables]
            },
            fwriter
        )


def load_forest(forest_file) -> MinHashLSHForest:
    with open(forest_file) as freader:
        forest_data = json.load(freader)

    forest = MinHashLSHForest(num_perm=forest_data['num_perm'], l=forest_data['l'])
    decode_hash = lambda h: base64.b64decode(h.encode('utf-8'))

    forest.k =                  forest_data['k']
    forest.hashranges =         forest_data['hashranges']
    forest.keys =               {k: [decode_hash(hash) for hash in v] for k, v in forest_data['keys'].items()}
    forest.hashtables =         [{decode_hash(k): v for k, v in hashtable.items()} for hashtable in forest_data['hashtables']]
    forest.sorted_hashtables =  [[decode_hash(hash) for hash in hashtable] for hashtable in forest_data['sorted_hashtables']]

    return forest


def _mmh3_hashfunc(d):
    return mmh3.hash(d, signed=False)


def get_or_create_forest(forest_file, num_perm, l, mode, hashfunc, tables_thresholds, *collections) -> MinHashLSHForest:
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



