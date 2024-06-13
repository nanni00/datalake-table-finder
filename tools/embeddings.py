import json
import bisect

import numpy as np
from time import time
from tqdm import tqdm

import faiss
import fasttext
import pandas as pd

import orjson

from tools.sloth.utils import parse_table
from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import ( 
    check_table_is_in_thresholds,
    get_mongodb_collections, 
    get_one_document_from_mongodb_by_key, 
    get_query_ids_from_query_file, 
    sample_queries
)



class LUT:
    """
    LookUp Table used for keep the ids of indexed vectors on FAISS index
    """
    def __init__(self) -> None:
        self.idxs = []
        self.ids = []

    @classmethod
    def load():
        pass

    def insert_index(self, numitems, id):
        " Insert a new record in the LUT, assuming ordered insert. "
        self.idxs.append(numitems - 1 if len(self.idxs) == 0 else self.idxs[-1] + numitems)
        self.ids.append(id)

    def _load(self, json_lut_filepath:str):
        with open(json_lut_filepath, 'rb') as reader:
            json_lut = orjson.loads(reader.read())
            try:
                assert len(json_lut['idxs']) == len(json_lut['ids'])
                self.idxs = json_lut['idxs']
                self.ids = json_lut['ids']
            except AssertionError:
                print(f"idxs and table ids have different lengths: {len(json_lut['idxs'])}, {len(json_lut['table_ids'])}")

    def save(self, json_lut_filepath:str):
        with open(json_lut_filepath, 'wb') as writer:
            data = orjson.dumps({"idxs": self.idxs, "ids": self.ids})
            writer.write(data)
 
    def lookup(self, vector_id):
        return self.ids[bisect.bisect_left(self.idxs, vector_id)]

    @property
    def ntotal(self):
        return len(self.idxs)

def embedding_table(model:fasttext.FastText._FastText, table, numeric_columns):
    table = [column for i, column in enumerate(parse_table(table, len(table[0]), 0)) if numeric_columns[i] == 0]

    column_embedding = [model.get_sentence_vector(' '.join(column).replace('\n', ' ')) for column in table]

    return np.array(column_embedding)


def data_preparation(column_lut_file, column_index_file, model:fasttext.FastText._FastText, table_thresholds, *collections) -> tuple[LUT, faiss.IndexFlatL2]:
    cidx = faiss.IndexFlatL2(model.get_dimension())
    clut = LUT()

    for collection in collections:
        print(f'Starting pool working on {collection.database.name}.{collection.name}...')
        for doc in tqdm(collection.find({}, projection={"_id_numeric": 1, "content": 1, "numeric_columns": 1}), total=collection.count_documents({})):
            _id_numeric, content, numeric_columns = doc['_id_numeric'], doc['content'], doc['numeric_columns']
            
            if check_table_is_in_thresholds(content, table_thresholds):
                column_embeddings = embedding_table(model, content, numeric_columns)
                
                if column_embeddings.shape[0] > 0:
                    cidx.add(column_embeddings)
                    clut.insert_index(column_embeddings.shape[0], _id_numeric)

    clut.save(column_lut_file)
    faiss.write_index(cidx, column_index_file)

    return clut, cidx


def load_lut_index(lut_file, index_file):
    lut = LUT()
    lut._load(lut_file)
    return lut, faiss.read_index(index_file)


def query(result_file, model, clut:LUT, cidx:faiss.Index, query_ids, k, *collections):
    results = []
    for query_id in tqdm(query_ids):
        doc = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
        content, numeric_columns = doc['content'], doc['numeric_columns']
        cembeddings = embedding_table(model, content, numeric_columns)
        if cembeddings.shape[0] > 0:
            _, ids = cidx.search(cembeddings, k)
            ccnt = np.unique(np.vectorize(clut.lookup)(ids), return_counts=True)
            ctopk = sorted(zip(ccnt[0], ccnt[1]), key=lambda x: x[1], reverse=True)
            results.append([query_id, [x[0] for x in ctopk[:k] if x[0] != query_id]])
        else:
            results.append([query_id, []])
    pd.DataFrame(results, columns=['query_id', 'results']).to_csv(result_file, index=False)    


def main_query():
    clut = LUT()
    clut._load(test_dir + '/clut.json')
    cidx = faiss.read_index(test_dir + '/cindex.index')
    
    sample_queries(query_file, 10, tables_thresholds, *collections)
    query_ids = get_query_ids_from_query_file(query_file)
    k = 10
    
    query(result_file, clut, cidx, query_ids, k, *collections)


if __name__ == '__main__':
    small = True
    mongoclient, collections = get_mongodb_collections(small)
    test_dir = 'experiments/embedding'
    query_file = test_dir + '/query.json'
    result_file = test_dir + '/results.csv'

    tables_thresholds = {
        'min_rows':     5,
        'min_columns':  2,
        'min_area':     50,
        'max_rows':     999999,
        'max_columns':  999999,
        'max_area':     999999,
    }

    print('Loading fastText model...')
    start = time()
    model = fasttext.load_model(defpath.model_path.fasttext + '/cc.en.300.bin')
    print('loaded in ', (time() - start))

    # main_load()
    main_query()

    mongoclient.close()


