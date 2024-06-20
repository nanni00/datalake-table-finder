import os
import bisect

import numpy as np
from time import time
from tqdm import tqdm

import faiss
import fasttext
import pandas as pd

import orjson

from tools.sloth.utils import parse_table
from tools.utils.utils import ( 
    check_table_is_in_thresholds,
    get_one_document_from_mongodb_by_key, 
    AlgorithmTester
)



class LUT:
    """
    LookUp Table used for keep the ids of indexed vectors on FAISS index
    """
    def __init__(self, json_lut_file=None) -> None:
        self.idxs = []
        self.ids = []

        if json_lut_file:
            self._load(json_lut_file)

    def insert_index(self, numitems, id):
        " Insert a new record in the LUT, assuming ordered insert. "
        self.idxs.append(numitems - 1 if len(self.idxs) == 0 else self.idxs[-1] + numitems)
        self.ids.append(id)

    def _load(self, json_lut_file):
        with open(json_lut_file, 'rb') as reader:
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



class FastTextTableEmbedder:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def embedding_table(self, table, numeric_columns):
        """ Return columns embeddings as a np.array """
        table = [column for i, column in enumerate(parse_table(table, len(table[0]), 0)) if numeric_columns[i] == 0]
        return np.array([self.model.get_sentence_vector(' '.join(column).replace('\n', ' ')) for column in table])



class EmbeddingTester(AlgorithmTester):
    def __init__(self, mode, small, tables_thresholds, num_cpu, *args) -> None:
        super().__init__(mode, small, tables_thresholds, num_cpu)
        self.model_path, self.clut_file, self.cidx_file = args
        
        self.model = FastTextTableEmbedder(self.model_path)
        self.clut = None
        self.cidx = None

    def data_preparation(self):
        start = time()
        self.cidx = faiss.IndexFlatL2(self.model.get_dimension())
        self.clut = LUT()

        for collection in self.collections:
            print(f'Starting pool working on {collection.database.name}.{collection.name}...')
            for doc in tqdm(collection.find({}, projection={"_id_numeric": 1, "content": 1, "numeric_columns": 1}), total=collection.count_documents({})):
                _id_numeric, content, numeric_columns = doc['_id_numeric'], doc['content'], doc['numeric_columns']
                
                if check_table_is_in_thresholds(content, self.tables_thresholds):
                    column_embeddings = self.model.embedding_table(content, numeric_columns)
                    
                    if column_embeddings.shape[0] > 0:
                        self.cidx.add(column_embeddings)
                        self.clut.insert_index(column_embeddings.shape[0], _id_numeric)

        self.clut.save(self.clut_file)
        faiss.write_index(self.cidx, self.cidx_file)

        return round(time() - start, 5), os.path.getsize(self.clut_file) / (1024 ** 3) + os.path.getsize(self.cidx_file) / (1024 ** 3)

    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        results = []
        for query_id in tqdm(query_ids):
            doc = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *self.collections)
            content, numeric_columns = doc['content'], doc['numeric_columns']
            cembeddings = self.model.embedding_table(content, numeric_columns)
            if cembeddings.shape[0] > 0:
                _, ids = self.cidx.search(cembeddings, k)
                ccnt = np.unique(np.vectorize(self.clut.lookup)(ids), return_counts=True)
                ctopk = sorted(zip(ccnt[0], ccnt[1]), key=lambda x: x[1], reverse=True)
                results.append([query_id, [x[0] for x in ctopk[:k] if x[0] != query_id]])
            else:
                results.append([query_id, []])
        pd.DataFrame(results, columns=['query_id', 'results']).to_csv(results_file, index=False)    
        return round(time() - start, 5)
    
    def clean(self):
        if os.path.exists(self.cidx_file):
            os.remove(self.cidx_file)
        
        if os.path.exists(self.clut_file):
            os.remove(self.clut_file)
    
