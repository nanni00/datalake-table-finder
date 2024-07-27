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
    get_local_time,
    get_one_document_from_mongodb_by_key,
    prepare_token,
    AlgorithmTester
)



class LUT:
    """
    LookUp Table used for keep the ids of indexed vectors on FAISS index
    --- Deprecated: FAISS indexes allow to use non-unique IDs for vectors, in
    this way the column vectors of a table can share the same ID, so when doing the
    top-K it's no longer needed a LUT to retrieve for each specifc vector ID its 
    table ID.
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



class TableEmbedder:
    def __init__(self, model_path) -> None:
        pass

    def get_dimension(self) -> int:
        pass

    def embedding_table(self, table, numeric_columns, **kwargs):
        pass


class FastTextTableEmbedder(TableEmbedder):
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def embedding_table(self, table, numeric_columns, **kwargs):
        """ 
        Return columns embeddings as a np.array of shape (#table_columns, #model_vector_size)

        For fastText.get_sentence_vector() see the github repo at src/fasttext.cc, lines 490-521
        it takes a sentence, splits it on blank spaces and for each word computes and normalises its 
        embedding; then gets the average of the normalised embeddings
        """
        table = [column for i, column in enumerate(parse_table(table, len(table[0]), 0)) if numeric_columns[i] == 0]
        return np.array([self.model.get_sentence_vector(' '.join(map(str, column)).replace('\n', ' ')) for column in table])

    def get_dimension(self):
        return self.model.get_dimension()



class TaBERTTableEmbedder(TableEmbedder):
    def __init__(self, model_path) -> None:
        from tools.table_bert.table import Table, Column
        from tools.table_bert.table_bert import TableBertModel
        self.model = TableBertModel.from_pretrained(model_name_or_path=model_path)

    def get_dimension(self):
        return self.model.output_size

    def _prepare_table(self, table, numeric_columns):
        return Table(
            id=None,
            header=[Column(f'Column_{i}', type='text') for i, _ in enumerate(numeric_columns) if numeric_columns[i] == 0],
            data=[[prepare_token(cell) for i, cell in enumerate(row) if numeric_columns[i] == 0] for row in table]
        ).tokenize(self.model.tokenizer)

    def embedding_table(self, table, numeric_columns, context=""):
        _, column_embedding, _ = self.model.encode(
            contexts=[self.model.tokenizer.tokenize(context)],
            tables=[self._prepare_table(table, numeric_columns)])
        return column_embedding.detach().numpy()[0]

    def embedding_table_batch(self, tables, n_numeric_columns):
        tables = [self._prepare_table(table, numeric_columns) for (table, numeric_columns) in zip(tables, n_numeric_columns)]

        _, column_embeddings, _ = self.model.encode(
            contexts=[self.model.tokenizer.tokenize("") for _ in range(len(tables))],
            tables=tables)

        column_embeddings = column_embeddings.detach().numpy()
        # print(len(tables), len(n_numeric_columns), column_embeddings.shape)
        return [column_embeddings[i, :len(table.header), :] for i, table in enumerate(tables)]



class EmbeddingTester(AlgorithmTester):
    def __init__(self, mode, small, tables_thresholds, num_cpu, *args) -> None:
        super().__init__(mode, small, tables_thresholds, num_cpu)
        self.model_path, self.cidx_file, collections = args
        self.collections = collections
        self.cidx = None
        
        if mode == 'fasttext':
            self.model = FastTextTableEmbedder(self.model_path)
        elif mode == 'tabert':
            self.model = TaBERTTableEmbedder(self.model_path)

    def _tabert_data_preparation(self):
        """Btw, takes too long without a GPU, not concretely applicable"""
        batch_size = 50
        
        batch_ids = []
        batch_tables = []
        batch_numeric_columns = []
        
        def process_batch(batch_ids, batch_tables, batch_numeric_columns):
            column_embeddings = self.model.embedding_table_batch(batch_tables, batch_numeric_columns)
            for (_id, colemb) in zip(batch_ids, column_embeddings):
                if colemb.shape[0] > 0:
                    self.cidx.add_with_ids(colemb, np.array([_id] * colemb.shape[0], dtype=np.int64))
            
        for collection in self.collections:
            print(f'Starting pool working on {collection.database.name}.{collection.name}...')
            for doc in tqdm(collection.find({}, projection={"_id_numeric": 1, "content": 1, "numeric_columns": 1}), total=collection.count_documents({})):
                _id_numeric, content, numeric_columns = doc['_id_numeric'], doc['content'], doc['numeric_columns']
                
                if check_table_is_in_thresholds(content, self.tables_thresholds) and not all(numeric_columns):
                    batch_tables.append(content)
                    batch_ids.append(_id_numeric)
                    batch_numeric_columns.append(numeric_columns)
                    if len(batch_tables) == batch_size:
                        process_batch(batch_ids, batch_tables, batch_numeric_columns)
                        batch_ids = []
                        batch_tables = []
                        batch_numeric_columns = []
            process_batch(batch_ids, batch_tables, batch_numeric_columns)

    def data_preparation(self):
        start = time()
        self.cidx = faiss.index_factory(self.model.get_dimension(), "Flat,IDMap2")
        
        if self.mode == 'tabert':
            # since TaBERT can compute encodings for batch of tables, use it
            # however, if a gpu isn't available, even with a good cpu computing
            # embeddings for a huge corpora is not achievable
            self._tabert_data_preparation()
        else:
            for collection in self.collections:
                print(f'{get_local_time()} Starting pool working on {collection.database.name}.{collection.name}...')
                for doc in tqdm(collection.find({}, projection={"_id_numeric": 1, "content": 1, "numeric_columns": 1}), total=collection.count_documents({})):
                    _id_numeric, content, numeric_columns = doc['_id_numeric'], doc['content'], doc['numeric_columns']
                    
                    if check_table_is_in_thresholds(content, self.tables_thresholds):
                        column_embeddings = self.model.embedding_table(content, numeric_columns)
            
                        if column_embeddings.shape[0] > 0:
                            self.cidx.add_with_ids(column_embeddings, np.array([_id_numeric] * len(column_embeddings), dtype=np.int64))
            
        faiss.write_index(self.cidx, self.cidx_file)
        return round(time() - start, 5), os.path.getsize(self.cidx_file) / (1024 ** 3)


    def query(self, results_file, k, query_ids, **kwargs):
        start = time()
        if not self.cidx and os.path.exists(self.cidx_file):
            self.cidx = faiss.read_index(self.cidx_file)
            
        results = []
        for query_id in tqdm(query_ids):
            doc = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *self.collections)
            content, numeric_columns = doc['content'], doc['numeric_columns']
            cembeddings = self.model.embedding_table(content, numeric_columns)
            if cembeddings.shape[0] > 0:
                start_s = time()
                _, ids = self.cidx.search(cembeddings, k)
                end_s = time()
                ccnt = np.unique(ids, return_counts=True)

                ctopk = sorted(zip(ccnt[0], ccnt[1]), key=lambda x: x[1], reverse=True)
                results.append([query_id, round(end_s - start_s, 3), [x[0] for x in ctopk[:k] if x[0] != query_id], []])
            else:
                results.append([query_id, 0, [], []])
        pd.DataFrame(results, columns=['query_id', 'duration', 'results', 'results_overlap']).to_csv(results_file, index=False)    
        return round(time() - start, 5)
    
    def clean(self):
        if os.path.exists(self.cidx_file):
            os.remove(self.cidx_file)
        
