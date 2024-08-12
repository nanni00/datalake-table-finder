from itertools import groupby
import logging
import os
import math
import multiprocessing as mp
from statistics import mean

import numpy as np
from time import time
from tqdm import tqdm

import faiss
import pandas as pd


from tools.utils.classes import AlgorithmTester
from tools.utils.parallel_worker import worker_embedding_data_preparation
from tools.utils.table_embedder import FastTextTableEmbedder, TaBERTTableEmbedder
from tools.utils.utils import check_table_is_in_thresholds
from tools.utils.mongodb_utils import get_one_document_from_mongodb_by_key
    

def chunks(sequence, chunk_size, *args):
    # Chunks of chunk_size documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield (sequence[j:j + chunk_size], *args)


class EmbeddingTester(AlgorithmTester):
    def __init__(self, mode, dataset, size, tables_thresholds, num_cpu, blacklist, *args) -> None:
        super().__init__(mode, dataset, size, tables_thresholds, num_cpu, blacklist)
        self.model_path, self.cidx_file, self.collections = args
        self.cidx = None
        
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
        # index "Flat,IDMap2" no good for this case, since we have to deal with more
        # than 10M vectors the search time is not acceptable
        
        d = 300 # self.model.get_dimension()
        
        xb = np.empty(shape=(0, d))
        xb_ids = np.empty(shape=(0, 1))

        # the FastText original model is a huge model of 7.4GB, loading it for each core 
        # is expensive, so reduce the number of actual CPUs...
        num_effective_cpu = 24
        total_docs = sum(coll.count_documents({}) for coll in self.collections)
        chunksize = max(total_docs // min(self.num_cpu, num_effective_cpu), 1)

        with mp.Pool(min(num_effective_cpu, self.num_cpu)) as pool:
            logging.info(f'Start embedding tables, chunk-size={chunksize}...')
            results = pool.map(worker_embedding_data_preparation, 
                               chunks(range(total_docs), chunksize, 
                                      self.mode, self.dataset, self.size, self.model_path, self.tables_thresholds, self.blacklist))
            logging.info('Tables embedded.')
            for result in results:
                xb_batch, xb_ids_batch = result
                xb = np.concatenate((xb, xb_batch))
                xb_ids = np.concatenate((xb_ids, xb_ids_batch))
                    
        N = xb.shape[0]
        K = min(4 * int(math.sqrt(N)), 65536)
        training_size = min(64 * K, N)
        HNSW_links_per_vertex = 32
        logging.info(f'Vector shapes: xb={xb.shape}, xb_ids={xb_ids.shape}')
        logging.info(f'FAISS index parameters: N={N}, K={K}, HNSW={HNSW_links_per_vertex}, training_size={training_size}')

        index = faiss.index_factory(300, f"IVF{K}_HNSW{HNSW_links_per_vertex},Flat")
        logging.info('Training column index...')
        index.train(xb[:training_size])
        self.cidx = faiss.IndexIDMap2(index)
        
        logging.info('Adding vectors with IDs...')
        self.cidx.add_with_ids(xb, xb_ids[:, 0])

        logging.info(f'Writing column index to {self.cidx_file}...')
        faiss.write_index(self.cidx, self.cidx_file)
        return round(time() - start, 5), os.path.getsize(self.cidx_file) / (1024 ** 3)


    def query(self, results_file, k, query_ids, **kwargs):
        match kwargs['query_mode']:
            case 'naive':
                return self.query_naive(results_file, k, query_ids, **kwargs)
            case 'distance':
                return self.query_dist(results_file, k, query_ids, **kwargs)

    def query_naive(self, results_file, k, query_ids, **kwargs):
        start = time()
        logging.info('Loading model...')
        if self.mode == 'fasttext':
            self.model = FastTextTableEmbedder(self.model_path)
        elif self.mode == 'tabert':
            self.model = TaBERTTableEmbedder(self.model_path)

        if not self.cidx:
            logging.info(f'Reading column index from {self.cidx_file}...')
            self.cidx = faiss.read_index(self.cidx_file)
        
        
        d = self.model.get_dimension()
        xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
        batch_size = 1000
        results = []
        logging.info('Start query time...')
        
        # batch query processing, because it's significantly faster
        for i, qid in tqdm(enumerate(query_ids), total=len(query_ids)):
            # when reached the batch threshold, execute the search for the 
            # current batch vectors
            if i % batch_size == 0 and xq.shape[0] > 0:
                start_s = time()
                D, I = self.cidx.search(xq, int(k))
                batch_mean_time = (time() - start_s) / xq.shape[0]
                
                # 
                res = np.concatenate((xq_ids, I), axis=1)
                res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])

                for i, qid in enumerate(np.unique(xq_ids)):
                    x = sorted(list(zip(*np.unique(res[i], return_counts=True))), key=lambda z: z[1], reverse=True)
                    results.append((qid, round(batch_mean_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1], []))
                xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)

            doc = get_one_document_from_mongodb_by_key('_id_numeric', int(qid), *self.collections)
            colemb = self.model.embedding_table(doc['content'], doc['numeric_columns'], self.blacklist)
            ids = np.expand_dims(np.repeat([qid], colemb.shape[0]), axis=0)
            xq_ids = np.concatenate((xq_ids, ids.T))
            xq = np.concatenate((xq, colemb), axis=0)

        start_s = time()
        D, I = self.cidx.search(xq, int(k))
        batch_mean_time = (time() - start_s) / xq.shape[0]
        
        res = np.concatenate((xq_ids, I), axis=1)
        res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])

        for i, qid in enumerate(np.unique(xq_ids)):
            x = sorted(list(zip(*np.unique(res[i], return_counts=True))), key=lambda z: z[1], reverse=True)
            results.append((qid, round(batch_mean_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1], []))

        pd.DataFrame(results, columns=['query_id', 'duration', 'results', 'results_overlap']).to_csv(results_file, index=False)    
        return round(time() - start, 5)
    
    def query_dist(self, results_file, k, query_ids, **kwargs):
        def compute_results(xq_ids, D, I, k):
            ids = np.unique(xq_ids)
            rI = np.concatenate((xq_ids, I), axis=1)
            rI = np.split(rI[:, 1:], np.unique(rI[:, 0], return_index=True)[1][1:])

            rD = np.concatenate((xq_ids, D), axis=1)
            rD = np.split(rD[:, 1:], np.unique(rD[:, 0], return_index=True)[1][1:])

            return [
                    [
                        int(y[0]) for y in
                        sorted([[g[0], mean(x[1] for x in g[1])] for g in groupby(raw_group, key=lambda x: x[0])], key=lambda x: x[1])[:k]
                    ]
                    for raw_group in 
                    [
                        sorted(
                            [
                                x for j in range(rI[i].shape[0]) for x in zip(rI[i][j], rD[i][j]) if x[0] not in {ids[i], -1}
                            ], 
                            key=lambda x: x[0]) 
                        for i in range(len(rI))
                    ]
                ]
        
        start = time()
        logging.info('Query mode: distance')
        logging.info('Loading model...')
        if self.mode in ['fasttext', 'fasttextdist']:
            self.model = FastTextTableEmbedder(self.model_path)
        elif self.mode == 'tabert':
            self.model = TaBERTTableEmbedder(self.model_path)

        if not self.cidx:
            logging.info(f'Reading column index from {self.cidx_file}...')
            self.cidx = faiss.read_index(self.cidx_file)
        
        d = self.model.get_dimension()
        xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
        batch_size = 1000
        results = []
        logging.info('Start query time...')
       
        # batch query processing, because it's significantly faster
        for qid in tqdm(query_ids, total=len(query_ids)):
            # when reached the batch threshold, execute the search for the 
            # current batch 
            doc = get_one_document_from_mongodb_by_key('_id_numeric', int(qid), *self.collections)
            colemb = self.model.embedding_table(doc['content'], doc['numeric_columns'], self.blacklist)
            if colemb.shape[0] == 0:
                continue
            ids = np.expand_dims(np.repeat([qid], colemb.shape[0]), axis=0)
            xq_ids = np.concatenate((xq_ids, ids.T))
            xq = np.concatenate((xq, colemb), axis=0)
            
            if xq.shape[0] > batch_size:
                start_s = time()
                D, I = self.cidx.search(xq, int(k))
                batch_results = compute_results(xq_ids, D, I, k)
                batch_mean_time = (time() - start_s) / xq.shape[0]
                    
                for j, jqid in enumerate(np.unique(xq_ids)):
                    results.append((jqid, round(batch_mean_time, 3), batch_results[j], []))

                xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
            
        start_s = time()
        D, I = self.cidx.search(xq, int(k))
        batch_results = compute_results(xq_ids, D, I, k)
        batch_mean_time = (time() - start_s) / xq.shape[0]
     
        for i, qid in enumerate(np.unique(xq_ids)):
            results.append((qid, round(batch_mean_time, 3), batch_results[i], []))

        pd.DataFrame(results, columns=['query_id', 'duration', 'results', 'results_overlap']).to_csv(results_file, index=False)    
        return round(time() - start, 5)


    def clean(self):
        if os.path.exists(self.cidx_file):
            os.remove(self.cidx_file)
        
