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


from tools.utils.misc import is_valid_table
from tools.utils.classes import AlgorithmTester
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.table_embedder import FastTextTableEmbedder, TaBERTTableEmbedder, CompressFastTextTableEmbedder




def worker_embedding_data_preparation(data) -> tuple[np.ndarray, np.ndarray]:
    global datalake_location, dataset, size, mapping_id_file, numeric_columns_file, blacklist, num_cpu, model_path, mode

    dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)
    match mode:
        case 'ft'|'ftdist':
            model = FastTextTableEmbedder(model_path)
        case 'cft'|"cftdist":
            model = CompressFastTextTableEmbedder(model_path)
    
    chunk = data[0]
    if os.getpid() % num_cpu == 0:
        print(f'Debug process {os.getpid()} works on {chunk} total batch size {chunk.stop - chunk.start}')
        start = time()
    d = model.get_dimension()
    
    xb = np.empty(shape=(0, d))
    xb_ids = np.empty(shape=(0, 1))
    xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
    batch_size = 1000

    for qid in tqdm(chunk, total=chunk.stop - chunk.start, leave=False, disable=False if os.getpid() % num_cpu == 0 else True):
        table_obj = dlh.get_table_by_numeric_id(qid)
        _id_numeric, content, numeric_columns = table_obj['_id_numeric'], table_obj['content'], table_obj['numeric_columns']
        if not is_valid_table(content, numeric_columns):
            continue
        if xb_batch.shape[0] > batch_size:
            xb = np.concatenate((xb, xb_batch))
            xb_ids = np.concatenate((xb_ids, xb_ids_batch))
            xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
        
        if os.getpid() % num_cpu == 0:
            print(f"\tdebug process working on table with shape {(len(content), len(content[0]))}...")
        
        colemb = model.embedding_table(content, numeric_columns, blacklist)
        if colemb.shape[0] == 0:
            continue

        ids = np.expand_dims(np.repeat([_id_numeric], colemb.shape[0]), axis=0)
        xb_ids_batch = np.concatenate((xb_ids_batch, ids.T))
        xb_batch = np.concatenate((xb_batch, colemb), axis=0)

    xb = np.concatenate((xb, xb_batch))
    xb_ids = np.concatenate((xb_ids, xb_ids_batch))

    dlh.close()
    if os.getpid() % num_cpu == 0:
        print(f'Debug process {os.getpid()} completed current batch of size {chunk.stop - chunk.start} in {round(time() - start, 3)}s')
    return xb, xb_ids




def chunks(sequence, chunk_size, *args):
    # Chunks of chunk_size documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield (sequence[j:j + chunk_size], *args)


def init_pool(_datalake_location, _dataset, _size, _mapping_id_file, _numeric_columns_file, _blacklist, _num_cpu, _model_path, _mode):
    global datalake_location, dataset, size, mapping_id_file, numeric_columns_file, blacklist, num_cpu, model_path, mode
    
    datalake_location, dataset, size, mapping_id_file, numeric_columns_file, blacklist, num_cpu, model_path, mode = \
        _datalake_location, _dataset, _size, _mapping_id_file, _numeric_columns_file, _blacklist, _num_cpu, _model_path, _mode


class EmbeddingTester(AlgorithmTester):
    def __init__(self, mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper, *args) -> None:
        super().__init__(mode, dataset, size, tables_thresholds, num_cpu, blacklist, datalake_helper)
        self.model_path, self.cidx_file, self.fasttext_model_dim = args
        self.cidx = None
        
    def data_preparation(self):
        start = time()
        # index "Flat,IDMap2" no good for this case, since we have to deal with more
        # than 10M vectors the search time is not acceptable
        logging.getLogger('TestLog').info(f'Embedding model: {self.model_path}')
        d = self.fasttext_model_dim
        
        xb = np.empty(shape=(0, d))
        xb_ids = np.empty(shape=(0, 1))

        # the FastText original model is a huge model of 7.4GB, loading it for each core 
        # is expensive, so reduce the number of actual CPUs...
        total_docs = self.datalake_helper.get_number_of_tables()
        chunksize = max(total_docs // self.num_cpu, 1)

        initargs = (
            self.datalake_helper.datalake_location, self.dataset, self.size, 
            self.datalake_helper._mapping_id_path, self.datalake_helper._numeric_columns_path, 
            self.blacklist, self.num_cpu, self.model_path, self.mode)

        with mp.Pool(self.num_cpu, initializer=init_pool, initargs=initargs) as pool:
            logging.getLogger('TestLog').info(f'Start embedding tables, chunk-size={chunksize}...')
            results = pool.map(worker_embedding_data_preparation, 
                               chunks(range(total_docs), chunksize))            
            logging.getLogger('TestLog').info('Tables embedded.')

            N = sum(xb.shape[0] for xb, _ in results)
            K = 8 * int(math.sqrt(N))
            M = 32
            training_size = min(M * K, N)
            HNSW_links_per_vertex = 32
            logging.getLogger('TestLog').info(f'FAISS index parameters: N={N}, K={K}, M={M}, HNSW={HNSW_links_per_vertex}, training_size=M*K={training_size}')
            
            logging.getLogger('TestLog').info('Preparing index training set...')
            for result in tqdm(results, leave=False):
                xb_batch, xb_ids_batch = result
                xb = np.concatenate((xb, xb_batch))
                xb_ids = np.concatenate((xb_ids, xb_ids_batch))
                if xb.shape[0] >= training_size:
                    break
                    

        index = faiss.index_factory(d, f"IVF{K}_HNSW{HNSW_links_per_vertex},Flat")
        logging.getLogger('TestLog').info('Training column index...')
        start_training = time()
        index.train(xb)
        self.cidx = faiss.IndexIDMap2(index)
        end_training = time()
        logging.getLogger('TestLog').info(f'Training column index time: {round(end_training - start_training, 3)}s')

        logging.getLogger('TestLog').info('Adding vectors with IDs...')
        for xb, xb_ids in results:
            self.cidx.add_with_ids(xb, xb_ids[:, 0])

        logging.getLogger('TestLog').info(f'Writing column index to {self.cidx_file}...')
        faiss.write_index(self.cidx, self.cidx_file)
        return round(time() - start, 5), os.path.getsize(self.cidx_file) / (1024 ** 3)


    def query(self, results_file, k, query_ids, **kwargs):
        match self.mode:
            case 'ft'|'cft':
                return self.query_naive(results_file, k, query_ids, **kwargs)
            case 'ftdist'|'cftdist':
                return self.query_dist(results_file, k, query_ids, **kwargs)

    def query_naive(self, results_file, k, query_ids, **kwargs):
        start = time()
        logging.getLogger('TestLog').info('Loading model...')
        match self.mode:
            case 'ft'|'ftdist':
                self.model = FastTextTableEmbedder(self.model_path)
            case 'cft'|'cftdist':
                self.model = CompressFastTextTableEmbedder(self.model_path)
            case 'tabert':
                self.model = TaBERTTableEmbedder(self.model_path)
            case _:
                raise ValueError('Unknown mode for table embedder with naive query mode: ' + self.mode)
        if not self.cidx:
            logging.getLogger('TestLog').info(f'Reading column index from {self.cidx_file}...')
            self.cidx = faiss.read_index(self.cidx_file)
        
        
        d = self.model.get_dimension()
        xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
        batch_size = 1000
        results = []
        logging.getLogger('TestLog').info('Start query time...')
        
        # batch query processing, because it's significantly faster
        for i, qid in tqdm(enumerate(query_ids), total=len(query_ids)):
            # when reached the batch threshold, execute the search for the 
            # current batch vectors
            if xq.shape[0] > batch_size:
                start_s = time()
                _, I = self.cidx.search(xq, int(k))
                batch_mean_time = (time() - start_s) / xq.shape[0]
                
                res = np.concatenate((xq_ids, I), axis=1)
                res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])

                for j, qid in enumerate(np.unique(xq_ids)):
                    x = sorted(list(zip(*np.unique(res[j], return_counts=True))), key=lambda z: z[1], reverse=True)
                    results.append((qid, round(batch_mean_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1], []))
                xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)

            doc = self.datalake_helper.get_table_by_numeric_id(int(qid))
            if doc == None:
                print('Error here', qid)
            try:
                colemb = self.model.embedding_table(doc['content'], doc['numeric_columns'], self.blacklist)
                ids = np.expand_dims(np.repeat([qid], colemb.shape[0]), axis=0)
                xq_ids = np.concatenate((xq_ids, ids.T))
                xq = np.concatenate((xq, colemb), axis=0)
            except ValueError:
                print(qid, colemb.shape)
        start_s = time()
        _, I = self.cidx.search(xq, int(k))
        res = np.concatenate((xq_ids, I), axis=1)
        res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])
        batch_mean_time = (time() - start_s) / xq.shape[0]

        for i, qid in enumerate(np.unique(xq_ids)):
            start_sort = time()
            x = sorted(list(zip(*np.unique(res[i], return_counts=True))), key=lambda z: z[1], reverse=True)
            sort_time = time() - start_time
            results.append((qid, round(batch_mean_time + sort_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1], []))

        pd.DataFrame(results, columns=['query_id', 'duration', 'results', 'results_overlap']).to_csv(results_file, index=False)    
        return round(time() - start, 5)
    
    def query_dist(self, results_file, k, query_ids, **kwargs):
        def compute_results(xq_ids, D, I, k):
            """From the search returned data, extract the 10 best values, through the average on the distance values"""
            ids = np.unique(xq_ids)
            rI = np.concatenate((xq_ids, I), axis=1)
            rI = np.split(rI[:, 1:], np.unique(rI[:, 0], return_index=True)[1][1:])

            rD = np.concatenate((xq_ids, D), axis=1)
            rD = np.split(rD[:, 1:], np.unique(rD[:, 0], return_index=True)[1][1:])

            return [
                    [
                        # the sorting is in ascending order and this is ok, 
                        # we work with distances, the lower the better in our case
                        int(y[0]) for y in
                        sorted([[g[0], mean(m[1] for m in g[1])] for g in groupby(raw_group, key=lambda n: n[0])], key=lambda x: x[1])[:k]
                    ]
                    for raw_group in 
                    [
                        #
                        sorted(
                            [
                                x for j in range(rI[i].shape[0]) for x in zip(rI[i][j], rD[i][j]) if x[0] not in {ids[i], -1}
                            ], 
                            key=lambda x: x[0]) 
                        for i in range(len(rI))
                    ]
                ]
        
        logging.getLogger('TestLog').info('Query mode: distance')
        logging.getLogger('TestLog').info('Loading model...')
        match self.mode:
            case 'ft'|'ftdist':
                self.model = FastTextTableEmbedder(self.model_path)
            case 'cft'|'cftdist':
                self.model = CompressFastTextTableEmbedder(self.model_path)
            case 'tabert':
                self.model = TaBERTTableEmbedder(self.model_path)
            case _:
                raise ValueError('Unknown mode for table embedder with naive query mode: ' + self.mode)
        if not self.cidx:
            logging.getLogger('TestLog').info(f'Reading column index from {self.cidx_file}...')
            self.cidx = faiss.read_index(self.cidx_file)
        
        d = self.model.get_dimension()
        xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
        batch_size = 1000
        results = []
        logging.getLogger('TestLog').info('Start query time...')
       
        start = time()
        # batch query processing, because it's significantly faster
        for qid in tqdm(query_ids, total=len(query_ids)):
            # when reached the batch threshold, execute the search for the 
            # current batch 
            doc = self.datalake_helper.get_table_by_numeric_id(int(qid))
            colemb = self.model.embedding_table(doc['content'], doc['numeric_columns'], self.blacklist)
            if colemb.shape[0] == 0:
                continue
            try:
                ids = np.expand_dims(np.repeat([qid], colemb.shape[0]), axis=0)
                xq_ids = np.concatenate((xq_ids, ids.T))
                xq = np.concatenate((xq, colemb), axis=0)
            except ValueError:
                # print(qid, colemb.shape)
                continue    
            if xq.shape[0] > batch_size:
                start_s = time()
                # here it searches K vectors for each query vector, i.e. for each column vector,
                # in the end there will be at least K results vector in almost every case
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
        end_total_query = time()

        for i, qid in enumerate(np.unique(xq_ids)):
            results.append((qid, round(batch_mean_time, 3), batch_results[i], []))

        pd.DataFrame(results, columns=['query_id', 'duration', 'results', 'results_overlap']).to_csv(results_file, index=False)    
        return round(end_total_query - start, 5)


    def clean(self):
        if os.path.exists(self.cidx_file):
            os.remove(self.cidx_file)
        
