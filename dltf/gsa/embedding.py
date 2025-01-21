import os
import math
from time import time
import multiprocessing as mp
from typing import Dict, List

import faiss
import numpy as np
import polars as pl
from tqdm import tqdm

from dltf.utils.misc import chunks
from dltf.utils.loghandler import info
from dltf.gsa.base_tester import AbstractGlobalSearchAlgorithm
from dltf.utils.datalake import DataLakeHandlerFactory
from dltf.utils.table_embedder import table_embedder_factory
from dltf.utils.tables import is_valid_table


def worker_embedding_data_preparation(data) -> tuple[np.ndarray, np.ndarray]:
    global mode, num_cpu, string_blacklist, string_translators, string_patterns, emb_model_path, dlhargs
    
    dlh = DataLakeHandlerFactory.create_handler(*dlhargs)
    model = table_embedder_factory(mode, emb_model_path)

    id_tables_range = data[0]
    if os.getpid() % num_cpu == 0:
        print(f'Debug process {os.getpid()} works on {id_tables_range} total batch size {id_tables_range.stop - id_tables_range.start}')
        start = time()

    d = model.get_dimension()
    
    xb = np.empty(shape=(0, d))
    xb_ids = np.empty(shape=(0, 1))
    xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
    batch_size = 1000

    for qid in tqdm(id_tables_range, total=id_tables_range.stop - id_tables_range.start, leave=False, disable=False if os.getpid() % num_cpu == 0 else True):
        table_obj = dlh.get_table_by_numeric_id(qid)
        _id_numeric, table, valid_columns = table_obj['_id_numeric'], table_obj['content'], table_obj['valid_columns']
        if not is_valid_table(table, valid_columns):
            continue

        if xb_batch.shape[0] > batch_size:
            xb = np.concatenate((xb, xb_batch))
            xb_ids = np.concatenate((xb_ids, xb_ids_batch))
            xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
        
        colemb = model.embed_columns(table, valid_columns, string_blacklist, string_translators, string_patterns)
        if colemb.shape[0] == 0:
            continue

        ids = np.expand_dims(np.repeat([_id_numeric], colemb.shape[0]), axis=0)
        xb_ids_batch = np.concatenate((xb_ids_batch, ids.T))
        xb_batch = np.concatenate((xb_batch, colemb), axis=0)

    xb = np.concatenate((xb, xb_batch))
    xb_ids = np.concatenate((xb_ids, xb_ids_batch))

    dlh.close()
    if os.getpid() % num_cpu == 0:
        print(f'Debug process {os.getpid()} completed current batch of size {id_tables_range.stop - id_tables_range.start} in {round(time() - start, 3)}s')
    return xb, xb_ids


def initializer( _mode, _num_cpu, _string_blacklist, _string_translators, _string_patterns, _emb_model_path, *_dlhargs):
    global mode, num_cpu, string_blacklist, string_translators, string_patterns, emb_model_path, dlhargs
    dlhargs             = _dlhargs[0]
    mode                = _mode
    num_cpu             = _num_cpu
    string_blacklist    = _string_blacklist
    string_translators  = _string_translators
    string_patterns     = _string_patterns
    emb_model_path      = _emb_model_path


class EmbeddingGS(AbstractGlobalSearchAlgorithm):
    def __init__(self, mode, dlh, string_blacklist, string_translators, string_patterns, 
                 num_cpu, model_path, column_index_file, embedding_dimension) -> None:
        super().__init__(mode, dlh, string_blacklist, string_translators, string_patterns)
        self.num_cpu = num_cpu
        self.model_path = model_path
        self.cidx_file = column_index_file
        self.embedding_model_dim = embedding_dimension

        self.cidx = None
        
    def data_preparation(self):
        start = time()
        # index "Flat,IDMap2" no good for this case, since we have to deal with more
        # than 1M vectors the search time is not acceptable
        info(f'Embedding model: {self.model_path}')
        d = self.embedding_model_dim
        
        xb = np.empty(shape=(0, d))
        xb_ids = np.empty(shape=(0, 1))

        work = range(self.dlh.count_tables())
        chunk_size = max(len(work) // self.num_cpu, 1)

        initargs = (self.mode, self.num_cpu, self.string_blacklist, self.string_translators, self.string_patterns, self.model_path, self.dlh.config())

        with mp.get_context('spawn').Pool(self.num_cpu, initializer=initializer, initargs=initargs) as pool:
            info(f'Start embedding tables, chunk-size={chunk_size}...')
            results = pool.map(worker_embedding_data_preparation, chunks(work, chunk_size))            
            info('Tables embedded.')

            N = sum(xb.shape[0] for xb, _ in results)
            K = 8 * int(math.sqrt(N))
            HNSW_links_per_vertex = 32
            training_size = min(HNSW_links_per_vertex * K, N)
            info(f'FAISS index parameters: N={N}, K={K}, M={HNSW_links_per_vertex}, training_size=M*K={training_size}')
            
            info('Preparing index training set...')
            for result in tqdm(results, leave=False):
                xb_batch, xb_ids_batch = result
                xb = np.concatenate((xb, xb_batch))
                xb_ids = np.concatenate((xb_ids, xb_ids_batch))
                if xb.shape[0] >= training_size:
                    break
        
        info(f"FAISS index: OPQ32,IVF{K}_HNSW{HNSW_links_per_vertex},PQ32")
        index = faiss.index_factory(d, f"OPQ32,IVF{K}_HNSW{HNSW_links_per_vertex},PQ32")
        info('Training column index...')
        start_training = time()
        index.train(xb)
        self.cidx = faiss.IndexIDMap2(index)
        end_training = time()
        info(f'Training column index time: {round(end_training - start_training, 3)}s')

        info('Adding vectors with IDs...')
        for xb, xb_ids in results:
            self.cidx.add_with_ids(xb, xb_ids[:, 0])

        info(f'Writing column index to {self.cidx_file}...')
        faiss.write_index(self.cidx, self.cidx_file)
        return round(time() - start, 5), os.path.getsize(self.cidx_file) / (1024 ** 3)

    def query(self, queries:List[int]|Dict[int,List], k:int, results_file:str):
        start = time()
        info('Loading model...')
        model = table_embedder_factory(self.mode, self.model_path)
        if not self.cidx:
            info(f'Reading column index from {self.cidx_file}...')
            self.cidx = faiss.read_index(self.cidx_file)        
        
        d = model.get_dimension()
        xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)
        batch_size = 1000
        results = []
        
        info('Running top-K...')
        # batch query processing, because it's significantly faster
        for i, qid in tqdm(enumerate(queries), total=len(queries)):
            # when reached the batch threshold, 
            # execute the search for the current batch vectors
            if xq.shape[0] > batch_size:
                start_topk_batch = time()
                _, I = self.cidx.search(xq, int(k))
                batch_mean_time = (time() - start_topk_batch) / xq.shape[0]
                
                res = np.concatenate((xq_ids, I), axis=1)
                res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])

                for j, qid in enumerate(np.unique(xq_ids)):
                    x = sorted(list(zip(*np.unique(res[j], return_counts=True))), key=lambda z: z[1], reverse=True)
                    results.append((qid, round(batch_mean_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1]))
                xq, xq_ids = np.empty(shape=(0, d), dtype=np.float64), np.empty(shape=(0, 1), dtype=np.int32)

            if isinstance(queries, list):
                doc = self.dlh.get_table_by_numeric_id(int(qid))
                table, valid_columns = doc['content'], doc['valid_columns']
            elif isinstance(queries, dict):
                table, valid_columns = queries[qid], [1] * len(queries[qid][0])

            colemb = model.embed_columns(table, valid_columns, self.string_blacklist, self.string_translators, self.string_patterns)
            ids = np.expand_dims(np.repeat([qid], colemb.shape[0]), axis=0)
            xq_ids = np.concatenate((xq_ids, ids.T))
            xq = np.concatenate((xq, colemb), axis=0)

        start_topk_batch = time()
        _, I = self.cidx.search(xq, int(k))
        res = np.concatenate((xq_ids, I), axis=1)
        res = np.split(res[:, 1:], np.unique(res[:, 0], return_index=True)[1][1:])
        batch_mean_time = (time() - start_topk_batch) / xq.shape[0]

        for i, qid in enumerate(np.unique(xq_ids)):
            start_sort = time()
            x = sorted(list(zip(*np.unique(res[i], return_counts=True))), key=lambda z: z[1], reverse=True)
            sort_time = time() - start_sort
            results.append((qid, round(batch_mean_time + sort_time, 3), [int(y[0]) for y in x if y[0] not in {qid, -1}][:k + 1]))

        pl.DataFrame([[qid, t, str(r)] for qid, t, r in results], schema=['query_id', 'duration', 'results'], orient='row').write_csv(results_file)    
        return round(time() - start, 5), [[qid, r] for qid, _, r in results]
    
    def clean(self):
        if os.path.exists(self.cidx_file):
            os.remove(self.cidx_file)
