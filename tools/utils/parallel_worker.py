import logging
from math import log2
import os
from time import time
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm

from tools.utils.mongodb_utils import get_mongodb_collections, get_one_document_from_mongodb_by_key
from tools.utils.table_embedder import FastTextTableEmbedder
from tools.utils.utils import check_table_is_in_thresholds



def worker_embedding_data_preparation(inp) -> tuple[np.ndarray, np.ndarray]:
    # global dataset, size, mode, model_path, tables_thresholds, blacklist
    # print(dataset, size, mode, model_path)
    chunk, mode, dataset, size, model_path, tables_thresholds, blacklist = inp
    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=size)
    if mode == 'fasttext':
        model = FastTextTableEmbedder(model_path)
    print(f'Process {os.getpid()} (ppid={os.getppid()}) works on {chunk} total batch size {chunk.stop - chunk.start}')
    d = model.get_dimension()
    
    xb = np.empty(shape=(0, d))
    xb_ids = np.empty(shape=(0, 1))

    xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
    batch_size = 1000

    start = time()
    i = 0
    for i, qid in tqdm(enumerate(chunk), total=chunk.stop - chunk.start, disable=False if os.getpid() % 12 == 0 else True):
        doc = get_one_document_from_mongodb_by_key('_id_numeric', qid, *collections)
        _id_numeric, content, numeric_columns = doc['_id_numeric'], doc['content'], doc['numeric_columns']
        
        if not check_table_is_in_thresholds(content, tables_thresholds) or all(numeric_columns):
            continue

        i += 1
        if i % batch_size == 0 and xb_batch.shape[0] > 0:
            xb = np.concatenate((xb, xb_batch))
            xb_ids = np.concatenate((xb_ids, xb_ids_batch))
            xb_batch, xb_ids_batch = np.empty(shape=(0, d)), np.empty(shape=(0, 1))
        
        colemb = model.embedding_table(content, numeric_columns, blacklist)
        ids = np.expand_dims(np.repeat([_id_numeric], colemb.shape[0]), axis=0)
        xb_ids_batch = np.concatenate((xb_ids_batch, ids.T))
        xb_batch = np.concatenate((xb_batch, colemb), axis=0)

    xb = np.concatenate((xb, xb_batch))
    xb_ids = np.concatenate((xb_ids, xb_ids_batch))
    mongoclient.close()
    print(f'Process {os.getpid()} completed current batch of size {chunk.stop - chunk.start} in {round(time() - start, 3)}s')
    return xb, xb_ids








def worker_fp_per_query(inp):
    am, am_group = inp
    y = []
    for _, q_group in am_group.groupby(['query_id']):
        cnt = q_group[q_group['sloth_overlap'] == 0].shape[0]
        num_query_results = q_group.shape[0]
        
        y.append([am[0], am[1], cnt, cnt / num_query_results])
    return y


from collections import Counter as multiset

def worker_precision(inp):
    query_id, data, p_values, query_silver_standard = inp

    prec_results = []
    true_relevances = [x[1] for x in query_silver_standard]
    
    true_relevances_multisets = {_p: multiset(true_relevances[:_p]) for _p in p_values}
    
    for (algorithm, mode), data in data.groupby(['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()
        
        # inv_rel_docs = defaultdict(set)
        # for x, y in sorted_relevances:
        #     inv_rel_docs[y].add(x)
        # inv_rel_docs = sorted(list(inv_rel_docs.items()), reverse=True)

        # to compute the Precision@P, take the intersection between the top-P relevant documents
        # and the result IDs
        # N.B. the top-P relevant documents "may" be more than P: in some cases, the same N documents
        # have identical SLOTH overlap, so they share the same position in ranking
        # TODO maybe directly check the overlap would not be right? 
        rr = multiset(result_relevances)
        for _p in p_values:
            tr = true_relevances_multisets[_p]
            precision_at_p = sum(x[1] for x in (tr & rr).items())
            # precision_at_k = len(set([r for r in true_relevances[:_p]]).intersection(result_relevances))
            prec_results.append([query_id, len(query_silver_standard), algorithm, mode, _p, precision_at_p])
    return prec_results


def ndcg_at_p(true_relevances, result_relevances, p):
    p = min(p, len(true_relevances), len(result_relevances))
    if p <= 0: # because computing nDCG is meaningful only if there is more than one document 
        return 0, 1
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:p], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:p], start=1))
    if idcg < dcg:
        raise ValueError()

    return dcg / idcg, p

def worker_ndcg(inp):
    query_id, query_res, p_values, query_silver_standard = inp
    ndcg_res = []

    true_relevances = [x[1] for x in query_silver_standard]

    for (algorithm, mode), data in query_res.groupby(by=['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()
        for _p in p_values:
            if len(true_relevances) == 0:
                ndcg_res.append([query_id, len(true_relevances), algorithm, mode, _p, _p, 0])
                continue
            try:
                # there could be errors in those cases where there isn't any actual relevant documents
                # i.e. all the retrivied documents doesn't have a non-zero SLOTH overlap
                # TODO maybe is better to check and drop before those query tables which don't have
                # any good pair? 
                ndcg, _actual_p = ndcg_at_p(true_relevances, result_relevances, _p)
            except ZeroDivisionError:
                logging.warning(f'ZeroDivisionError - nDCG - query_id {query_id}, {algorithm}, {mode}')
                continue
            except ValueError:
                logging(f'ValueError - nDCG - query_id {query_id}, {algorithm}, {mode}')
                continue
            ndcg_res.append([query_id, len(true_relevances), algorithm, mode, _p, _p - _actual_p, ndcg])
    return ndcg_res
