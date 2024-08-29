from math import log2
from collections import Counter as multiset




def chunks(sequence, chunk_size, *args):
    # Chunks of chunk_size documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield (sequence[j:j + chunk_size], *args)


def worker_fp_per_query(inp):
    am, am_group = inp
    y = []
    for _, q_group in am_group.groupby(['query_id']):
        cnt = q_group[q_group['sloth_overlap'] == 0].shape[0]
        num_query_results = q_group.shape[0]
        
        y.append([am[0], am[1], cnt, cnt / num_query_results])
    return y


def worker_precision(inp):
    query_id, data, p_values, query_silver_standard = inp

    results = []
    
    true_relevances = [x[1] for x in query_silver_standard]
    
    tr = multiset(true_relevances)
    true_relevances_multisets = {_p: multiset(true_relevances[:_p]) for _p in p_values}
    
    for (algorithm, mode), data in data.groupby(['algorithm', 'mode']):
        predicted_relevances = data['sloth_overlap'].values.tolist()
        
        for _p in p_values:
            if _p > len(true_relevances):
                continue
            rr = multiset(predicted_relevances[:_p])
            num_rel_res_in_top_p = sum(x[1] for x in (tr & rr).items())
            precision = num_rel_res_in_top_p / _p
            
            num_rel_res_in_top_p_v2 = sum(x[1] for x in (true_relevances_multisets[_p] & rr).items())
            precision_v2 = num_rel_res_in_top_p_v2 / _p
            recall = num_rel_res_in_top_p / len(true_relevances)

            try:
                f1 = (2 * precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
            results.append([query_id, len(query_silver_standard), algorithm, mode, _p, precision, precision_v2, recall, f1])
    
    return results


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
                continue
            except ValueError:                
                continue
            ndcg_res.append([query_id, len(true_relevances), algorithm, mode, _p, _p - _actual_p, ndcg])
    return ndcg_res
