from dltftools.utils.metrics import *


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
    query_id, data, k_values, query_silver_standard = inp

    results = []
    true_relevances = [x[1] for x in query_silver_standard]    
    
    for (algorithm, mode), data in data.groupby(['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()    
        for k in k_values:
            if k > len(true_relevances):
                continue            
            rec = recall_at_k(true_relevances, result_relevances, k)
            prec = precision_at_k(true_relevances, result_relevances, k)
            rel_prec = relevance_precision_at_k(true_relevances, result_relevances, k)
            f1 = f_score(prec, rec)
            results.append([query_id, len(query_silver_standard), algorithm, mode, k, prec, rel_prec, rec, f1])
    return results


def worker_ndcg(inp):
    query_id, query_res, k_values, query_silver_standard = inp
    all_ndcg_results = []
    true_relevances = [x[1] for x in query_silver_standard]

    for (algorithm, mode), data in query_res.groupby(by=['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()
        for k in k_values:
            if len(true_relevances) == 0:
                all_ndcg_results.append([query_id, len(true_relevances), algorithm, mode, k, 0])
                continue
            # there could be errors in those cases where there isn't any actual relevant documents
            # i.e. all the retrivied documents doesn't have a non-zero SLOTH overlap                
            ndcg = ndcg_at_k(true_relevances, result_relevances, k)

            all_ndcg_results.append([query_id, len(true_relevances), algorithm, mode, k, ndcg])
    return all_ndcg_results
