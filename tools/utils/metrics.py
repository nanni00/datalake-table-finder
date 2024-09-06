from math import log2


def ndcg_at_p(true_relevances, result_relevances, p):
    p = min(p, len(true_relevances), len(result_relevances))
    if p <= 0: # because computing nDCG is meaningful only if there is more than one document 
        return 0, 1
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:p], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:p], start=1))
    if idcg < dcg:
        raise ValueError()
    return dcg / idcg, p