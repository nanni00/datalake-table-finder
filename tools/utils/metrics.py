from math import log2, pow
from collections import Counter as multiset



def proximity(x:int|float, y:int|float):
    return 1 - abs(x - y) / max(x, y)


def relevance_precision_at_k(true_relevances:list, result_relevances:list, k):
    return precision_at_k(true_relevances[:k], result_relevances[:k], min(k, len(true_relevances)))    


def precision_at_k(true_relevances:list, result_relevances:list, k):
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances[:k])
    return sum(x[1] for x in (true_relevances & result_relevances).items()) / k
            

def recall_at_k(true_relevances:list, result_relevances:list, k):
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances[:k])
    return sum(x[1] for x in (true_relevances & result_relevances).items()) / len(true_relevances)


def f_score(p, r, beta=1):
    return (1 + pow(beta, 2)) * (p * r) / (pow(beta, 2) * p  + r) if not (p, r) == (0, 0) else 0


def ndcg_at_k(true_relevances, result_relevances, k):
    k = min(k, len(true_relevances), len(result_relevances))
    if k <= 0: # because computing nDCG is meaningful only if there is more than one document 
        return 0, 1
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:k], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:k], start=1))
    if idcg < dcg:
        raise ValueError()
    return dcg / idcg, k




