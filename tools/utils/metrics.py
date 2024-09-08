from math import log2, pow
from collections import Counter as multiset



def relevance_precision(true_relevances:list, result_relevances:list, p):
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances)
    return sum(x[1] for x in (true_relevances & result_relevances).items()) / p
    

def precision(true_relevances:list, result_relevances:list, p):
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances)
    return sum(x[1] for x in (true_relevances & result_relevances).items()) / p
            

def recall(true_relevances:list, result_relevances:list, p):
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances)
    return sum(x[1] for x in (true_relevances & result_relevances).items()) / len(true_relevances)


def f_score(p, r, beta=1):
    return (1 + pow(beta, 2)) * (p * r) / (pow(beta, 2) * p  + r)


def ndcg_at_p(true_relevances, result_relevances, p):
    p = min(p, len(true_relevances), len(result_relevances))
    if p <= 0: # because computing nDCG is meaningful only if there is more than one document 
        return 0, 1
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:p], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:p], start=1))
    if idcg < dcg:
        raise ValueError()
    return dcg / idcg, p




