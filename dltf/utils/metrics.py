from math import log2, pow
from collections import Counter as multiset



def jaccard(X:set, Y:set):
    # set.union seems to be quite unefficient, 
    # more than computing in this way the union
    intersection_size = 0
    union_size = 0
    for x in X:
        if x in Y:
            intersection_size += 1
        union_size += 1
    if not intersection_size or not union_size: 
        return 0
    for y in Y:
        if y not in X: union_size += 1
    return intersection_size / union_size if union_size else 0


def containment(X:set, Y:set):
    return len(X.intersection(Y)) / min(len(X), len(Y)) if min(len(X), len(Y)) > 0 else 0


metrics = {
    'jaccard':      jaccard, # lambda X, Y: len(X.intersection(Y)) / len(X.union(Y)) if len(X.union(Y)) > 0 else 0,
    'containment':  containment  
}


def proximity(x:int|float, y:int|float):
    return 1 - abs(x - y) / max(x, y)


def relevance_precision_at_k(true_relevances:list, result_relevances:list, k:int, negative_tag=0):
    """
    Returns the RP@K for the given result and true relevances. 
    The negative tag is the value which indicates non-relevant items
    """
    return precision_at_k(true_relevances[:k], result_relevances[:k], min(k, len(true_relevances)), negative_tag)    


def precision_at_k(true_relevances:list, result_relevances:list, k:int, negative_tag=0) -> float:
    """
    Returns the precision@K for the given result and true relevances. 
    The negative tag is the value which indicates non-relevant items.
    NB: the true and result relevances lists should contain not the relevance values themselves,
    but true/false values indicating whether the i-th item is relevant or not.
    """
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances[:k])
    if negative_tag in true_relevances:     del true_relevances[negative_tag]
    if negative_tag in result_relevances:   del result_relevances[negative_tag]
    return sum((true_relevances & result_relevances).values()) / k
            

def recall_at_k(true_relevances:list, result_relevances:list, k:int, negative_tag=0):
    """
    Returns the recall@K for the given result and true relevances. 
    The negative tag is the value which indicates non-relevant items
    NB: the true and result relevances lists should contain not the relevance values themselves,
    but true/false values indicating whether the i-th item is relevant or not.
    """
    if len(result_relevances) == 0: 
        return 0
    true_relevances = multiset(true_relevances)
    result_relevances = multiset(result_relevances[:k])
    if negative_tag in true_relevances:     del true_relevances[negative_tag]
    if negative_tag in result_relevances:   del result_relevances[negative_tag]
    
    return sum((true_relevances & result_relevances).values()) / sum(true_relevances.values())


def f_score(p, r, beta=1):
    return (1 + pow(beta, 2)) * (p * r) / (pow(beta, 2) * p  + r) if not (p, r) == (0, 0) else 0


def ndcg_at_k(true_relevances, result_relevances, k):
    # padding with 0 values
    if len(true_relevances) < k:
        true_relevances += [0] * (k - len(true_relevances))
    if len(result_relevances) < k:
        result_relevances += [0] * (k - len(result_relevances))
    
    if not any(true_relevances) or not any(result_relevances):
        return 0
    
    # computing nDCG is meaningful only if there is more than one document 
    if k <= 0: 
        return 0
    
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:k], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:k], start=1))
    
    if idcg < dcg:
        raise ValueError(f'Ideal DCG is lower than current DCG: {idcg} < {dcg}')
    if idcg < 0:
        raise ValueError(f'Ideal DCG is lower than 0: {idcg}')
    if dcg < 0:
        raise ValueError(f'Computed DCG is lower than 0: {dcg}, with values {result_relevances} and silver standard {true_relevances}')
    
    return dcg / idcg




