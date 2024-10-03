from math import log2, pow
from collections import Counter as multiset



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

    # k = min(k, len(true_relevances), len(result_relevances))
    # computing nDCG is meaningful only if there is more than one document 
    if k <= 0: 
        return 0, 1
    idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:k], start=1))
    dcg = sum(rel / log2(i + 1) for i, rel in enumerate(result_relevances[:k], start=1))
    
    if idcg < dcg:
        raise ValueError(f'Ideal DCG is lower than current DCG: {idcg} < {dcg}')
    
    return dcg / idcg, k




