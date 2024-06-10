import statistics
from collections import defaultdict
import os

import pandas as pd
from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_mongodb_collections, get_query_ids_from_query_file


if __name__ == '__main__':
    small = True
    k = 10

    test_name = 'full'

    test_dir = defpath.data_path.josie_tests + '/' + test_name
    results_dir = test_dir + '/results'
    query_file = test_dir + '/query.json'

    algorithms = ['josie', 'lshforest']
    modes = ['set', 'bag']

    sampled_ids = get_query_ids_from_query_file(query_file)
    
    silver_standard = defaultdict(set)

    for algorithm in algorithms:
        for mode in modes:
            fname = f"{results_dir}/a{algorithm}_m{mode}_k{k}_extracted.csv"

            if not os.path.exists(fname):
                continue
            
            results_ids = pd.read_csv(fname).convert_dtypes().groupby(by='query_id')[['result_id', 'sloth_overlap']]

            for query_id, ids_overlaps in results_ids:
                for i in ids_overlaps.values:
                    _id, _overlap = i
                    silver_standard[query_id].add((_id, _overlap))


    mongoclient, collections = get_mongodb_collections(small)
    
    for query_id, v in silver_standard.items():
        print(query_id, 'nresults: ', len(v) )
        silver_standard[query_id] = sorted(list(v), key=lambda x: x[1], reverse=True)
        # print(silver_standard[k])
    print()

    ############### ANALYSIS PRECISION-AT-K ##################
    # For each query (aka silver standard), the algorithm ALG returned at least the K best
    # results among all?

    k_precision = 7

    solvers = dict()

    for algorithm in algorithms:
        for mode in modes:
            fname = f"{results_dir}/a{algorithm}_m{mode}_k{k}_extracted.csv"

            if not os.path.exists(fname):
                continue
            solvers[(algorithm, mode)] = pd.read_csv(fname).convert_dtypes()
    
    precision_at_k_results = []

    for query_id in silver_standard.keys():
        qss = [x[1] for x in silver_standard[query_id]]
        avg_overlap = round(statistics.mean(qss), 3)
        stdev_overlap = round(statistics.stdev(qss))

        for solver, result in solvers.items():
            ids = result[result['query_id'] == query_id]['result_id'].values.tolist()
            real_topk = [x[0] for x in silver_standard[query_id][:k_precision]]
            precision_at_k = set(real_topk).intersection(ids)
            
            precision_at_k_results.append([query_id, len(qss), avg_overlap, stdev_overlap, solver[0], solver[1], k_precision, len(precision_at_k)])

    columns = [
        'query_id',
        'silver_std_size',
        'silver_std_ov_mean',
        'silver_std_ov_stdev',
        'algorithm',
        'mode',
        'k',
        'precision_at_k'
    ]

    precision_at_k_results = pd.DataFrame(precision_at_k_results, columns=columns)
    print(precision_at_k_results)


    
    

