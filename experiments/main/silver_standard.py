from collections import defaultdict
import os
from pprint import pprint

import pandas as pd
from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_query_ids_from_query_file


if __name__ == '__main__':
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
            
            results_ids = pd.read_csv(fname).convert_dtypes().groupby(by='query_id')['result_id']

            for query_id, ids in results_ids:
                for i in ids:
                    silver_standard[query_id].add(i)


    for k, v in silver_standard.items():
        print(k, 'nresults: ', len(v) )
            
    

