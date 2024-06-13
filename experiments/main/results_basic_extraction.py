import os
import re
import argparse

import pandas as pd

import multiprocessing as mp

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import (
    apply_sloth,
    get_mongodb_collections, 
    get_one_document_from_mongodb_by_key, 
    _create_token_set
)



def _worker_result_extractor(inp):
    global algorithm, mode, small
        
    query_id, result_id, rank, algorithm_overlap = inp
    if not result_id:
        return (query_id, None, None, None, None, None, None, None)
    result_id = int(result_id)

    mongoclient, collections = get_mongodb_collections(small)
    doc_table1 = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
    doc_table2 = get_one_document_from_mongodb_by_key('_id_numeric', result_id, *collections)
    mongoclient.close()
    
    table1 = doc_table1['content']
    table2 = doc_table2['content']

    numeric_columns1 = doc_table1['numeric_columns']
    numeric_columns2 = doc_table2['numeric_columns']

    set1 = _create_token_set(table1, mode if mode != 'fasttext' else 'bag', numeric_columns1)
    if algorithm == 'josie':
        algorithm_overlap = int(algorithm_overlap)
    elif algorithm == 'lshforest':
        set2 = _create_token_set(table2, mode, numeric_columns2)
        algorithm_overlap = len(set(set1).intersection(set(set2)))
    elif algorithm == 'embedding':
        set2 = _create_token_set(table2, 'bag', numeric_columns2)
        algorithm_overlap = len(set(set1).intersection(set(set2)))
    
        
    query_size = len(set1)
    sloth_overlap = apply_sloth(table1, table2, numeric_columns1, numeric_columns2)
    difference_overlap = algorithm_overlap - sloth_overlap

    max_table_overlap_size = len(table1) * (len(table1[0]) - sum(numeric_columns1))
    try:    
        difference_overlap_norm = difference_overlap / max_table_overlap_size
    except ZeroDivisionError:
        return query_id, result_id, algorithm, mode, -1, -1, rank, algorithm_overlap, sloth_overlap, difference_overlap, difference_overlap_norm
    return query_id, result_id, algorithm, mode, query_size, max_table_overlap_size, rank, algorithm_overlap, sloth_overlap, difference_overlap, difference_overlap_norm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
    parser.add_argument('-a', '--algorithm', 
                        required=False, default='josie',
                        choices=['josie', 'lshforest', 'embedding'])
    parser.add_argument('-m', '--mode', 
                        required=False, default='set',
                        choices=['set', 'bag', 'fasttext'])
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('--analyse-up-to', required=False, type=int)
    parser.add_argument('--small', required=False, action='store_true',
                        help='works on small collection versions (only for testing)')
    parser.add_argument('-w', '--num-workers', 
                        type=int, required=False, default=min(os.cpu_count(), 64),
                        help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
    

    args = parser.parse_args()
    test_name = args.test_name
    algorithm = args.algorithm
    mode =      args.mode
    k =         args.k
    ank =       args.analyse_up_to
    small =     args.small
    nworkers =  args.num_workers


    ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
    results_base_directory =    ROOT_TEST_DIR + '/results/base'
    results_extr_directory =    ROOT_TEST_DIR + '/results/extracted'

    results_file =              results_base_directory + f'/a{algorithm}_m{mode}_k{k}.csv'
    extracted_results_file =    results_extr_directory + f'/a{algorithm}_m{mode}_k{k}_extracted.csv' 

    results = pd.read_csv(results_file)[['query_id', 'results']]

    if algorithm == 'josie':
        jr = results.values.tolist()
        work = [
            (r[0], s, rank, o) 
            for r in jr 
                for rank, (s, o) in enumerate(zip(
                    re.findall(r'\d+', r[1])[::2] if type(r[1]) == str else [None], 
                    re.findall(r'\d+', r[1])[1::2] if type(r[1]) == str else [None],
                    ), start=1) if rank <= ank and r[1]
            ]
    elif algorithm == 'lshforest' or algorithm == 'embedding':
        results['results'] = results['results'].apply(eval)
        work = [
            (res[0], sid, rank, None)
            for res in results.values.tolist() 
            for rank, sid in enumerate(res[1], start=1)
        ]

    print('Start processing results...')
    with mp.Pool(processes=nworkers) as pool:
        res = pool.map(_worker_result_extractor, work)

    # "query_res != None" because maybe there isn't enough work 
    # and a processor may return None instead of a list of tuples...
    res = [query_res for query_res in res if query_res != None]

    columns=[
        'query_id', 
        'result_id',
        'algorithm',
        'mode',
        'query_size',
        'max_table_overlap_size',
        'rank', 
        'algorithm_overlap', 
        'sloth_overlap', 
        'difference_overlap',
        'difference_overlap_norm'
        ]
    
    pd.DataFrame(res, columns=columns) \
        .convert_dtypes() \
        .to_csv(extracted_results_file, index=False)

    print(f'Extracted results saved to {extracted_results_file}.')
