import os
import re
import argparse

import pandas as pd
import pymongo

import multiprocessing as mp

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import apply_sloth, get_one_document_from_mongodb_by_key, _create_token_set




def _worker_compute_sloth(inp):
    global small
    mongoclient = pymongo.MongoClient()

    if not small:
        wikitables_coll = mongoclient.optitab.turl_training_set
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables
    else:
        wikitables_coll = mongoclient.optitab.turl_training_set_small
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables_small
        
    query_id, sid, josie_overlap = inp
    if not sid or not josie_overlap:
        return (query_id, None, None, None, None, None, None)
    
    sid, josie_overlap = int(sid), int(josie_overlap)
    
    doc_table1 = get_one_document_from_mongodb_by_key('_id_numeric', query_id, wikitables_coll, snapshot_coll)
    doc_table2 = get_one_document_from_mongodb_by_key('_id_numeric', sid, wikitables_coll, snapshot_coll)
    
    str_id1 = doc_table1['_id']
    str_id2 = doc_table2['_id']

    table1 = doc_table1['content']
    table2 = doc_table2['content']

    numeric_columns1 = doc_table1['numeric_columns']
    numeric_columns2 = doc_table2['numeric_columns']
    
    set1 = _create_token_set(table1, 'set', numeric_columns1)
    set2 = _create_token_set(table2, 'set', numeric_columns2)
    actual_set_overlap = len(set(set1).intersection(set(set2)))
    error = josie_overlap - actual_set_overlap

    largest_ov_sloth = apply_sloth(table1, table2, numeric_columns1, numeric_columns2)

    return (query_id, str_id1, sid, str_id2, josie_overlap, largest_ov_sloth, int(josie_overlap) - largest_ov_sloth, actual_set_overlap, error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
    parser.add_argument('-m', '--mode', 
                        required=False, default='set',
                        choices=['set', 'bag'])
    parser.add_argument('-k', required=True, type=int)
    parser.add_argument('--analyse-up-to', required=False, type=int)
    parser.add_argument('--small', required=False, action='store_true',
                        help='works on small collection versions (only for testing)')

    args = parser.parse_args()
    test_name = args.test_name
    mode =      args.mode
    k =         args.k
    ank =       args.analyse_up_to
    small =     args.small

    # TODO as in main tester, should be handled multiple copies such as results.csv, results(1).csv, results(2).csv...?
    ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
    results_directory =         ROOT_TEST_DIR + '/results'
    josie_results_file =        results_directory + f'/ajosie_m{mode}_k{k}.csv'
    extracted_results_file =    results_directory + f'/ajosie_m{mode}_k{k}_extracted.csv' 

    josie_res = pd.read_csv(josie_results_file)[['query_id', 'results']]

    jr = josie_res.values.tolist()
    work = [
        (r[0], s, o) 
        for r in jr 
        for i, (s, o) in enumerate(zip(
            re.findall(r'\d+', r[1])[::2] if type(r[1]) == str else [None], 
            re.findall(r'\d+', r[1])[1::2] if type(r[1]) == str else [None],
            ), start=1) if i <= ank and r[1]
        ]

    print('Start processing results...')
    with mp.Pool(processes=os.cpu_count()) as pool:
        res = pool.map(_worker_compute_sloth, work)

    # "query_res != None" because maybe there isn't enough work 
    # and a processor may return None instead of a list of tuples...
    res = [query_res for query_res in res if query_res != None]

    pd.DataFrame(res, columns=[
        'int_query_id', 
        'str_query_id', 
        'int_set_id', 
        'str_set_id', 
        'josie_overlap', 
        'sloth_overlap', 
        'difference_josie_sloth_overlap',
        'actual_set_overlap',
        'error'
        ]) \
        .convert_dtypes() \
        .to_csv(extracted_results_file, index=False)

    print(f'Extracted results saved to {extracted_results_file}.')
