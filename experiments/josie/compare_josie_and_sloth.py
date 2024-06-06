import os
from pprint import pprint
import re
import argparse

import pandas as pd
import pymongo

from tools.utils.settings import DefaultPath as defpath
from tools.sloth.sloth import sloth

import multiprocessing as mp


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
    
    doc_table1 = wikitables_coll.find_one({'_id_numeric': query_id})
    if not doc_table1:
        doc_table1 = snapshot_coll.find_one({'_id_numeric': query_id})

    doc_table2 = wikitables_coll.find_one({'_id_numeric': sid})
    if not doc_table2:
        doc_table2 = snapshot_coll.find_one({'_id_numeric': sid})
    
    str_id1 = doc_table1['_id']
    str_id2 = doc_table2['_id']

    table1 = doc_table1['content']
    table2 = doc_table2['content']

    numeric_columns1 = doc_table1['numeric_columns']
    numeric_columns2 = doc_table2['numeric_columns']

    num_null = 0

    def format_value_for_excluding_nan(t):
        nonlocal num_null 
        if not t or pd.isna(t):
            num_null += 1
            return f'{t}@{num_null}'
        return t
    
    table1 = [[format_value_for_excluding_nan(row[i]) for row in table1] for i in range(len(table1[0])) if numeric_columns1[i] == 0]
    table2 = [[format_value_for_excluding_nan(row[i]) for row in table2] for i in range(len(table2[0])) if numeric_columns2[i] == 0]
    
    metrics = []
    _, metrics = sloth(table1, table2, verbose=False, metrics=metrics)
    largest_ov_sloth = metrics[-2]
    return (query_id, str_id1, sid, str_id2, josie_overlap, largest_ov_sloth, int(josie_overlap) - largest_ov_sloth)


parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('-k', required=True, type=int)
parser.add_argument('--analyse-up-to', required=False, type=int)
parser.add_argument('--small', required=False, action='store_true',
                    help='works on small collection versions (only for testing)')

args = parser.parse_args()
test_name = args.test_name
k =         args.k
ank =       args.analyse_up_to
small =     args.small

ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
results_directory =         ROOT_TEST_DIR + '/results'
josie_results_file =        results_directory + f'/result_k_{k}.csv'
extracted_results_file =    results_directory + f'/extracted_results_k_{k}.csv' 

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
    'difference_josie_sloth_overlap']) \
    .convert_dtypes() \
    .to_csv(extracted_results_file, index=False)

print(f'Extracted results saved to {extracted_results_file}.')
