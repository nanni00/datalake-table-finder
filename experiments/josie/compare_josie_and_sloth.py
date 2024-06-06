import os
import re
import argparse

import pandas as pd
import pymongo

from tools.utils.settings import DefaultPath as defpath
from tools.sloth.sloth import sloth

import multiprocessing as mp


def _worker_compute_sloth(inp):
    global small, josie_sloth_ids
    mongoclient = pymongo.MongoClient()

    if not small:
        wikitables_coll = mongoclient.optitab.turl_training_set
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables
    else:
        wikitables_coll = mongoclient.optitab.turl_training_set_small
        snapshot_coll = mongoclient.sloth.latest_snapshot_tables_small
        
    query_id, sid, josie_overlap = inp 
    
    s_id1 = josie_sloth_ids[josie_sloth_ids['josie_id'] == query_id]['sloth_id'].values[0]
    s_id2 = josie_sloth_ids[josie_sloth_ids['josie_id'] == int(sid)]['sloth_id'].values[0]

    doc_table1 = wikitables_coll.find_one({'_id': s_id1})
    if not doc_table1:
        doc_table1 = snapshot_coll.find_one({'_id': s_id1})

    doc_table2 = wikitables_coll.find_one({'_id': s_id2})
    if not doc_table2:
        doc_table2 = snapshot_coll.find_one({'_id': s_id2})
    
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
    return (query_id, s_id1, sid, s_id2, josie_overlap, largest_ov_sloth, int(josie_overlap) - largest_ov_sloth)


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
josie_sloth_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids.csv'
josie_results_file =        results_directory + f'/result_k_{k}.csv'
extracted_results_file =    results_directory + f'/extracted_results_k_{k}.csv' 

josie_res = pd.read_csv(josie_results_file)[['query_id', 'results']]
josie_sloth_ids = pd.read_csv(josie_sloth_ids_file, header=None, names=['josie_id', 'sloth_id'], 
                              dtype={'josie_id': int, 'sloth_id': str})


jr = josie_res.values.tolist()
work = [(r[0], s, o) for r in jr for i, (s, o) in enumerate(zip(re.findall(r'\d+', r[1])[::2], re.findall(r'\d+', r[1])[1::2]), start=1) if i <= ank]

print('Start processing results...')
with mp.Pool(processes=os.cpu_count()) as pool:
    res = pool.map(_worker_compute_sloth, work)

# "query_res != None" because maybe there isn't enough work 
# and a processor may return None instead of a list of tuples...
res = [query_res for query_res in res if query_res != None]

pd.DataFrame(res, columns=[
    'josie_query_id', 
    'wiki_query_id', 
    'josie_set_id', 
    'wiki_set_id', 
    'josie_overlap', 
    'sloth_overlap', 
    'difference_josie_sloth_overlap']) \
    .to_csv(extracted_results_file, index=False)

print(f'Extracted results saved to {extracted_results_file}.')
