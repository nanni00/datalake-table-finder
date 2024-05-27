import os
import re
import argparse

import pandas as pd
import pymongo

from tools.utils.settings import DefaultPath as defpath
from tools.sloth.sloth import sloth

import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('-m', '--mode', choices=['set', 'bag'])

args = parser.parse_args()

mode =                      args.mode
test_name =                 f'm{mode}' if not args.test_name else args.test_name

ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
results_directory =         ROOT_TEST_DIR + '/results'
josie_sloth_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids.csv'
josie_results_file =        results_directory + '/result_k_5.csv'
extracted_results_file =    results_directory + '/extracted_josie_sloth_results.csv' 

josie_res = pd.read_csv(josie_results_file)[['query_id', 'results']]
josie_sloth_ids = pd.read_csv(josie_sloth_ids_file, header=None, names=['josie_id', 'sloth_id'], 
                              dtype={'josie_id': int, 'sloth_id': str})

def _worker_compute_sloth(inp):
    mongoclient = pymongo.MongoClient()
    wikitables_coll = mongoclient.optitab.turl_training_set
    snapshot_coll = mongoclient.sloth.latest_snapshot_tables
    query_id, sid, josie_overlap = inp 
    
    s_id1 = josie_sloth_ids[josie_sloth_ids['josie_id'] == query_id]['sloth_id'].values[0]
    s_id2 = josie_sloth_ids[josie_sloth_ids['josie_id'] == int(sid)]['sloth_id'].values[0]

    tab1 = wikitables_coll.find_one({'_id': s_id1})['content']        
    tab2 = wikitables_coll.find_one({'_id': s_id2})
    if not tab2:
        tab2 = snapshot_coll.find_one({'_id': s_id2})
    tab2 = tab2['content']

    tab1 = [[row[i] for row in tab1] for i in range(len(tab1[0]))]
    tab2 = [[row[i] for row in tab2] for i in range(len(tab2[0]))]
    
    metrics = []
    _, metrics = sloth(tab1, tab2, verbose=False, metrics=metrics)
    largest_ov_sloth = metrics[-2]
    return (query_id, s_id1, sid, s_id2, josie_overlap, largest_ov_sloth, int(josie_overlap) - largest_ov_sloth)


print('Start processing results...')
pool = mp.Pool(processes=os.cpu_count())
jr = josie_res.values.tolist()

work = [(r[0], s, o) for r in jr for s, o in zip(re.findall(r'\d+', r[1])[::2], re.findall(r'\d+', r[1])[1::2])]
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
