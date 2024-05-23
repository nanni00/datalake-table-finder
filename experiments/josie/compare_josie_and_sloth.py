import os
import re
import argparse

import pandas as pd
import pymongo

from tools.josiestuff.datapreparation import _create_token_set
from tools.utils.settings import DefaultPath as defpath
from tools.sloth.sloth import sloth

import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('-m', '--mode', choices=['set', 'bag'])

args = parser.parse_args()

mode =      args.mode
test_name =  f'm{mode}' if not args.test_name else args.test_name


ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
josie_results_file =        ROOT_TEST_DIR + '/results/result_k_5.csv'
josie_sloth_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids.csv'
extracted_results_file =    ROOT_TEST_DIR + '/results/extracted_josie_sloth_results.csv' 

josie_res = pd.read_csv(josie_results_file)[['query_id', 'results']]
josie_sloth_ids = pd.read_csv(josie_sloth_ids_file, header=None)
josie_sloth_ids.rename({0: 'josie_id', 1: 'sloth_id'}, axis='columns', inplace=True)


def _worker_compute_sloth(inp):
    final_results = []
    mongoclient = pymongo.MongoClient()
    wikitables_coll = mongoclient.optitab.turl_training_set
    
    query_id, results = inp 
    sids, overlaps = re.findall(r'\d+', results)[::2], re.findall(r'\d+', results)[1::2]
    s_id1 = josie_sloth_ids[josie_sloth_ids['josie_id'] == query_id]['sloth_id'].values[0]

    for sid, josie_overlap in zip(sids, overlaps):                
        s_id2 = josie_sloth_ids[josie_sloth_ids['josie_id'] == int(sid)]['sloth_id'].values[0]
        tab1 = wikitables_coll.find({'_id': s_id1}).next()['content']
        tab2 = wikitables_coll.find({'_id': s_id2}).next()['content']

        set1 = set(_create_token_set(tab1, mode))
        set2 = set(_create_token_set(tab2, mode))

        tab1 = [[row[i] for row in tab1] for i in range(len(tab1[0]))]
        tab2 = [[row[i] for row in tab2] for i in range(len(tab2[0]))]

        metrics = []
        _, metrics = sloth(tab1, tab2, verbose=False, metrics=metrics)

        largest_ov_sloth = metrics[-2]
        my_overlap = len(set1.intersection(set2))
        error = abs(len(set1.intersection(set2)) - int(josie_overlap)) not in (0, 1)
        
        final_results.append((query_id, s_id1, sid, s_id2, josie_overlap, my_overlap, largest_ov_sloth, error))
    return final_results



pool = mp.Pool(processes=os.cpu_count())
res = pool.map(_worker_compute_sloth, josie_res.values.tolist())

# "query_res != None" because maybe there isn't enough work and a processor may return None instead of a list of tuples...
res = [r for query_res in res if query_res != None for r in query_res]

pd.DataFrame(res, columns=['josie_query_id', 'wiki_query_id', 'josie_set_id', 'wiki_set_id', 'josie_overlap', 'checked_overlap', 'sloth_overlap', 'error']) \
    .to_csv(extracted_results_file, index=False)

