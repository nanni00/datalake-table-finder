import os
import argparse
from time import time
import multiprocessing as mp

import pandas as pd
import polars as pl
from numerize_denumerize.numerize import numerize


from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import (
    apply_sloth,
    get_local_time,
    get_mongodb_collections, 
    get_one_document_from_mongodb_by_key, 
    _create_token_set
)


def _worker_result_extractor(inp):
    global tot_iter, lock, global_results
    algorithm, mode, (query_id, _, result_ids, algorithm_overlaps) = inp
    if not result_ids:
        return [[query_id, None, algorithm, mode, None, None, None, None, None]]
    result_ids, algorithm_overlaps = eval(result_ids), eval(algorithm_overlaps)
    
    mongoclient, collections = get_mongodb_collections(small=False)
    
    doc_table_q = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
    table_q = doc_table_q['content']
    numeric_columns_q = doc_table_q['numeric_columns']

    rv = []
    for i, r in enumerate(result_ids):
        doc_tables_r = get_one_document_from_mongodb_by_key('_id_numeric', r, *collections)
        table_r = doc_tables_r['content']
        numeric_columns_r = doc_tables_r['numeric_columns']

        # JOSIE already returns the couple exact overlap, referred to the used semantic
        # LSHForest, instead, returns only the ranked results without any other information,
        # then now compute the overlap between the query and the result tables
        if algorithm_overlaps:
            algorithm_overlap = algorithm_overlaps[i]
        else:
            set_q = _create_token_set(table_q, mode, numeric_columns_q)
            set_r = _create_token_set(table_r, mode, numeric_columns_r)
            algorithm_overlap = len(set(set_q).intersection(set_r))
        
        # if already exists a couple with these ID, take its computed SLOTH overlap
        x = global_results.filter(((pl.col('query_id') == query_id) & pl.col('result_id') == r) | ((pl.col('query_id') == r) & pl.col('result_id') == query_id))
        if x.shape[0] > 0:
            sloth_overlap = x.rows()[0][4]
        else:
            sloth_overlap = apply_sloth(table_q, table_r, numeric_columns_q, numeric_columns_r)

        # the intersection size is used for computing Jaccard Similarity or other metrics like containment, 
        # so compute using the set semantic, since it considers the intersection of the table "basic" values
        set_q = _create_token_set(table_q, 'set', numeric_columns_q)
        set_r = _create_token_set(table_r, 'set', numeric_columns_r)
        intersection_size = len(set(set_q).intersection(set_r))

        size_q, size_r = len(set_q), len(set_r)

        rv.append([query_id, r, algorithm, mode, algorithm_overlap, sloth_overlap, size_q, size_r, intersection_size])
    
    mongoclient.close()
    return rv




parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('--small', required=False, action='store_true',
                    help='works on small collection versions (only for testing)')
parser.add_argument('-w', '--num-workers', 
                    type=int, required=False, default=min(os.cpu_count(), 64),
                    help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
parser.add_argument('--num-query-samples',
                    type=int, required=False, default=1)


args = parser.parse_args()
test_name = args.test_name
small =     args.small
nworkers =  args.num_workers
num_query_samples = args.num_query_samples


ROOT_TEST_DIR =             defpath.data_path.tests + f'/{test_name}'
results_base_directory =    ROOT_TEST_DIR + '/results/base'
results_extr_directory =    ROOT_TEST_DIR + '/results/extracted'
global_results_file =       results_extr_directory + f'/final_results_q{numerize(num_query_samples, asint=True)}.csv'

statistics_dir =            ROOT_TEST_DIR  + '/statistics'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
storage_stat_file =         statistics_dir + '/storage.csv'


global_results = pl.DataFrame(schema={
    'query_id': pl.Int64, 
    'result_id': pl.Int64, 
    'algorithm': pl.String, 
    'mode': pl.String, 
    'algorithm_overlap': pl.Int64, 
    'sloth_overlap': pl.Int64, 
    'query_size': pl.Int64, 
    'res_tab_size': pl.Int64, 
    'intersection_mode_size': pl.Int64
    })


start_analysis = time()
with mp.Pool(processes=nworkers) as pool:
    for result_file in os.listdir(results_base_directory):
        if result_file.endswith('.raw'):
            continue
        
        if f"_q{numerize(num_query_samples, asint=True)}.csv" not in result_file:
            continue
        print(result_file)
        results = pl.read_csv(results_base_directory + '/' + result_file)
        algorithm, mode, nsamples, k = [x[1:] for x in result_file[:-4].split('_')]
        print(f"Working on {algorithm}-{mode}")
        sss = time()
        work = [(algorithm, mode, row) for row in results.iter_rows()]
        data = []

        print(get_local_time(), ' Starting work...')

        result = pool.map(_worker_result_extractor, work)

        for r in result:
            data += r

        global_results = pl.concat([global_results, pl.DataFrame(data, schema=global_results.schema, infer_schema_length=10)])
        print(f"{get_local_time()} Completed: {round(time() - sss)}s")

global_results.write_csv(global_results_file)

add_header = not os.path.exists(runtime_stat_file)
with open(runtime_stat_file, 'a') as rfw:
    if add_header:
        rfw.write("local_time,algorithm,mode,task,time\n")

    rfw.write(f"{get_local_time()},analysis,analysis,analysis,{round(time() - start_analysis, 3)}\n")

storage_size = os.path.getsize(global_results_file) / (1024 ** 3)

append = os.path.exists(storage_stat_file)
dbsize = pd.DataFrame([['analysis', 'analysis', storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

