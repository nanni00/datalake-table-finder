import logging
import logging.handlers
import warnings
warnings.filterwarnings('ignore')
import os
import argparse
from time import time
import multiprocessing as mp

import pandas as pd
import polars as pl
from numerize_denumerize.numerize import numerize
from tqdm import tqdm

from tools.utils.mongodb_utils import get_mongodb_collections, get_one_document_from_mongodb_by_key
from tools.utils.classes import ResultDatabase
from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import (
    apply_sloth,
    get_local_time,
    create_token_set,
    logging_setup
)



def _worker_result_extractor(chunk):
    global dbname, table_name, dataset, size, blacklist
    resultsdb = ResultDatabase(dbname, table_name)
    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=size)

    rv = []
    hit = 0

    for (algorithm, mode, (query_id, _, result_ids, algorithm_overlaps)) in tqdm(chunk, total=len(chunk), disable=False if os.getpid() % 72 == 0 else True):
        if not result_ids:
            rv.append([query_id, None, algorithm, mode, None, None, None, None, None, None])
            continue
        
        # here we need eval because on the csv file values are stored as strings
        result_ids, algorithm_overlaps = eval(result_ids), eval(algorithm_overlaps)
        
        # retrieve the query information from MongoDB
        doc_table_q = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
        assert query_id == doc_table_q['_id_numeric']
        table_q = doc_table_q['content']
        numeric_columns_q = doc_table_q['numeric_columns']

        for i, _id_r in enumerate(result_ids):
            # retrieve the result table information from MongoDB
            doc_table_r = get_one_document_from_mongodb_by_key('_id_numeric', _id_r, *collections)
            assert _id_r == doc_table_r['_id_numeric']
            table_r = doc_table_r['content']
            numeric_columns_r = doc_table_r['numeric_columns']

            # JOSIE already returns the couple exact overlap, referred to the used semantic
            # LSHForest, instead, returns only the ranked results without any other information,
            # then now compute the overlap between the query and the result tables with the 
            # overlap of the table sets with set/bag semantic
            if algorithm_overlaps:
                algorithm_overlap = algorithm_overlaps[i]
            else:
                set_q = create_token_set(table_q, 'set' if mode in ['fasttext', 'tabert'] else mode, numeric_columns_q, blacklist=blacklist)
                set_r = create_token_set(table_r, 'set' if mode in ['fasttext', 'tabert'] else mode, numeric_columns_r, blacklist=blacklist)
                algorithm_overlap = len(set(set_q).intersection(set_r))
            
            # if already exists a couple with these ID, take its computed SLOTH overlap
            r_id, s_id = (query_id, _id_r) if query_id <= _id_r else (_id_r, query_id)
            x = resultsdb.lookup_result_table(r_id, s_id), 0
            if x[0]:
                hit += 1
            sloth_overlap, sloth_time = x if x[0] else apply_sloth(table_q, table_r, numeric_columns_q, numeric_columns_r, blacklist=blacklist)
            if sloth_overlap == -1 and not x[0]:
                logging.warning(f"Pair {query_id} - {_id_r} SLOTH failed")

            # the intersection size is used for computing Jaccard Similarity or other metrics like containment, 
            # so compute using the set semantic, since it considers the intersection of the table "basic" values
            set_q = create_token_set(table_q, 'set', numeric_columns_q, blacklist=blacklist)
            set_r = create_token_set(table_r, 'set', numeric_columns_r, blacklist=blacklist)
            intersection_size = len(set(set_q).intersection(set_r))

            size_q, size_r = len(set_q), len(set_r)

            rv.append([query_id, _id_r, algorithm, mode, algorithm_overlap, sloth_overlap, size_q, size_r, intersection_size, sloth_time])
        
    mongoclient.close()
    resultsdb.close()
    return hit, rv


def chunks(sequence, chunk_size):
    # Chunks of chunk_size documents at a time.
    for j in range(0, len(sequence), chunk_size):
        yield sequence[j:j + chunk_size]



parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('--num-cpu', 
                    type=int, required=False, default=min(os.cpu_count(), 64),
                    help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
parser.add_argument('--num-query-samples',
                    type=int, required=False, default=1000,
                    help='extract results only for the given result set size (e.g. 1000)')
parser.add_argument('--dbname', 
                    type=str, required=True, default='user',
                    help='The database in which will be stored the computed SLOTH overlap for future analyses')
parser.add_argument('--dataset', 
                    required=True, choices=['wikipedia', 'gittables'])
parser.add_argument('--size', required=False, default='standard', choices=['standard', 'small'],
                    help='works on small collection versions (only for testing)')

args = parser.parse_args()
test_name =         args.test_name
nworkers =          args.num_cpu
num_query_samples = args.num_query_samples
dbname =            args.dbname
dataset =           args.dataset
size =              args.size

test_name = test_name.lower()

num_query_samples = numerize(num_query_samples, asint=True)

ROOT_TEST_DIR =             defpath.data_path.tests + f'/{test_name}'
TEST_DATASET_DIR =          ROOT_TEST_DIR + f'/{dataset}'
results_base_directory =    TEST_DATASET_DIR + '/results/base'
results_extr_directory =    TEST_DATASET_DIR + '/results/extracted'
final_results_file =        results_extr_directory + f'/final_results_q{num_query_samples}.csv'
logfile =                   TEST_DATASET_DIR + '/logging.log'

statistics_dir =            TEST_DATASET_DIR  + '/statistics'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
storage_stat_file =         statistics_dir + '/storage.csv'

logging_setup(logfile)
logging.info(f'{"#" * 10} {test_name.upper()} - {dataset.upper()} - {size.upper()} - EXTRACTION {"#" * 10}')

blacklist = {'{"$numberDouble": "NaN"}', 'comment', 'story'} if dataset == 'gittables' else set()
# blacklist = {'{"$numberDouble": "NaN"}'} if dataset == 'gittables' else set()

table_name = f'results_table_d{dataset}_s{size}_blacklist' 

if os.path.exists(final_results_file):
    logging.info(f"Extracted results file already exists at {final_results_file}, appending new results to it")
    final_results = pl.read_csv(final_results_file)
else:
    final_results = pl.DataFrame(schema={
        'query_id': pl.Int64, 
        'result_id': pl.Int64, 
        'algorithm': pl.String, 
        'mode': pl.String, 
        'algorithm_overlap': pl.Int64, 
        'sloth_overlap': pl.Int64, 
        'query_size': pl.Int64, 
        'res_tab_size': pl.Int64, 
        'intersection_mode_size': pl.Int64,
        'sloth_time(s)': pl.Float64
        }
    )

start_analysis = time()
resultsdb = ResultDatabase(dbname, table_name)
# clear the result table (occhio a farlo che poi si perdono i dati già salvati...)
resultsdb.clear()
resultsdb.create_table()

# just to know if the results database is actually useful
hit_rates = []

with mp.Pool(processes=nworkers) as pool:
    for result_file in os.listdir(results_base_directory):
        if result_file.endswith('.raw'): continue
        if f"_q{num_query_samples}.csv" not in result_file: continue
        
        results = pl.read_csv(results_base_directory + '/' + result_file)
        algorithm, mode, nsamples, k = [x[1:] for x in result_file[:-4].split('_')]
        
        logging.info(f'Extracting results from {result_file} ({algorithm}-{mode})...')
        
        sss = time()
        # keeping the same order leads to longer time
        # usually bad pairs where SLOTH fails in long time are adjacents, so 
        # a shuffle whould be better, even if it need ~x2 memory peak
        work = [(algorithm, mode, row) for row in results.sample(fraction=1.0, shuffle=True).iter_rows()]
        data = []
        
        # smaller chunk-size offer the possibility to better parallelization, because
        # SLOTH often takes time and some processes finish while others still have many computations to do 
        chunksize = max(min(len(work) // nworkers, 1000) // 4, 1)
        logging.info(f'Total work length: {len(work)}, total workers: {nworkers}, chunk-size: {chunksize}. Starting extraction...')
        extr_res = pool.map(_worker_result_extractor, chunks(work, chunksize))
        logging.info(f"Completed extraction in {round(time() - sss)}s")
        
        hit = 0
        for h, r in extr_res:
            hit += h
            data += r
            resultsdb.insert_results([[x[0], x[1], x[5]] if x[0] < x[1] else [x[1], x[0], x[5]] for x in r if x[5] != None])

        logging.info(f'hit = {hit} ({round(100 * hit / len(data), 3)}%)')
        hit_rates.append(hit / len(data))
        final_results = pl.concat([final_results, pl.DataFrame(data, schema=final_results.schema, infer_schema_length=10)])

final_results.write_csv(final_results_file)
logging.info(f"Hit rates: {hit_rates}")

# save the statistics about the analysis time
add_header = not os.path.exists(runtime_stat_file)
with open(runtime_stat_file, 'a') as rfw:
    if add_header:
        rfw.write("local_time,algorithm,mode,task,time\n")

    rfw.write(f"{get_local_time()},analysis,,extraction_q{num_query_samples},{round(time() - start_analysis, 3)}\n")

# save statistics about analysis file size
storage_size = os.path.getsize(final_results_file) / (1024 ** 3)

append = os.path.exists(storage_stat_file)
dbsize = pd.DataFrame([['analysis', f'extraction_q{num_query_samples}', storage_size]], columns=['algorithm', 'mode', 'size(GB)'])
dbsize.to_csv(storage_stat_file, index=False, mode='a' if append else 'w', header=False if append else True)

resultsdb.close()

