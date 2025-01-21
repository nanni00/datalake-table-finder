import os
import random
import warnings
from time import time
import multiprocessing as mp
warnings.filterwarnings('ignore')

import polars as pl
from tqdm import tqdm

from dltf.utils.tables import table_to_tokens
from dltf.utils.misc import chunks
from dltf.utils.metrics import proximity
from dltf.utils.overlapdb import OverlapsDBHandler
from dltf.utils.settings import get_all_paths
from dltf.utils.loghandler import logging_setup, info
from dltf.utils.datalake import DataLakeHandlerFactory
from dltf.utils.misc import largest_overlap_sloth, numerize


def worker_result_extractor(data):
    # TODO a different connection for each pool process isn't a great way
    # maybe use a SessionMaker is better, or smth else
    # TODO sometimes N>1 processes satisfy pid % num_cpu == 0, how to create some
    # sort of info/debug here?
    global dlhargs, num_cpu, blacklist, translators, patterns, connection_info
    debug_process = os.getpid() % (num_cpu) == 0
    
    chunk = data[0]
    dlh = DataLakeHandlerFactory.create_handler(*dlhargs)
    resultsdb = OverlapsDBHandler(connection_info=connection_info)
    
    rv = []
    hit = 0
    
    for (algorithm, mode, query_id, result_id) in tqdm(chunk, total=len(chunk), leave=False, disable=not debug_process):
        # here we need eval because on the csv file the values are stored as strings
        # result_ids, algorithm_overlaps = eval(result_ids), eval(algorithm_overlaps)
        
        # retrieve the query information
        doc_q = dlh.get_table_by_numeric_id(query_id)
        assert doc_q is not None, print(query_id)
        assert query_id == doc_q['_id_numeric']
        table_q = doc_q['content']
        valid_columns_q = doc_q['valid_columns']

        # retrieve the result table
        doc_r = dlh.get_table_by_numeric_id(result_id)
        assert result_id == doc_r['_id_numeric']
        table_r = doc_r['content']
        valid_columns_r = doc_r['valid_columns']

        r_id, s_id = (query_id, result_id) if query_id <= result_id else (result_id, query_id)
        

        if lookup_results := resultsdb.lookup(r_id, s_id):
            # if already exists a couple with these ID, 
            # take its already computed information
            hit += 1
            (
                sloth_overlap, 
                set_q_size, set_r_size, set_overlap, set_union_size,
                bag_q_size, bag_r_size, bag_overlap, 
                sloth_time, set_time, bag_time
            ) = lookup_results
        else:
            # otherwise, compute all the necessary metadata
            sloth_overlap, sloth_time = largest_overlap_sloth(table_q, table_r, valid_columns_q, valid_columns_r, blacklist=blacklist, verbose=False)
            sloth_overlap = max(sloth_overlap, 0)
            
            # the intersection size is used for computing Jaccard Similarity or other metrics like containment, 
            # so compute using the set semantic, since it considers the intersection of the table "basic" values
            set_q = set(table_to_tokens(table_q, valid_columns_q, 'set', None, blacklist, translators, patterns))
            set_r = set(table_to_tokens(table_r, valid_columns_r, 'set', None, blacklist, translators, patterns))
            sstart = time()
            set_overlap = len(set_q & set_r)
            set_time = time() - sstart
            assert set_overlap <= min(len(set_q), len(set_r))

            set_union_size = len(set_q | set_r)

            bag_q = set(table_to_tokens(table_q, valid_columns_q, 'bag', None, blacklist, translators, patterns))
            bag_r = set(table_to_tokens(table_r, valid_columns_r, 'bag', None, blacklist, translators, patterns))
            bstart = time()
            bag_overlap = len(bag_q & bag_r)
            bag_time = time() - bstart
            assert bag_overlap <= min(len(bag_q), len(bag_r))

            set_q_size, set_r_size = len(set_q), len(set_r)
            bag_q_size, bag_r_size = len(bag_q), len(bag_r)

        algorithm_overlap = set_overlap if mode in ['set', 'ft', 'ftdist'] else bag_overlap
            
        if set_overlap > 0:
            try:
                jaccard_sim =               set_overlap / set_union_size
                multi_jaccard_sim =         set_overlap / (set_q_size + set_r_size)
                containment =               set_overlap / set_q_size
                overlap_set_similarity =    set_overlap / min(bag_q_size, bag_r_size)
                prox =                      proximity(sloth_overlap, algorithm_overlap)

                # the area ratio, as stated in SLOTH paper 
                # is "the largest overlap normalized by the area of the smaller table", 
                # and we could consider the token set in bag mode as the table area
                area_ratio = sloth_overlap / min(bag_q_size, bag_r_size)
            except ZeroDivisionError:
                jaccard_sim = multi_jaccard_sim = containment = overlap_set_similarity = area_ratio = prox = 0
        else:
            jaccard_sim = multi_jaccard_sim = containment = overlap_set_similarity = area_ratio = prox = 0

        rv.append([
            # 0...5
            query_id, result_id, algorithm, mode, sloth_overlap, algorithm_overlap,
            
            # 6...12 
            set_q_size, set_r_size, set_overlap, set_union_size,
            bag_q_size, bag_r_size, bag_overlap,
            
            # 13...18
            jaccard_sim, 
            multi_jaccard_sim, 
            containment,
            overlap_set_similarity,
            prox,
            area_ratio,
            
            # 19...21
            round(sloth_time, 5),
            round(set_time, 5),
            round(bag_time, 5)
        ])

    dlh.close()
    resultsdb.close()
    return hit, rv


def initializer(_dlhargs, _num_cpu, _blacklist, _translators, _patterns, _connection_info):
    global dlhargs, blacklist, num_cpu, connection_info, translators, patterns
    dlhargs         = _dlhargs
    blacklist       = _blacklist
    num_cpu         = _num_cpu
    translators     = _translators
    patterns        = _patterns
    connection_info = _connection_info
    

def extract_results(test_name, 
                    k, 
                    num_query_samples,
                    datalake_location:str,
                    datalake_name:str, 
                    datalake_options:list[str],
                    string_blacklist, 
                    string_translators,
                    string_patterns,
                    num_cpu,
                    connection_info:dict, 
                    clear_results_table=False):
    
    assert int(k) > 0
    assert int(num_cpu) > 0
    
    final_results = pl.DataFrame(schema={
        'query_id': pl.Int32, 
        'result_id': pl.Int32, 
        'algorithm': pl.String, 
        'mode': pl.String, 
        'sloth_overlap': pl.Int32,
        'algorithm_overlap': pl.Int32,

        'set_size_q': pl.Int32,
        'set_size_r': pl.Int32, 
        'set_overlap': pl.Int32,
        'set_union_size': pl.Int32,

        'bag_size_q': pl.Int32, 
        'bag_size_r': pl.Int32, 
        'bag_overlap': pl.Int32,
        
        'jaccard_sim': pl.Float32,
        'multi_jaccard_sim': pl.Float32,
        'containment': pl.Float32,
        'overlap_set_sim': pl.Float32,
        'proximity': pl.Float32,
        'area_ratio': pl.Float32,
        
        'sloth_time(s)': pl.Float32,
        'set_time(s)': pl.Float32,
        'bag_time(s)': pl.Float32
        }
    )

    num_query_samples   = numerize(num_query_samples)
    paths               = get_all_paths(test_name.lower(), datalake_name, k, num_query_samples)
    final_results_file  = f"{paths['results_extr_dir']}/k{k}_q{num_query_samples}.csv"
    results_base_dir    = paths['results_base_dir']
    string_blacklist    = set(string_blacklist)
    dlhargs             = [datalake_location, datalake_name, datalake_options]
    
    logging_setup(paths['logfile'])

    info(f' {test_name.upper()} - {k} - {num_query_samples} - EXTRACTION '.center(150, '#'))
    
    # create an instance of the OverlapsDBHandler to create the overlaps table 
    overlaps_dbhandler = OverlapsDBHandler(connection_info=connection_info)
    
    # clear the result table (occhio a farlo che poi si perdono i dati già salvati...)
    if clear_results_table:
        info('Clearing overlaps table...')
        overlaps_dbhandler.drop_table()
    overlaps_dbhandler.create_table()
    
    # just to know if the results database is actually useful
    hit_rates = []

    # Arguments passed to parallel workers
    initargs = (dlhargs, num_cpu, string_blacklist, string_translators, string_patterns, connection_info)

    with mp.get_context('spawn').Pool(processes=num_cpu, initializer=initializer, initargs=initargs) as pool:
        for result_file in os.listdir(results_base_dir):
            if result_file.endswith('.raw'): continue
            
            algorithm, mode = result_file[:-4].split('_')
            results = pl.read_csv(f'{results_base_dir}/{result_file}')
            
            # Keeping the same order leads to longer time.
            # Usually bad pairs where SLOTH fails in long time are adjacents, 
            # shuffling distributes these pairs
            work = [(algorithm, mode, row[0], result_id) for row in results.iter_rows() if row[2] != None for result_id in eval(str(row[2]))]
            random.shuffle(work)
            data = []
            
            # Smaller chunk-size offer the possibility to better parallelization, because
            # SLOTH often takes time and some processes finish while others still have many computations to do 
            chunksize = max(min(len(work) // num_cpu, 1000), 1)
            info(f'Extracting results from {result_file} ({algorithm}-{mode})...')
            info(f'Total work length: {len(work)}, total workers: {num_cpu}, chunk-size: {chunksize}. Starting extraction...')
            etraction_time_start = time()
            extr_res = pool.map(worker_result_extractor, chunks(work, chunksize))
            info(f"Completed extraction in {round(time() - etraction_time_start)}s")
            
            # Store the computed metadata and results on the database
            hit = 0
            for h, r in extr_res:
                hit += h
                data += r
                overlaps_dbhandler.add_overlaps([[x[0], x[1], x[4], *x[6:13], *x[19:]] 
                                                 if x[0] < x[1] else 
                                                 [x[1], x[0], x[4], *x[6:13], *x[19:]] 
                                                 for x in r if x[4] != None])

            info(f'hit = {hit} ({round(100 * hit / len(data), 3)}%)')
            hit_rates.append(hit / len(data))
            final_results = pl.concat([final_results, pl.DataFrame(data, schema=final_results.schema, infer_schema_length=10)])
            final_results.write_csv(final_results_file)

    info(f"Hit rates: {hit_rates}")
    overlaps_dbhandler.close()
