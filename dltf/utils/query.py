import json
import time
import random
import multiprocessing as mp

from dltf.utils.misc import chunks
from dltf.utils.tables import is_valid_table
from dltf.utils.datalake import DataLakeHandlerFactory


def _sample_query_task(data):
    chunk, data_lake_args = data[0], data[1:]
    dlh = DataLakeHandlerFactory.create_handler(*data_lake_args)
    s = set()
    for table_id in chunk:
        table_obj = dlh.get_table_by_numeric_id(table_id)
        if not table_obj or not is_valid_table(table_obj['content'], table_obj['valid_columns']):
            continue
        
        s.add(table_id)
    dlh.close()
    return s


def sample_queries(output_query_json, nsamples, num_cpu, *data_lake_args):
    """This assumes that the tables IDs are consecutive numbers from 0 to #num_tables"""
    s = set()
    start = time.time()
    dlh = DataLakeHandlerFactory.create_handler(*data_lake_args)
    N = dlh.count_tables()
    dlh.close()
    
    print(f'Sampling {nsamples} tables from {N} total tables...')

    with mp.get_context('spawn').Pool(num_cpu) as pool:
        while len(s) < nsamples: 
            work = random.sample(range(N), nsamples - len(s))
            chunk_size = max((nsamples - len(s)) // num_cpu, 1)
            results = pool.map(_sample_query_task, chunks(work, chunk_size, *data_lake_args))
            for taskres in results:
                for x in taskres:
                    s.add(int(x))
            print(f'Sampled {len(s)} ({round(len(s) * 100 / nsamples)}%)', end='\r')
            if time.time() - start > 3:
                break
    samples = {'_id_numeric': list(s)[:nsamples]}
    
    print(f"Sampled {len(s)} tables ({round(len(s) * 100 / nsamples)}%).")
    with open(output_query_json, 'w') as wf:
        json.dump(samples, wf, indent=1)
    return len(samples['_id_numeric'])


def read_query_ids(query_file):
    with open(query_file) as fr:
        return sorted(json.load(fr)['_id_numeric'])
