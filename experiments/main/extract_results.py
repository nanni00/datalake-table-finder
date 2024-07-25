import os
import argparse
from time import time
import multiprocessing as mp

import pandas as pd
import polars as pl
from numerize_denumerize.numerize import numerize
import psycopg
import psycopg.rows
from tqdm import tqdm

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import (
    apply_sloth,
    get_local_time,
    get_mongodb_collections, 
    get_one_document_from_mongodb_by_key, 
    _create_token_set
)



class ResultDatabase:
    """ Used only for testing, in order to avoid computing each time the SLOTH overlap """
    def __init__(self, dbname, table_name='results_table'):
        self.dbname = dbname
        self.table_name = table_name
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}", row_factory=psycopg.rows.dict_row)

    def create_table(self):
        q = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name   = '{self.table_name}'
            );
        """
        exists = self._dbconn.execute(q).fetchone()['exists'] == True
        if not exists:
            self._dbconn.execute(
                f"""
                CREATE TABLE {self.table_name} (
                    r_id integer NOT NULL,
                    s_id integer NOT NULL,
                    sloth_overlap integer NOT NULL,
                    PRIMARY KEY (r_id, s_id)
                );

                CREATE INDEX {self.table_name}_r_id_index ON {self.table_name}(r_id, s_id);
                """
            )
        else:
            print('Results table already exists')
        self._dbconn.commit()

    def insert_results(self, values:list[list[int,int,int]]):
        self._dbconn \
            .cursor() \
                .executemany(f"INSERT INTO {self.table_name} VALUES(%s, %s, %s) ON CONFLICT (r_id, s_id) DO NOTHING RETURNING (r_id);", values)
        self._dbconn.commit()

    def lookup_result_table(self, r_id, s_id):
        """ Where the r_id < s_id """

        result = self._dbconn.execute(
            f"""
            SELECT sloth_overlap FROM {self.table_name}
            WHERE r_id = {r_id} AND s_id = {s_id}
            """
        )

        try:
            result = result.fetchone()
        except psycopg.ProgrammingError:
            print('error', r_id, s_id)
            raise Exception()

        return None if result == None else result['sloth_overlap']
        
    def clear(self):
        self._dbconn.execute(f"TRUNCATE {self.table_name};")
        self._dbconn.commit()

    def close(self):
        self._dbconn.close()



def _worker_result_extractor(inp):
    global dbname, table_name, small
    resultsdb = ResultDatabase(dbname, table_name)
    algorithm, mode, (query_id, _, result_ids, algorithm_overlaps) = inp

    if not result_ids:
        return [[query_id, None, algorithm, mode, None, None, None, None, None]]
    
    # here we need eval because on csv values are stored as strings
    result_ids, algorithm_overlaps = eval(result_ids), eval(algorithm_overlaps)
    
    mongoclient, collections = get_mongodb_collections(small=small)
    
    # retrieve the query information from MongoDB
    doc_table_q = get_one_document_from_mongodb_by_key('_id_numeric', query_id, *collections)
    table_q = doc_table_q['content']
    numeric_columns_q = doc_table_q['numeric_columns']

    rv = []
    for i, r in enumerate(result_ids):
        # retrieve the result table information from MongoDB
        doc_tables_r = get_one_document_from_mongodb_by_key('_id_numeric', r, *collections)
        table_r = doc_tables_r['content']
        numeric_columns_r = doc_tables_r['numeric_columns']

        # JOSIE already returns the couple exact overlap, referred to the used semantic
        # LSHForest, instead, returns only the ranked results without any other information,
        # then now compute the overlap between the query and the result tables with the 
        # overlap of the table sets with set/bag semantic
        if algorithm_overlaps:
            algorithm_overlap = algorithm_overlaps[i]
        else:
            set_q = _create_token_set(table_q, 'set' if mode in ['fasttext', 'tabert'] else mode, numeric_columns_q)
            set_r = _create_token_set(table_r, 'set' if mode in ['fasttext', 'tabert'] else mode, numeric_columns_r)
            algorithm_overlap = len(set(set_q).intersection(set_r))
        
        # if already exists a couple with these ID, take its computed SLOTH overlap
        r_id, s_id = (query_id, r) if query_id <= r else (r, query_id)

        x = resultsdb.lookup_result_table(r_id, s_id)
        if x != None:
            sloth_overlap = x
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
    resultsdb.close()
    return rv




parser = argparse.ArgumentParser()
parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('--small', required=False, action='store_true',
                    help='works on small collection versions (only for testing)')
parser.add_argument('--num-cpu', 
                    type=int, required=False, default=min(os.cpu_count(), 96),
                    help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
parser.add_argument('--num-query-samples',
                    type=int, required=False, default=1000,
                    help='extract results only for the given result set size (e.g. 1000)')
parser.add_argument('--dbname', 
                    type=str, required=True, default='user',
                    help='The database in which will be stored the computed SLOTH overlap for future analyses')

args = parser.parse_args()
test_name =         args.test_name
small =             args.small
nworkers =          args.num_cpu
num_query_samples = args.num_query_samples
dbname =            args.dbname

table_name='results_table' if not small else 'results_table_small'

num_query_samples = numerize(num_query_samples, asint=True)

ROOT_TEST_DIR =             defpath.data_path.tests + f'/{test_name}'
results_base_directory =    ROOT_TEST_DIR + '/results/base'
results_extr_directory =    ROOT_TEST_DIR + '/results/extracted'
final_results_file =        results_extr_directory + f'/final_results_q{num_query_samples}.csv'

statistics_dir =            ROOT_TEST_DIR  + '/statistics'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
storage_stat_file =         statistics_dir + '/storage.csv'

if os.path.exists(final_results_file):
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
        'intersection_mode_size': pl.Int64
        }
    )

start_analysis = time()
resultsdb = ResultDatabase(dbname, table_name)
resultsdb.create_table()

# clear the result table (occhio a farlo che poi si perdono i dati già salvati...)
# resultsdb.clear()

with mp.Pool(processes=nworkers) as pool:
    for result_file in os.listdir(results_base_directory):
        if result_file.endswith('.raw'):
            continue
        
        if f"_q{num_query_samples}.csv" not in result_file:
            continue
        print(result_file)
        results = pl.read_csv(results_base_directory + '/' + result_file)
        algorithm, mode, nsamples, k = [x[1:] for x in result_file[:-4].split('_')]
        
        print(f'{get_local_time()} Working on {algorithm}-{mode}')
        
        sss = time()
        work = [(algorithm, mode, row) for row in results.iter_rows()]
        data = []

        chunksize = min(len(work) // nworkers, 50) # in order to (hopefully) use multiple pairs...
        print(nworkers, len(work), ' chunksize: ', chunksize)
        print(f'{get_local_time()} Created work block. Starting extraction...')
        for r in tqdm(pool.imap(_worker_result_extractor, work, chunksize=chunksize), total=len(work)):
            data += r
            resultsdb.insert_results([[x[0], x[1], x[4]] if x[0] < x[1] else [x[1], x[0], x[4]] for x in r if x[1] != None])

        print(f'{get_local_time()} Workers have finished')
        
        final_results = pl.concat([final_results, pl.DataFrame(data, schema=final_results.schema, infer_schema_length=10)])
        print(f"{get_local_time()} Completed: {round(time() - sss)}s")

final_results.write_csv(final_results_file)

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

