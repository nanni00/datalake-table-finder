import os
from pprint import pprint
import sys
import warnings

import pandas as pd
import psycopg.rows
warnings.filterwarnings('ignore')
from time import time
from collections import defaultdict

import psycopg
import pyspark
from tqdm import tqdm

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import print_info

from tools.josiedataprep.db import *
from tools.josiedataprep.preparation_functions import (
    parallel_extract_starting_sets_from_tables,
    extract_starting_sets_from_tables, 
    create_index, 
    create_raw_tokens
)
from tools.josiedataprep.josie_results_analysis import extract_from_josie_results_pairs_and_overlaps


def do_test(n_tables=500, mode='set', spark_context=None, k=3):
    ############# SET UP #############
    runtime_metrics = defaultdict(float)
    
    dbname = f'sloth_n{n_tables}_m{mode}'
    table_db_prefix = dbname

    intput_tables_csv_dir =     defpath.data_path.wikitables + '/sloth-subset/csv'
    ROOT_TEST_DIR =             defpath.josie_stuff_path.data + f'/test-{n_tables}-{mode}'
    statistics_dir =            ROOT_TEST_DIR + '/statistics'
    
    tables_stat_file =          statistics_dir + '/tables.csv'
    columns_stat_file =         statistics_dir + '/columns.csv'
    runtime_stat_file =         statistics_dir + '/runtime.csv'     
    db_stat_file =              statistics_dir + '/db.csv'

    josie_to_sloth_id_file =    ROOT_TEST_DIR + '/josie_sloth_ids.csv'
    raw_tokens_file =           ROOT_TEST_DIR + '/tables.raw-tokens' 
    set_file =                  ROOT_TEST_DIR + '/tables.set'
    integer_set_file =          ROOT_TEST_DIR + '/tables.set-2'
    inverted_list_file =        ROOT_TEST_DIR + '/tables.inverted-list'
    
    josie_results_file =        ROOT_TEST_DIR + '/result_k_3.csv'
    pairs_with_overlap_file =   ROOT_TEST_DIR + '/pairs_overlap.csv'


    if os.path.exists(ROOT_TEST_DIR):
        if True:
        # if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (y/n) ') == 'y':
            os.system(f'rm -rf {ROOT_TEST_DIR}')
        else:
            print('Cannot continue if directory already exists. Exiting...')
            sys.exit()

    print_info(msg_before=f'Creating test directory {ROOT_TEST_DIR}...', msg_after='Completed.') \
                (os.system)(f'mkdir -p {ROOT_TEST_DIR}')
    
    print_info(msg_before=f'Creating test statistics directory {statistics_dir}...', msg_after='Completed.') \
                (os.system)(f'mkdir -p {statistics_dir}')
    
    if not spark_context:
        conf = pyspark.SparkConf().setAppName('CreateIndex')    
        spark_context = print_info(msg_before='Creating Spark context...', msg_after='Completed.') \
                                    (pyspark.SparkContext)(conf=conf)

    ############# DATA PREPARATION #############
    start = time()
    # extract_starting_sets_from_tables(
    parallel_extract_starting_sets_from_tables(
        input_tables_csv_dir=intput_tables_csv_dir,
        final_set_file=set_file,
        id_table_file=josie_to_sloth_id_file,
        tables_stat_file=tables_stat_file,
        columns_stat_file=columns_stat_file,
        ntables_to_load_as_set=n_tables,
        with_=mode
    )
    runtime_metrics['1.extract_sets'] = round(time() - start, 5)

    start = time()
    create_raw_tokens(
        set_file,
        raw_tokens_file,
        spark_context=spark_context,
        single_txt=True
    )
    runtime_metrics['2.create_raw_tokens'] = round(time() - start, 5)
    
    start = time()
    create_index(
        set_file,
        integer_set_file,
        inverted_list_file,
        spark_context
    )
    runtime_metrics['3.create_index'] = round(time() - start, 5)

    start = time()
    with psycopg.connect(f"port=5442 host=/tmp dbname={dbname}") as conn:
        with conn.cursor() as cur:
            drop_tables(cur, table_db_prefix)
            create_tables(cur, table_db_prefix)
            
            parts = [p for p in os.listdir(inverted_list_file) if p.startswith('part-')]
            print(f'Inserting inverted lists...')
            for i, part in tqdm(enumerate(parts), total=len(parts)):
                insert_data_into_inverted_list_table(cur, f'{inverted_list_file}/{part}', table_db_prefix)
            print('Completed')

            parts = [p for p in os.listdir(integer_set_file) if p.startswith('part-')]
            print(f'Inserting integer sets...')
            for i, part in tqdm(enumerate(parts), total=len(parts)):
                insert_data_into_sets_table(cur, f'{integer_set_file}/{part}', table_db_prefix)
            print('Completed')

            create_sets_index(cur, table_db_prefix)
            create_inverted_list_index(cur, table_db_prefix)
        conn.commit()
    runtime_metrics['4.db_operations'] = round(time() - start, 5)
    
    ############# RUNNING JOSIE #############
    HOME = os.environ['HOME']
    josie_cmd_dir = f'{HOME}/go/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)

    # in this basic case every set is used as query set
    start = time()
    os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
              --pg-database={dbname} \
                --pg-table-queries={table_db_prefix}_sets \
                    --benchmark={table_db_prefix}')
    
    os.system(f'go run {josie_cmd_dir}/topk/main.go \
              --pg-database={dbname} \
                --benchmark={table_db_prefix} \
                    --output={ROOT_TEST_DIR} \
                        --k={k}')
    runtime_metrics['5.josie'] = round(time() - start, 5)

    runtime_metrics['total_time'] = sum(runtime_metrics.values())

    ############# GETTING DATABASE STATISTICS #############
    with psycopg.connect(f"port=5442 host=/tmp dbname={dbname}") as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            pd.DataFrame(get_statistics_from_(cur, table_db_prefix)).to_csv(db_stat_file, index=False)
    
    pd.DataFrame([runtime_metrics]).to_csv(runtime_stat_file, index=False)

    ############# ANALYSING JOSIE RESULTS #############
    # extract_from_josie_results_pairs_and_overlaps(josie_results_file, pairs_with_overlap_file)


if __name__ == '__main__':
    # create the SparkContext object (very basic)
    conf = pyspark.SparkConf().setAppName('CreateIndex')    
    spark_context = print_info(msg_before='Creating Spark context...', msg_after='Completed.') \
        (pyspark.SparkContext.getOrCreate)(conf=conf)
    
    for n in [45000]:        
        start = time()
        do_test(n, mode='set', spark_context=spark_context, k=5)
        print(f'\n\n\nTotal time for n={n}: {round(time() - start, 3)}s', end='\n\n\n')

    # stop the SparkContext object
    spark_context.cancelAllJobs()
    spark_context.stop()