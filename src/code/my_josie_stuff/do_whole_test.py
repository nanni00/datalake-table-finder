import os
import sys
from time import time
import psycopg
import pyspark
from tqdm import tqdm
from code.utils.settings import DefaultPath

from code.my_josie_stuff.db_loading import *
from code.my_josie_stuff.preparation_functions import extract_starting_sets_from_tables, create_index, create_raw_tokens 
from code.my_josie_stuff.josie_results_analysis import extract_from_josie_results_pairs_and_overlaps


def do_test(n_tables=500, spark_context=None):
    ############# SET UP #############
    maindb = 'giovanni'
    dbname = f'sloth{n_tables}'
    table_db_prefix = dbname

    input_tables_file =  DefaultPath.data_path.wikitables + 'threshold-r5-c2-a50/sloth-tables-r5-c2-a50.jsonl'
    ROOT_TEST_DIR = DefaultPath.josie_stuff_path.base + f'josie-test-{n_tables}/'
    
    josie_to_sloth_id_file =    ROOT_TEST_DIR + 'id_table.csv'
    raw_tokens_file =           ROOT_TEST_DIR + 'sloth_tables.raw-tokens' 
    set_file =                  ROOT_TEST_DIR + 'sloth_tables.set'
    integer_set_file =          ROOT_TEST_DIR + 'sloth_tables.set-2'
    inverted_list_file =        ROOT_TEST_DIR + 'sloth_tables.inverted-list'
    
    josie_results_file =        ROOT_TEST_DIR + f'result_k_3.csv'
    josie_to_sloth_id_file =    ROOT_TEST_DIR + 'id_table.csv'
    pairs_with_overlap_file =   ROOT_TEST_DIR + f'pairs_overlap.csv'
    
    if os.path.exists(ROOT_TEST_DIR):
        if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (y/n) ') == 'y':
            os.system(f'rm -rf {ROOT_TEST_DIR}')
        else:
            print('Cannot continue if directory already exists. Exiting...')
            sys.exit()

    print_info(msg_before=f'Creating test results directory {ROOT_TEST_DIR}...', msg_after='Completed.') \
                (os.system)(f'mkdir -p {ROOT_TEST_DIR}')
    
    if not spark_context:
        conf = pyspark.SparkConf().setAppName('CreateIndex')    
        spark_context = print_info(msg_before='Creating Spark context...', msg_after='Completed.') \
                                    (pyspark.SparkContext)(conf=conf)

    
    ############# DATA PREPARATION #############
    extract_starting_sets_from_tables(
        input_tables_file,
        set_file,
        josie_to_sloth_id_file,
        n_tables,
        with_='infer'
    )

    create_raw_tokens(
        set_file,
        raw_tokens_file,
        spark_context=spark_context,
        single_txt=True
    )
    
    create_index(
        set_file,
        integer_set_file,
        inverted_list_file,
        spark_context
    )

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

    ############# RUNNING JOSIE #############
    josie_cmd_dir = '/home/giovanni/go/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)
    
    os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
              --pg-database={dbname} \
                --pg-table-queries={table_db_prefix}_sets \
                    --benchmark={table_db_prefix}')
    
    os.system(f'go run {josie_cmd_dir}/topk/main.go \
              --pg-database={dbname} \
                --benchmark={table_db_prefix} \
                    --output={ROOT_TEST_DIR}')
    
    ############# ANALYSING JOSIE RESULTS #############
    extract_from_josie_results_pairs_and_overlaps(josie_results_file, pairs_with_overlap_file)


if __name__ == '__main__':
    conf = pyspark.SparkConf().setAppName('CreateIndex')    
    spark_context = print_info(msg_before='Creating Spark context...', msg_after='Completed.') \
        (pyspark.SparkContext)(conf=conf)
    
    for n in [100, 1000, 10000]:        
        start = time()
        do_test(n, spark_context)
        print(f'\n\n\nTotal time for n={n}: {round(time() - start, 3)}s', end='\n\n\n')

