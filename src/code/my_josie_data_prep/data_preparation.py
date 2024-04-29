import os
import sys
import psycopg
import pyspark
from code.utils.settings import DefaultPath

from code.my_josie_data_prep.data_prep_functions import extract_starting_sets_from_tables, create_index, create_raw_tokens 
from code.my_josie_data_prep.db_loading import *


if __name__ == '__main__':    
    n_tables = 500
    table_db_prefix = f'test_{n_tables}'

    input_tables_file =  DefaultPath.data_path.wikitables + 'threshold-r5-c2-a50/sloth-tables-r5-c2-a50.jsonl'
    ROOT_TEST_DIR = DefaultPath.josie_stuff_path.base + f'josie-test-{n_tables}/'
    
    if os.path.exists(ROOT_TEST_DIR):
        if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (y/n) ') == 'y':
            os.system(f'rm -rf {ROOT_TEST_DIR}')
        else:
            print('Cannot continue if directory already exists. Exiting...')
            sys.exit()
    
    raw_tokens_file =       ROOT_TEST_DIR + 'sloth_tables.raw-tokens' 
    set_file =              ROOT_TEST_DIR + 'sloth_tables.set'
    integer_set_file =      ROOT_TEST_DIR + 'sloth_tables.set-2'
    inverted_list_file =    ROOT_TEST_DIR + 'sloth_tables.inverted-list'

    print_info(msg_before=f'Creating test results directory {ROOT_TEST_DIR}...', msg_after='Completed.') \
                (os.system)(f'mkdir -p {ROOT_TEST_DIR}')
    
    conf = pyspark.SparkConf().setAppName('CreateIndex')    
    spark_context = print_info(msg_before='Creating Spark context...', msg_after='Completed.') \
                                (pyspark.SparkContext)(conf=conf)

    extract_starting_sets_from_tables(
        input_tables_file,
        set_file,
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

    with psycopg.connect("port=5442 host=/tmp dbname=giovanni user=giovanni") as conn:
        with conn.cursor() as cur:
            create_tables(cur, table_db_prefix)
            
            # insert_data_into_sets_table(cur, integer_set_file, table_db_prefix)
            # insert_data_into_inverted_list_table(cur, inverted_list_file, table_db_prefix)
            parts = [p for p in os.listdir(inverted_list_file) if p.startswith('part-')]
            for i, part in enumerate(parts):
                print(f'Inserting inverted lists part {i} of {len(parts) - 1}...', end='\r')
                insert_data_into_inverted_list_table(cur, f'{inverted_list_file}/{part}', table_db_prefix)
            print()

            parts = [p for p in os.listdir(integer_set_file) if p.startswith('part-')]
            for i, part in enumerate(parts):
                print(f'Inserting integer sets part {i} of {len(parts) - 1}...', end='\r')
                insert_data_into_sets_table(cur, f'{integer_set_file}/{part}', table_db_prefix)
            print()

            create_sets_index(cur, table_db_prefix)
            create_inverted_list_index(cur, table_db_prefix)

        conn.commit()
    
    print()
    print('Data preparation done.')
