import os
import sys
import pyspark
from code.utils.settings import DefaultPath

from code.my_josie_data_prep.create_initial_sets import extract_starting_sets_from_tables
from code.my_josie_data_prep.create_index import create_index
from code.my_josie_data_prep.create_raw_tokens import create_raw_tokens


if __name__ == '__main__':
    input_tables_file =  DefaultPath.data_path.wikitables + 'threshold-r5-c2-a50/sloth-tables-r5-c2-a50.jsonl'
    
    ROOT_TEST_DIR = DefaultPath.data_path.wikitables + 'threshold-r5-c2-a50/josie-test/'
    if os.path.exists(ROOT_TEST_DIR):
        if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (y/n) ') == 'y':
            os.system(f'rm -rf {ROOT_TEST_DIR}')
        else:
            print('Cannot continue if directory already exists. Exiting...')
            sys.exit()

    raw_tokens_file =       ROOT_TEST_DIR + 'sloth_tables.raw-tokens' 
    set_file =              ROOT_TEST_DIR + 'sloth_tables_n10.set'
    integer_set_file =      ROOT_TEST_DIR + 'sloth_tables_n10.set-2'
    inverted_list_file =    ROOT_TEST_DIR + 'sloth_tables_n10.inverted-list'

    n_tables = 5

    print(f'Creating test results directory {ROOT_TEST_DIR}...')
    os.system(f'mkdir -p {ROOT_TEST_DIR}')

    print(f'Creating Spark context...')
    conf = pyspark.SparkConf() \
            .setAppName('CreateIndex')                
    spark_context = pyspark.SparkContext(conf=conf)

    print(f'Extracting sets from tables at {input_tables_file}...')
    extract_starting_sets_from_tables(
        input_tables_file,
        set_file,
        n_tables,
    )
    print('Completed.')

    print('Creating raw tokens...')
    create_raw_tokens(
        set_file,
        raw_tokens_file,
        spark_context=spark_context
    )
    print('Completed.')

    print('Creating inverted index...')
    create_index(
        set_file,
        integer_set_file,
        inverted_list_file,
        spark_context
    )
    print('Completed.')
