import os
from pprint import pprint
import sys
import warnings

import pymongo

warnings.filterwarnings('ignore')
from time import time
from collections import defaultdict

import pandas as pd

from tools.utils.settings import DefaultPath as defpath

from tools.josiedataprep.db import *
from tools.josiedataprep.preparation_functions import (
    extract_tables_from_jsonl_to_mongodb,
    format_spark_set_file,
    parallel_extract_starting_sets_from_tables,
    extract_starting_sets_from_tables, 
    create_index, 
    create_raw_tokens,
    sample_query_sets
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', choices=['set', 'bag'])
parser.add_argument('-k', type=int, required=False, help='the K value for the top-K search of JOSIE')
parser.add_argument('-t', '--tasks', nargs='*', choices=['all', 'createindex', 'createrawtokens', 'dbsetup', 'josietest'])
parser.add_argument('-d', '--dbname', help='the database where will be uploaded the data used by JOSIE. It must be already running on the machine')
parser.add_argument('-Q', '--queryfile', help='path to the file containing the set IDs that will be used as queries for JOSIE')
parser.add_argument('--statistics', action='store_true', help='collect statistics from data (number of NaNs per column/table,...) and runtime metrics')


args = parser.parse_args()

mode = args.mode
tasks = args.tasks
k = args.k
user_dbname = args.dbname
query_file = args.queryfile

ALL =               'all' in tasks
EXTR_TO_MONGODB =   'createmongodb' in tasks
INVERTED_IDX =      'createindex' in tasks
RAW_TOKENS =        'createrawtokens' in tasks
DBSETUP =           'dbsetup' in tasks
JOSIE_TEST =        'josietest' in tasks

COLLECT_STAT =      parser.statistics

use_scala_jar = False


# SPECIFICARE CONFIGURAZIONE DEL TEST
# k = 5
# mode = 'set'
# user_dbname = 'nanni'
# n_tables = 45673    # questo è un parametro ancora poco flessibile, se è un numero diverso
                    # da quello delle tabelle presenti nella cartella csv va tutto in pappa
                    # si può ottenere ad es con "ls /path/to/csvdirectory | wc -l"

# non dovrei usare questa roba qua, quando si lancia lo script i dati dovrebbero essere già
# pronti per venire estratti e usati
# original_tables_jsonl_file = defpath.data_path.wikitables + '/original_turl_train_tables.jsonl'
# original_sloth_results_file = defpath.data_path.wikitables + '/original_sloth_results.csv'


test_tag = f'm{mode}'

# original JSONL and SLOTH results files
original_turl_train_tables_jsonl_file  =    defpath.data_path.wikitables + '/original_turl_train_tables.jsonl'
original_sloth_results_csv_file =           defpath.data_path.wikitables + '/original_sloth_results.csv'

# input files
tables_subset_directory =   defpath.data_path.wikitables + '/tables-subset'
intput_tables_csv_dir =     tables_subset_directory + '/csv'
input_sloth_res_csv_file =  tables_subset_directory + '/sloth-results-r5-c2-a50.csv'

# output files
ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_tag}'
sloth_josie_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids.csv'
raw_tokens_file =           ROOT_TEST_DIR + '/tables.raw-tokens' 
set_file =                  ROOT_TEST_DIR + '/tables.set'
integer_set_file =          ROOT_TEST_DIR + '/tables.set-2'
inverted_list_file =        ROOT_TEST_DIR + '/tables.inverted-list'

# statistics stuff
statistics_dir =            ROOT_TEST_DIR  + '/statistics'
tables_stat_file =          statistics_dir + '/tables.csv'
# columns_stat_file =         statistics_dir + '/columns.csv'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
db_stat_file =              statistics_dir + '/db.csv'

runtime_metrics = defaultdict(float)
    

############# SET UP #############
if os.path.exists(ROOT_TEST_DIR):
    if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (yes/no) ') in ('y', 'yes'):
        os.system(f'rm -rf {ROOT_TEST_DIR}')
        
if not os.path.exists(ROOT_TEST_DIR): 
        print(f'Creating test directory {ROOT_TEST_DIR}...')
        os.system(f'mkdir -p {ROOT_TEST_DIR}')

        print(f'Creating test statistics directory {statistics_dir}...')
        os.system(f'mkdir -p {statistics_dir}')

mongoclient = pymongo.MongoClient()
mongodb = mongoclient.optitab
mongo_table_collection = mongodb.wikitables


############# DATA PREPARATION #############
if ALL or EXTR_TO_MONGODB:
    start = time()    
    extract_tables_from_jsonl_to_mongodb(
        original_turl_train_tables_jsonl_file,
        original_sloth_results_csv_file,
        mongo_table_collection,
        input_sloth_res_csv_file,
        statistics_file=tables_stat_file if COLLECT_STAT else None,
        thresholds={
           'min_rows': 5,
           'min_columns': 2,
           'min_area': 50
        }        
    )
    runtime_metrics['extract_jsonl_tables'] = round(time() - start, 5)

############# SETS EXTRACTION #############
# start = time()
# parallel_extract_starting_sets_from_tables(
#     intput_tables_csv_dir,
#     set_file,
#     sloth_josie_ids_file,
#     tables_stat_file,
#     columns_stat_file,
#     n_tables,
#     mode
# )
# runtime_metrics['extract_sets'] = round(time() - start, 5)

############# CREATE JOSIE TABLES #############
if use_scala_jar:
    create_raw_tokens_jar_path = defpath.data_path.base + '/josie-tests/create_raw_tokens.jar'
    create_inverted_list_jar_path = defpath.data_path.base + '/josie-tests/indexing.jar'
    print('Start creating raw tokens (JAR mode)...')
    start = time()
    os.system(f"/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -Dfile.encoding=UTF-8 -jar {create_raw_tokens_jar_path}      file://{set_file} file://{raw_tokens_file}")
    runtime_metrics['create_raw_tokens'] = round(time() - start, 5)
    
    print('Start creating integer sets and inverted index (JAR mode)...')
    start = time()
    os.system(f"/usr/lib/jvm/java-1.8.0-openjdk-amd64/bin/java -Dfile.encoding=UTF-8 -jar {create_inverted_list_jar_path}   file://{set_file} file://{integer_set_file} file://{inverted_list_file}")
    runtime_metrics['create_index'] = round(time() - start, 5)

    print('Formatting integer sets file...')
    start = time()
    format_spark_set_file(integer_set_file, os.path.dirname(integer_set_file) + '/tables-formatted.set-2', on_inverted_index=False)

    print('Formatting inverted list file...')
    format_spark_set_file(inverted_list_file, os.path.dirname(inverted_list_file) + '/tables-formatted.inverted-list', on_inverted_index=True)
    integer_set_file = os.path.dirname(integer_set_file) + '/tables-formatted.set-2'
    inverted_list_file = os.path.dirname(inverted_list_file) + '/tables-formatted.inverted-list'
    runtime_metrics['formatting'] = round(time() - start, 5)

if (ALL or INVERTED_IDX) and not use_scala_jar:
    # start = time()
    # create_raw_tokens(
    #     set_file,
    #     raw_tokens_file
    # )
    # runtime_metrics['create_raw_tokens'] = round(time() - start, 5)
    
    start = time()
    create_index(
        mode, # set_file,
        sloth_josie_ids_file,
        integer_set_file,
        inverted_list_file
    )
    runtime_metrics['create_index'] = round(time() - start, 5)


############# SAMPLING TEST VALUES FOR JOSIE ##############
samples, sampled_ids = sample_query_sets(
    input_sloth_res_csv_file,
    sloth_josie_ids_file,
    intervals=[2, 17, 34, 59, 92, 145, 213, 361, 817, 1778, 3036],
    num_sample_per_interval=10
)

################### DATABASE OPERATIONS ####################
if ALL or DBSETUP:
    start = time()
    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_tag)
    josiedb.open()
    josiedb.drop_tables()
    josiedb.create_tables()
    # if use_scala_jar:
    #     josiedb.insert_data_into_inverted_list_table(inverted_list_file)
    #     josiedb.insert_data_into_sets_table(integer_set_file)
    # else:
    parts = [p for p in sorted(os.listdir(inverted_list_file)) if p.startswith('part-')]
    for part in parts:
        josiedb.insert_data_into_inverted_list_table(f'{inverted_list_file}/{part}')
    parts = [p for p in sorted(os.listdir(integer_set_file)) if p.startswith('part-')]
    for part in parts:
        josiedb.insert_data_into_sets_table(f'{integer_set_file}/{part}')
    josiedb.insert_data_into_query_table(sampled_ids)
    josiedb.create_sets_index()
    josiedb.create_inverted_list_index()

############# GETTING DATABASE STATISTICS #############
    if COLLECT_STAT:
        pd.DataFrame(josiedb.get_statistics()).to_csv(db_stat_file, index=False)
    josiedb.close()
    runtime_metrics['db_operations'] = round(time() - start, 5)


############# RUNNING JOSIE #############
if ALL or JOSIE_TEST:
    GOPATH = os.environ['GOPATH']
    josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)

    start = time()
    os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                --pg-database={user_dbname} \
                --test_tag={test_tag} \
                --pg-table-queries={test_tag}_queries')

    os.system(f'go run {josie_cmd_dir}/topk/main.go \
                --pg-database={user_dbname} \
                --test_tag={test_tag} \
                --output={ROOT_TEST_DIR} \
                    --k={k}')

    runtime_metrics['josie'] = round(time() - start, 5)
    runtime_metrics['total_time'] = sum(runtime_metrics.values())

pd.DataFrame([runtime_metrics]).to_csv(runtime_stat_file, index=False)
