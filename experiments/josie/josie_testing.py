import os
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore')
import subprocess
from time import time

import pymongo
import pandas as pd

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_current_time

from tools.josiestuff.db import JosieDB
from tools.josiestuff.functions import (
    extract_tables_from_jsonl_to_mongodb,
    create_index,
    get_query_ids_from_query_file,
    get_tables_statistics_from_mongodb,
    josie_test,
    sample_queries
)


parser = argparse.ArgumentParser()
parser.add_argument('--test-name', 
                    type=str, required=True, 
                    help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('-m', '--mode', 
                    required=False, default='set',
                    choices=['set', 'bag'])
parser.add_argument('-k', 
                    type=int, required=False, 
                    help='the K value for the top-K search of JOSIE', default=5)
parser.add_argument('-t', '--tasks', 
                    required=False, nargs='+', 
                    choices=['all', 'createmongodb', 'createindex', 'createrawtokens', 'samplequeries', 'dbsetup', 'josietest'], 
                    help='the tasks to do. The tables extraction from the JSONL file to MongoDB must be specified, since it isn\'t included in "all" tasks')
parser.add_argument('-d', '--dbname', 
                    required=False, default='userdb',
                    help='the PostgreSQL database where will be uploaded the data used by JOSIE. It must be already running on the machine')
parser.add_argument('-l', '--tables-limit', 
                    required=False, type=int, default=100000,
                    help='number of tables to effectivly load for processing from sloth.latest_snapshot_tables')

parser.add_argument('--jsonl-tables-file', required=False, type=str, help='the JSONL file containing the tables that will be extracted to the MongoDB')
parser.add_argument('--table-statistics-file', required = False, type=str, help='an absoluth path to an exisisting file containing table statistics used for querying')

parser.add_argument('--queries-file', required=False, type=str, help='an absolute path to an existing file containing the queries which will be used for JOSIE tests')
parser.add_argument('--convert-query-ids', required=False, action='store_true', help='when passing a queries file from a different test, the JOSIE IDs used in that test may be different from the current one, so convert them')

parser.add_argument('--use-scala', required=False, action='store_true', help='instead of use pure Python implementation of the program, use a Scala version for creating inverted index and integer sets.')
parser.add_argument('--small', required=False, action='store_true',
                    help='works on small collection versions (only for testing)')

parser.add_argument('--clean', required=False, action='store_true', help='remove PostgreSQL database tables and other big files')


args = parser.parse_args()
test_name =         args.test_name
mode =              args.mode
tasks =             args.tasks if args.tasks else []
k =                 args.k
user_dbname =       args.dbname
tables_limit =      args.tables_limit
convert_query_ids = args.convert_query_ids
use_scala =         args.use_scala
small =             args.small

ALL =               'all' in tasks
EXTR_TO_MONGODB =   'createmongodb' in tasks
INVERTED_IDX =      'createindex' in tasks
RAW_TOKENS =        'createrawtokens' in tasks
SAMPLE_QUERIES =    'samplequeries' in tasks
DBSETUP =           'dbsetup' in tasks
JOSIE_TEST =        'josietest' in tasks
CLEAN =             args.clean

# original JSONL and SLOTH results files
original_turl_train_tables_jsonl_file  =    defpath.data_path.wikitables + '/original_turl_train_tables.jsonl' \
                                                if not args.jsonl_tables_file else args.jsonl_tables_file
original_sloth_results_csv_file =           defpath.data_path.wikitables + '/original_sloth_results.csv'

# Scala JAR file used for indexing
scala_jar_indexing_path =                   defpath.root_project_path + '/tools/josiestuff/scala/indexing.jar'
java_path  =                                '/usr/lib/jvm/java-11-openjdk-amd64/bin/java'

# output files
ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
josie_sloth_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids'
raw_tokens_file =           ROOT_TEST_DIR + '/tables.raw-tokens' 
set_file =                  ROOT_TEST_DIR + '/tables.set'
integer_set_file =          ROOT_TEST_DIR + '/tables.set-2'
inverted_list_file =        ROOT_TEST_DIR + '/tables.inverted-list'
query_file =                ROOT_TEST_DIR + '/queries.json'             if not args.queries_file else args.queries_file
results_dir =               ROOT_TEST_DIR + '/results'

# statistics stuff
statistics_dir =            ROOT_TEST_DIR  + '/statistics'
tables_stat_file =          statistics_dir + '/tables.csv'              if not args.table_statistics_file else args.table_statistics_file
# columns_stat_file =         statistics_dir + '/columns.csv'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
db_stat_file =              statistics_dir + '/db.csv'

runtime_metrics = []


############# SET UP #############
if os.path.exists(ROOT_TEST_DIR):
    if input(f'Directory {ROOT_TEST_DIR} already exists: delete it to continue? (yes/no) ') in ('y', 'yes'):
        shutil.rmtree(ROOT_TEST_DIR)
        
if not os.path.exists(ROOT_TEST_DIR): 
    print(f'Creating test directory {ROOT_TEST_DIR}...')
    os.makedirs(ROOT_TEST_DIR)
    print(f'Creating test statistics directory {statistics_dir}...')
    os.makedirs(statistics_dir)
    print(f'Creating results statistics directory {results_dir}...')
    os.makedirs(results_dir)

print('Init MongoDB client...')
mongoclient = pymongo.MongoClient()

# the DB and its collection where are stored the 570k wikitables 
# (and where are the ~45000 used for basic SLOTH testing and for next JOSIE querying)
optitab_db = mongoclient.optitab
wikitables_coll = optitab_db.turl_training_set


############# DATA PREPARATION #############

if EXTR_TO_MONGODB:
    start = time()    
    extract_tables_from_jsonl_to_mongodb(
        original_turl_train_tables_jsonl_file,
        wikitables_coll     
    )
    runtime_metrics.append(('extract-jsonl-tables', round(time() - start, 5), get_current_time))

if ALL or INVERTED_IDX:    
    start = time()
    if not use_scala:
        create_index(
            mode,
            original_sloth_results_csv_file,
            josie_sloth_ids_file,
            integer_set_file,
            inverted_list_file,
            thresholds={
                'min_rows': 5,
                'min_columns': 2,
                'min_area': 50
            },
            tables_limit=tables_limit,
            small=small
        )
    else:
        print("SCALA VERSION: creating inverted list and integer sets...")
        subprocess.call(args=[java_path, "-jar", scala_jar_indexing_path, 
                             mode, 
                             original_sloth_results_csv_file, 
                             josie_sloth_ids_file, 
                             integer_set_file, 
                             inverted_list_file, 
                             tables_limit])
        print("Completed.")

    runtime_metrics.append(('create-invidx-intsets', round(time() - start, 5), get_current_time()))

    print('Creating a single mapping IDs file...')
    with open(josie_sloth_ids_file + '.csv','wb') as wfd:
        for f in [file for file in sorted(os.listdir(josie_sloth_ids_file)) if file.startswith('part-')]:
            with open(josie_sloth_ids_file + os.sep + f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)
    shutil.rmtree(josie_sloth_ids_file)

josie_sloth_ids_file += '.csv'

############# SAMPLING TEST VALUES FOR JOSIE ##############
if ALL or SAMPLE_QUERIES:
    # if not use_scala and not os.path.exists(tables_stat_file):
    #     print('Get statistics from MongoDB wikitables...')
    #     start = time()
    #     get_tables_statistics_from_mongodb(wikitables_coll, tables_stat_file)
    #     runtime_metrics.append(('tables-stat-from-MongoDB', round(time() - start, 5), get_current_time()))

    start = time()
    sample_queries(
        josie_sloth_ids_file,
        ids_for_queries_file,
        query_file
    )
    runtime_metrics.append(('sampling-queries', round(time() - start, 5), get_current_time()))

################### DATABASE OPERATIONS ####################
if ALL or DBSETUP:
    # reading the IDs for queries
    sampled_ids = get_query_ids_from_query_file(josie_sloth_ids_file, query_file, convert_query_ids)
        
    print(f'Total sampled IDs: {len(sampled_ids)}')
    
    start = time()
    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()
    josiedb.drop_tables()
    josiedb.create_tables()
    josiedb.insert_data_into_inverted_list_table(inverted_list_file)
    josiedb.insert_data_into_sets_table(integer_set_file)
    josiedb.insert_data_into_query_table(sampled_ids)
    josiedb.create_sets_index()
    josiedb.create_inverted_list_index()
    # database statistics
    dbstat = josiedb.get_statistics()
    dbstat[0]['test-name'] = test_name
    pd.DataFrame(dbstat).to_csv(db_stat_file, index=False)
    josiedb.close()
    runtime_metrics.append(('josie-db-operations', round(time() - start, 5), get_current_time()))

################## RUNNING JOSIE ##################
if ALL or JOSIE_TEST:
    start = time()
    josie_test(josie_dbname=user_dbname, test_name=test_name, results_directory=results_dir, k=k)
    runtime_metrics.append(('josie-test', round(time() - start, 5), get_current_time()))

with open(runtime_stat_file, 'a') as rfw:
    for task in runtime_metrics:
        rfw.write(f"{task[0]},{task[1]},{task[2]}\n")

if CLEAN:
    if os.path.exists(integer_set_file):
        shutil.rmtree(integer_set_file)
    if os.path.exists(inverted_list_file):
        shutil.rmtree(inverted_list_file)

    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()
    josiedb.drop_tables(all=True)
    josiedb.close()

print('All tasks have been completed.')


