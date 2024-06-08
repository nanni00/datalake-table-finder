import os
import shutil
import argparse
from time import time

import pandas as pd

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_current_time

from tools.josiestuff.db import JosieDB
from tools.josiestuff.functions import (
    create_index,
    get_query_ids_from_query_file,
    josie_test,
    sample_queries
)

# TODO ok argparse and CLI, maybe better a file .py with variables and import them? 
# TODO integrate LSHForest testing in main tester 
parser = argparse.ArgumentParser()
parser.add_argument('--test-name', 
                    type=str, required=True)
parser.add_argument('-m', '--mode', 
                    required=False, default='set',
                    choices=['set', 'bag'])
parser.add_argument('-k', 
                    type=int, required=False, default=5,
                    help='the K value for the top-K search of JOSIE')
parser.add_argument('-t', '--tasks', 
                    required=False, nargs='+',
                    choices=['all', 
                             'j-createindex', 'f-createforest',
                             'samplequeries', 
                             'j-dbsetup', 
                             'j-updatequeries', 
                             'j-query', 'f-query'
                             ], 
                    help='the tasks to do. \
                        The prefix "j-" refers to JOSIE tasks, the "f-" to the LSH Forest ones.')
parser.add_argument('-d', '--dbname', 
                    required=False, default='userdb',
                    help='the PostgreSQL database where will be uploaded the data used by JOSIE. It must be already running on the machine')
parser.add_argument('--num-table-sampled', 
                    type=int, required=False, default=100,
                    help='the number of tables that will be sampled from the collections and that will be used as query id for JOSIE (the actual number) \
                        may be less than the specified one due to thresholds tables parameter')
parser.add_argument('--forest-index', 
                    required=False, type=str, 
                    help='the location of the LSH Forest index file that will be used for querying. If it does not exist, \
                        a new index will be created at that location.')
parser.add_argument('--query-file', 
                    required=False, type=str, 
                    help='an absolute path to an existing file containing the queries which will be used for JOSIE tests')
parser.add_argument('--small', 
                    required=False, action='store_true',
                    help='works on small collection versions (only for testing)')
parser.add_argument('--clean', 
                    required=False, action='store_true', 
                    help='remove PostgreSQL database tables and other big files')


args = parser.parse_args()
test_name =         args.test_name
mode =              args.mode
tasks =             args.tasks if args.tasks else []
k =                 args.k
user_dbname =       args.dbname
small =             args.small
nsamples =          args.num_table_sampled

# TODO set thresholds as a CLI parameter or somethig else?
tables_thresholds = {
    'min_rows':     0,
    'min_columns':  0,
    'min_area':     0,
    'max_rows':     999999,
    'max_columns':  999999,
    'max_area':     999999,
}

ALL =                   'all' in tasks
J_CREATE_INVIDX =       'j-createindex' in tasks
F_CREATE_FOREST =       'f-createforest' in tasks
SAMPLE_QUERIES =        'samplequeries' in tasks
J_DBSETUP =             'j-dbsetup' in tasks
UPDATE_QUERY_TABLE =    'j-updatequeries' in tasks
JOSIE_TEST =            'j-query' in tasks
LSHF_QUERY =            'f-query' in tasks

any_task = bool(args.tasks)

CLEAN =             args.clean


# output files
ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
set_file =                  ROOT_TEST_DIR + '/tables.set'
integer_set_file =          ROOT_TEST_DIR + '/tables.set-2'
inverted_list_file =        ROOT_TEST_DIR + '/tables.inverted-list'
query_file =                ROOT_TEST_DIR + '/query.json'             if not args.query_file else args.query_file
results_dir =               ROOT_TEST_DIR + '/results'

# statistics stuff
statistics_dir =            ROOT_TEST_DIR  + '/statistics'
runtime_stat_file =         statistics_dir + '/runtime.csv'     
db_stat_file =              statistics_dir + '/db.csv'

runtime_metrics = []


############# SET UP #############
if any_task:
    if os.path.exists(ROOT_TEST_DIR):
        if input(f'Directory {ROOT_TEST_DIR} already exists: delete it (old data will be lost)? (yes/no) ') in ('y', 'yes'):
            shutil.rmtree(ROOT_TEST_DIR)

    if not os.path.exists(ROOT_TEST_DIR): 
        print(f'Creating test directory {ROOT_TEST_DIR}...')
        os.makedirs(ROOT_TEST_DIR)
        print(f'Creating test statistics directory {statistics_dir}...')
        os.makedirs(statistics_dir)
        print(f'Creating results statistics directory {results_dir}...')
        os.makedirs(results_dir)

############# DATA PREPARATION #############
if ALL or J_CREATE_INVIDX:    
    start = time()
    
    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()
    josiedb.create_tables()
    josiedb.close()

    create_index(mode, tables_thresholds, small, test_name)

    runtime_metrics.append(('create-invidx-intsets', round(time() - start, 5), get_current_time()))

############# SAMPLING TEST VALUES FOR JOSIE ##############
if ALL or SAMPLE_QUERIES:
    start = time()
    sample_queries(query_file, nsamples, small, tables_thresholds)
    runtime_metrics.append(('sampling-queries', round(time() - start, 5), get_current_time()))

################### INSERTING QUERIES INTO POSTGRESQL DATABASE ####################
if ALL or J_DBSETUP or UPDATE_QUERY_TABLE:
    # reading the IDs for queries
    sampled_ids = get_query_ids_from_query_file(query_file)
        
    print(f'Total sampled IDs: {len(sampled_ids)}')
    
    start = time()
    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()

    if ALL or J_DBSETUP:
        josiedb.insert_data_into_query_table(sampled_ids)
        josiedb.create_sets_index()
        josiedb.create_inverted_list_index()
    elif UPDATE_QUERY_TABLE:
        josiedb.clear_query_table()
        josiedb.insert_data_into_query_table(sampled_ids)

    # database statistics
    pd.DataFrame(josiedb.get_statistics()).to_csv(db_stat_file, index=False)
    josiedb.close()
    
    runtime_metrics.append(('josie-db-operations', round(time() - start, 5), get_current_time()))

################## RUNNING JOSIE ##################
if ALL or JOSIE_TEST:
    start = time()
    josie_test(josie_dbname=user_dbname, test_name=test_name, results_directory=results_dir, k=k)
    runtime_metrics.append(('josie-test', round(time() - start, 5), get_current_time()))

if any_task:
    with open(runtime_stat_file, 'a') as rfw:
        for task in runtime_metrics:
            rfw.write(f"{task[0]},{task[1]},{task[2]}\n")
    print('All tasks have been completed.')


if CLEAN:
    print('Cleaning directories and database...')
    if os.path.exists(integer_set_file):
        shutil.rmtree(integer_set_file)
    if os.path.exists(inverted_list_file):
        shutil.rmtree(inverted_list_file)

    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()
    josiedb.drop_tables(all=True)
    josiedb.close()
    print('Cleaning completed.')


