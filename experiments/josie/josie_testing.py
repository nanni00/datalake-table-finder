import os
import sys
import json
import pprint
import shutil
import argparse
import warnings
warnings.filterwarnings('ignore')
from time import time

import pymongo
import pandas as pd

from tools.utils.settings import DefaultPath as defpath

from tools.josiestuff.db import JosieDB
from tools.josiestuff.datapreparation import (
    extract_tables_from_jsonl_to_mongodb,
    create_index,
    get_tables_statistics_from_mongodb,
    sample_queries
)


parser = argparse.ArgumentParser()
parser.add_argument('--test-name',          required=False, type=str, help='a user defined test name, used instead of the default one m<mode>')
parser.add_argument('-m', '--mode',         choices=['set', 'bag'])
parser.add_argument('-k',                   required=False, type=int, help='the K value for the top-K search of JOSIE')
parser.add_argument('-t', '--tasks',        nargs='+',      choices=['all', 'createmongodb', 'createindex', 'createrawtokens', 'samplequeries', 'dbsetup', 'josietest'])
parser.add_argument('-d', '--dbname',       required=False, help='the PostgreSQL database where will be uploaded the data used by JOSIE. It must be already running on the machine')
parser.add_argument('--sample-threshold',   required=False, type=float, help='[0,1] percentage of tables to sample from the sloth.latest_snapshot_tables collection for testing')

args = parser.parse_args()

mode =              args.mode
tasks =             args.tasks
k =                 args.k if args.k else 5
user_dbname =       args.dbname if args.dbname else 'user'
sample_threshold =  args.sample_threshold if args.sample_threshold else 0.01
test_name =         args.test_name if args.test_name else f'm{mode}'

ALL =               'all' in tasks
EXTR_TO_MONGODB =   'createmongodb' in tasks
INVERTED_IDX =      'createindex' in tasks
RAW_TOKENS =        'createrawtokens' in tasks
SAMPLE_QUERIES =    'samplequeries' in tasks
DBSETUP =           'dbsetup' in tasks
JOSIE_TEST =        'josietest' in tasks


# original JSONL and SLOTH results files
original_turl_train_tables_jsonl_file  =    defpath.data_path.wikitables + '/original_turl_train_tables.jsonl'
original_sloth_results_csv_file =           defpath.data_path.wikitables + '/original_sloth_results.csv'

# input files

# output files
ROOT_TEST_DIR =             defpath.data_path.base + f'/josie-tests/{test_name}'
subset_sloth_results_file = ROOT_TEST_DIR + '/sub_sloth-results.csv'
ids_for_queries_file =      ROOT_TEST_DIR + '/ids_for_queries.csv'
sloth_josie_ids_file =      ROOT_TEST_DIR + '/josie_sloth_ids'
raw_tokens_file =           ROOT_TEST_DIR + '/tables.raw-tokens' 
set_file =                  ROOT_TEST_DIR + '/tables.set'
integer_set_file =          ROOT_TEST_DIR + '/tables.set-2'
inverted_list_file =        ROOT_TEST_DIR + '/tables.inverted-list'
query_file =                ROOT_TEST_DIR + '/queries.json'
results_dir =               ROOT_TEST_DIR + '/results'


# statistics stuff
statistics_dir =            ROOT_TEST_DIR  + '/statistics'
tables_stat_file =          statistics_dir + '/tables.csv'
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

print('Init MongoDB client...')
mongoclient = pymongo.MongoClient()

# the DB and its collection where are stored the 570k wikitables 
# (and where are the ~45000 used for basic SLOTH testing and for next JOSIE querying)
optitab_db = mongoclient.optitab
wikitables_coll = optitab_db.turl_training_set


############# DATA PREPARATION #############

if ALL or EXTR_TO_MONGODB:
    start = time()    
    extract_tables_from_jsonl_to_mongodb(
        original_turl_train_tables_jsonl_file,
        wikitables_coll     
    )
    runtime_metrics.append(('extract-jsonl-tables', round(time() - start, 5)))

if ALL or INVERTED_IDX:    
    start = time()
    create_index(
        mode,
        original_sloth_results_csv_file,
        subset_sloth_results_file,
        ids_for_queries_file,
        sloth_josie_ids_file,
        integer_set_file,
        inverted_list_file,
        thresholds={
            'min_rows': 5,
            'min_columns': 2,
            'min_area': 50
        },
        latsnaptab_sample_threshold=sample_threshold
    )
    runtime_metrics.append(('create-invidx-intsets', round(time() - start, 5)))

    print('Creating a single mapping IDs file...')
    with open(sloth_josie_ids_file + '.csv','wb') as wfd:
        for f in [file for file in sorted(os.listdir(sloth_josie_ids_file)) if file.startswith('part-0')]:
            with open(sloth_josie_ids_file + os.sep + f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)

    shutil.rmtree(sloth_josie_ids_file)
sloth_josie_ids_file += '.csv'

############# SAMPLING TEST VALUES FOR JOSIE ##############
sampled_ids = None
if ALL or SAMPLE_QUERIES:
    if not os.path.exists(tables_stat_file):
        print('Get statistics from MongoDB wikitables...')
        start = time()
        get_tables_statistics_from_mongodb(wikitables_coll, tables_stat_file)
        runtime_metrics.append(('tables-stat-from-MongoDB', round(time() - start), 5))

    start = time()
    sampled_ids = sample_queries(
        subset_sloth_results_file,
        sloth_josie_ids_file,
        tables_stat_file,
        query_file
    )
    runtime_metrics.append(('sampling-queries', round(time() - start, 5)))

################### DATABASE OPERATIONS ####################
if ALL or DBSETUP:
    # reading the IDs for queries
    if not sampled_ids:
        with open(query_file) as fr:
            query_stuff = json.load(fr)
        sampled_ids = [_id for interval in query_stuff for _id in interval['ids']]
    
    start = time()
    josiedb = JosieDB(dbname=user_dbname, table_prefix=test_name)
    josiedb.open()
    josiedb.drop_tables()
    josiedb.create_tables()

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
    pd.DataFrame(josiedb.get_statistics()).to_csv(db_stat_file, index=False)
    josiedb.close()
    runtime_metrics.append(('josie-db-operations', round(time() - start, 5)))


############# RUNNING JOSIE #############
if ALL or JOSIE_TEST:
    GOPATH = os.environ['GOPATH']
    josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)

    start = time()
    os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                --pg-database={user_dbname} \
                --test_tag={test_name} \
                --pg-table-queries={test_name}_queries')

    os.system(f'go run {josie_cmd_dir}/topk/main.go \
                --pg-database={user_dbname} \
                --test_tag={test_name} \
                --output={results_dir} \
                    --k={k}')

    runtime_metrics.append(('josie', round(time() - start, 5)))

with open(runtime_stat_file, 'a') as rfw:
    for task in runtime_metrics:
        rfw.write(f"{task[0]},{task[1]}\n")
