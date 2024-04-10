import os
import gc
import itertools
from time import time

import jsonlines
import pandas as pd
from tqdm import tqdm
import chromadb

from code.utils.settings import DefaultPath
from code.fasttext.embedding_utils import TableEncoder
from code.utils.utils import rebuild_table


import sys

n_tables =      int(sys.argv[1])
batch_size =    int(sys.argv[2])
add_label =     eval(sys.argv[3])
with_metadatas = eval(sys.argv[4])
remove_numbers =    False

# n_tables_step = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

DEBUG = False

CLEAN_DB_ROOT_DIRECTORY = False
REMOVE_DATABASE_AT_EACH_STEP = False
KEEP_ONLY_DATABASES_WITH_n_tables = [25000, 50000]
DB_ROOT_PATH = DefaultPath.db_path.chroma + 'double_collection_v1/'
STATISTICS_FILENAME = DefaultPath.data_path.wikitables + 'double_collection_v1.csv'

# in the train_tables.jsonl file
N_TOTAL_WIKITABLES = 570171

N_TOTAL_SLOTH_TABLES = 113098


def get_index_size(start_path):
    return sum(
        [
           os.path.getsize(start_path + os.sep + subdir + os.sep + f) 
           for subdir in os.listdir(start_path) if os.path.isdir(start_path + os.sep + subdir) 
           for f in os.listdir(start_path + os.sep + subdir)
        ]
    )


def get_db_size(db_path):
    return os.path.getsize(db_path)

if CLEAN_DB_ROOT_DIRECTORY:
    os.system(f'rm -rf {DB_ROOT_PATH}')
    os.system(f'mkdir {DB_ROOT_PATH}')


tabenc = TableEncoder()

if os.path.exists(STATISTICS_FILENAME):
    stat = pd.read_csv(
        STATISTICS_FILENAME        
        )
    id_test = len(stat)
else:
    stat = pd.DataFrame(
        columns=[
            'id_test',
            'vector_size',
            'n_tables', 
            'n_tables (% sloth_tables)', 
            'n_tables (% train_tables)',
            'total processed embeddings',
            'batch_size',
            'add_label',
            'remobe_numbers',
            'with_metadatas',
            'preprocessing time (s)',
            'preprocessing time (%)',
            'storing time (s)',
            'storing time (%)',
            'total time (s)',
            'db size (GB)',
            'index size (GB)',
            'db-over-index size fraction',
            'batch processing timestep'
            ]
    )
    id_test = 0

# for id_test, (n_tables, batch_size, add_label) in enumerate(itertools.product(n_tables_step, [500, 2000], [False, True])):
print(f'########## RUNNING TEST {id_test} --- n_tables {n_tables} --- batch_size {batch_size} --- add_label {add_label} --- with_metadatas {with_metadatas} ##########')
db_name = f'v_{id_test}_{n_tables}_add_label_{add_label}'
batch_size = batch_size if batch_size < n_tables else n_tables
chroma_client = chromadb.PersistentClient(
                    path=DB_ROOT_PATH + db_name,
                    settings=chromadb.config.Settings(anonymized_telemetry=False)
                    )

try: chroma_client.delete_collection('rows')
except: pass
finally: row_collection = chroma_client.create_collection('rows', metadata={"hnsw:space": "cosine"})

try: chroma_client.delete_collection('columns')
except: pass
finally: column_collection = chroma_client.create_collection('columns', metadata={"hnsw:space": "cosine"})

total_read_tables = 0
num_processed_embeddings = 0
total_preprocessing_time, total_storing_time = 0, 0
batch_processing_timestep = []

start_test_time = time()
with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
    while total_read_tables < n_tables:
        row_embeddings, row_ids = [], []
        column_embeddings, column_ids = [], []
        row_metadatas, column_metadatas = [] if with_metadatas else None, [] if with_metadatas else None
        
        # reading tables for the current batch
        start_preprocessing_time = time()
        num_bacth_read_tables = 0
        for table in reader:
            num_bacth_read_tables += 1
            re, ce = tabenc.full_embedding(rebuild_table(table), add_label)
            row_embeddings.extend(re.tolist())
            column_embeddings.extend(ce.tolist())
            if with_metadatas:
                row_metadatas.extend([
                    {'table_id': table['_id']}
                    for _ in range(re.shape[0])
                ])

                column_metadatas.extend([
                    {'table_id': table['_id']}
                    for _ in range(ce.shape[0])
                ])
            
            row_ids.extend([
                f"{table['_id']}#{row_id}"
                for row_id in range(re.shape[0])
            ])

            column_ids.extend([
                f"{table['_id']}#{col_id}"
                for col_id in range(ce.shape[0])
            ])
            
            print(f'read tables: {round((total_read_tables + num_bacth_read_tables) * 100 / n_tables, 3)}%         ', end='\r')
            if num_bacth_read_tables >= batch_size \
                or len(row_embeddings) + len(column_embeddings) > 30000 \
                    or num_bacth_read_tables + total_read_tables >= n_tables:
                break
        end_preprocessing_time = time()
        num_processed_embeddings += (len(row_embeddings) + len(column_embeddings))

        total_read_tables += num_bacth_read_tables

        start_storing_time = time()
        row_collection.add(
            ids=row_ids,
            metadatas=row_metadatas,
            embeddings=row_embeddings
        )
        
        column_collection.add(
            ids=column_ids,
            metadatas=column_metadatas,
            embeddings=column_embeddings
        )
        end_storing_time = time()

        total_preprocessing_time += end_preprocessing_time - start_preprocessing_time
        total_storing_time += end_storing_time - start_storing_time

        batch_time = round((end_preprocessing_time - start_preprocessing_time) + (end_storing_time - start_storing_time), 3)
        batch_processing_timestep.append(batch_time)
        mean_batch_proc_time = round(sum(batch_processing_timestep) / len(batch_processing_timestep), 3)
        print(f'{round(time() - start_test_time, 3)}s: Processed {total_read_tables} tables ({(total_read_tables * 100 / n_tables)}%), batch proc time={batch_time}s, mean batch proc time={mean_batch_proc_time}s...')


end_test_time = time()
print('Test completed.')
total_time = total_preprocessing_time + total_storing_time

db_size = get_db_size(DB_ROOT_PATH + db_name + os.sep + 'chroma.sqlite3')
index_size = get_index_size(DB_ROOT_PATH + db_name)

stat.loc[len(stat)] = [
    id_test,
    tabenc.model_size,
    n_tables,
    n_tables * 100 / N_TOTAL_SLOTH_TABLES,
    n_tables * 100 / N_TOTAL_WIKITABLES,
    num_processed_embeddings,
    batch_size,
    add_label,
    remove_numbers,
    with_metadatas,
    total_preprocessing_time,
    total_preprocessing_time * 100/ total_time,
    total_storing_time,
    total_storing_time * 100 / total_time,
    total_time,
    db_size / (10 ** 9),
    index_size / (10 ** 9),
    db_size / index_size,
    batch_processing_timestep if n_tables == 50000 else []
]

for c in stat.columns:
    try: stat[c] = stat[c].apply(lambda x: round(x, 5))
    except: pass

# avoid memory run out
if n_tables not in KEEP_ONLY_DATABASES_WITH_n_tables:
    os.system(f"rm -rf {DB_ROOT_PATH + db_name}")

stat.to_csv(STATISTICS_FILENAME, index=False)
gc.collect()

