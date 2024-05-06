import os
import sys
import json
from time import time
import jsonlines
import numpy as np
import pandas as pd

import faiss

from code.fasttext.embedding_utils import TableEncoder
from code.utils.settings import DefaultPath
from code.utils.utils import rebuild_table


# sizes in GB are always computed on the disk objects
stat_fields = [
    'test_id',
    'n_tables_read',
    'n_row_embeddings',
    'n_column_embeddings',
    'model name'
    'index type',
    'metric',
    'add label',
    'indexing batch size',
    'row indexing time (s)',
    'column indexing time (s)',
    'total indexing time (s)'
    'row index write time (s)',
    'column index write time (s)',
    'LUT write time (s)',
    'total write time (s)',
    'row index size (GB)',
    'column index size (GB)',
    'LUT size (GB)',
    'total size (GB)',
    'row batch timestep',
    'column batch timestep'
]

if __name__ == '__main__':
    print('Running as main.')
    test_id = 0
    n_tables_read = 50000
    add_label = True
    batch_size = 500
    to_remove = False
    JSONL_TABLE_FILE = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/sloth-tables-r5-c2-a50.jsonl'
    version = 'vCastorp'
else:
    if len(sys.argv) != 8:
        print(f'Error: usage is python {sys.argv[0]} <test_id> <n_tables_read> <add_label> <batch_size> <to_remove> <jsonl_table_file> <version>')
        sys.exit(0)
    print('Test called.')
    test_id =           sys.argv[1]
    n_tables_read =     eval(sys.argv[2])
    add_label =         eval(sys.argv[3])
    batch_size =        eval(sys.argv[4]) # #vectors passed to the index each time
    to_remove =         eval(sys.argv[5])
    JSONL_TABLE_FILE =  sys.argv[6]
    version =           sys.argv[7]


metric = 'L2'
db_name = f"{test_id}_{n_tables_read}_add_label_{add_label}_{metric}"

DB_ROOT_PATH =          DefaultPath.db_path.faiss + version + os.path.sep + f'{db_name}/'
ROW_INDEX_FILEPATH =    DB_ROOT_PATH + 'row_index.index'
COLUMN_INDEX_FILEPATH = DB_ROOT_PATH + 'column_index.index'
ROW_LUT_FILEPATH =      DB_ROOT_PATH + 'row_lut.json'
COLUMN_LUT_FILEPATH =   DB_ROOT_PATH + 'column_lut.json'

if not os.path.exists(DefaultPath.db_path.faiss + version):
    os.mkdir(DefaultPath.db_path.faiss + version)

if os.path.exists(DB_ROOT_PATH):
    print(f'Test folder {DB_ROOT_PATH} already exists!')
    sys.exit()
else:
    os.mkdir(DB_ROOT_PATH)

STATISTICS_FILENAME =   DefaultPath.data_path.wikitables + f'faiss_stat/{version}.csv'
JSON_STAT_FILEPATH =    DB_ROOT_PATH + 'info_metadata.json'

# using a better fastText version
# model_name = 'ft_cc.en.300_freqprune_400K_100K_pq_300.bin'
model_name = 'cc.en.300.compressed.bin'
tabenc = TableEncoder(
    model_path=DefaultPath.model_path.fasttext + model_name
    )

d = 300 # fastText vectors are 300D
row_index = faiss.IndexFlatL2(d)
column_index = faiss.IndexFlatL2(d)


row_LUT = [
    [], # rows steps
    [], # corresponding table IDs
]

column_LUT = [
    [], # columns steps
    [], # corresponding table IDs
]

row_indexing_time = time()
column_indexing_time = time()
row_batch_timesteps = []
column_batch_timesteps = []

n_row_embeddings, n_column_embeddings = 0, 0
total_preindex_time = 0
total_concatenate_time = 0
total_row_index_time, total_column_index_time = 0, 0
n_tables_really_read = 0

print(f"##################### RUNNING TEST {test_id} : n_tables_read {n_tables_read} : add_label {add_label} : batch_size {batch_size} : to_remove {to_remove} #####################")

start_test_time = time()

with jsonlines.open(JSONL_TABLE_FILE) as reader:
    
    while n_tables_really_read < n_tables_read:
        n_batch_row_emb = 0
        n_batch_column_emb = 0
        batch_row_emb = None
        batch_col_emb = None

        start_batch_preindex_time = time()
        for json_table in reader:
        # while (n_tables_really_read < n_tables_read) and (n_batch_row_emb < batch_size) and (n_batch_column_emb < batch_size) and (json_table := reader.read()):
            print(f"{round(time() - start_test_time)}: read tables: {n_tables_really_read} ({round(n_tables_really_read * 100 / n_tables_read, 3)}%)", end='\r')
            table = rebuild_table(json_table)
            n_tables_really_read += 1
            
            n_batch_row_emb += table.shape[0]
            n_batch_column_emb += table.shape[1]
            n_row_embeddings += table.shape[0]
            n_column_embeddings += table.shape[1]

            row_embeddings, column_embeddings = tabenc.full_embedding(table, add_label=add_label, remove_numbers=False)

            row_LUT[0].append(row_embeddings.shape[0] - 1 if row_LUT[0] == [] else row_LUT[0][-1] + row_embeddings.shape[0])
            row_LUT[1].append(json_table['_id'])

            column_LUT[0].append(column_embeddings.shape[0] - 1 if column_LUT[0] == [] else column_LUT[0][-1] + column_embeddings.shape[0])
            column_LUT[1].append(json_table['_id'])

            start_concatenate_time = time()
            batch_row_emb = row_embeddings if batch_row_emb is None else np.concatenate([batch_row_emb, row_embeddings])
            batch_col_emb = column_embeddings if batch_col_emb is None else np.concatenate([batch_col_emb, column_embeddings])
            total_concatenate_time += (time() - start_concatenate_time)

            if (n_tables_really_read >= n_tables_read - 1) or (n_batch_row_emb >= batch_size) or (n_batch_column_emb >= batch_size):
                break


        total_preindex_time += (time() - start_batch_preindex_time)
        if batch_row_emb is None or batch_col_emb is None:
            break  
        start_row_index_time = time()
        row_index.add(batch_row_emb)
        total_row_index_time += (time() - start_row_index_time)

        start_column_index_time = time()
        column_index.add(batch_col_emb)
        total_column_index_time += (time() - start_column_index_time)
        if batch_size >= 1000:
            row_batch_timesteps.append(time() - start_row_index_time)
            column_batch_timesteps.append(time() - start_column_index_time)


start_row_write_time = time()
faiss.write_index(row_index,    ROW_INDEX_FILEPATH)
row_index_write_time = time() - start_row_write_time

start_column_write_time = time()
faiss.write_index(column_index, COLUMN_INDEX_FILEPATH)
column_index_write_time = time() - start_column_write_time


with open(ROW_LUT_FILEPATH, 'w') as row_lut_writer:
    with open(COLUMN_LUT_FILEPATH, 'w') as col_lut_writer:
        start_lut_write_time = time()
        json.dump({'idxs': row_LUT[0], 'table_ids': row_LUT[1]}, row_lut_writer)
        json.dump({'idxs': column_LUT[0], 'table_ids': column_LUT[1]}, col_lut_writer)
        lut_write_time = time() - start_lut_write_time

total_test_time = time() - start_test_time

import os

stat_fields = [
    'test_id',
    'n_tables_read',
    'n_row_embeddings',
    'n_column_embeddings',
    'model name',
    'index type',
    'metric',
    'add label',
    'indexing batch size',
    'total concatenate time (s)',
    'total preindex time (s)',
    'row indexing time (s)',
    'column indexing time (s)',
    'total indexing time (s)',
    'row index write time (s)',
    'column index write time (s)',
    'LUT write time (s)',
    'total write time (s)',
    'total test time (s)',
    'row index size (GB)',
    'column index size (GB)',
    'LUT size (GB)',
    'total size (GB)',
    'row batch timestep',
    'column batch timestep'
]

stat_record = [
    test_id,                                                                        # test_id
    n_tables_really_read,                                                                  # n_tables_read
    n_row_embeddings,                                                               # n_row_embeddings
    n_column_embeddings,                                                            # n_column_embeddings
    model_name,                                                                     # module name
    'IndexFlatL2',                                                                  # index type
    metric,                                                                         # metric
    add_label,                                                                      # add label
    batch_size,                                                                     # indexing batch size
    round(total_concatenate_time, 5),
    round(total_preindex_time, 5),
    round(total_row_index_time, 5),                                                 # row indexing time (s)
    round(total_column_index_time, 5),                                              # column indexing time (s)
    round(total_row_index_time + total_column_index_time, 5),                       # total indexing time (s)
    round(row_index_write_time, 5),                                                 # row index write time (s)
    round(column_index_write_time, 5),                                              # column index write time (s)
    round(lut_write_time, 5),                                                       # LUT write time (s)
    round(row_index_write_time + column_index_write_time + lut_write_time, 5),
    round(total_test_time, 5),
    round(os.path.getsize(ROW_INDEX_FILEPATH) / (2 ** 30), 5),
    round(os.path.getsize(COLUMN_INDEX_FILEPATH) / (2 ** 30), 5),
    round(os.path.getsize(ROW_LUT_FILEPATH) / (2 ** 30), 5),
    round(
        os.path.getsize(ROW_INDEX_FILEPATH) / (2 ** 30) + \
            os.path.getsize(COLUMN_INDEX_FILEPATH) / (2 ** 30) + \
                os.path.getsize(ROW_LUT_FILEPATH) / (2 ** 30),
        5
    ),
    row_batch_timesteps,
    column_batch_timesteps
]

info_to_json = dict(zip(stat_fields, stat_record))
with open(JSON_STAT_FILEPATH) as writer:
    json.dump(info_to_json, writer)


if os.path.exists(STATISTICS_FILENAME):
    stat = pd.read_csv(STATISTICS_FILENAME)
else:
    stat = pd.DataFrame(columns=stat_fields)

stat.loc[len(stat)] = stat_record
stat.to_csv(STATISTICS_FILENAME, index=False)

if to_remove:
    os.system(f"rm -rf {DB_ROOT_PATH}")