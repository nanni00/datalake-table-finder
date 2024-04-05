import os
from time import time

import jsonlines
import pandas as pd
from tqdm import tqdm
import chromadb

from code.utils.settings import DefaultPath
from code.fasttext.embedding_utils import TableEncoder
from code.utils.utils import rebuild_table


DEBUG = False

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

os.system(f'rm -rf {DefaultPath.db_path.chroma}')
os.system(f'mkdir {DefaultPath.db_path.chroma}')

add_label, remove_numbers = False, False
with_metadatas = False

n_tables_step = [100, 250, 500, 1000, 2500] #, 5000, 10000, 25000, 50000]
tests = list(range(len(n_tables_step)))

tabenc = TableEncoder()


stat = pd.DataFrame(
    columns=[
        'id_test', 
        'n_tables', 
        'n_tables (% sloth_tables)', 
        'n_tables (% train_tables)',
        'batch_size',
        'add_label',
        'remobe_numbers',
        'with_metadatas',
        'reading time (s)',
        'building table time (s)',
        'embedding time (s)',
        'storing time (s)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)'
        ]
    )

for id_test, n_tables in enumerate(n_tables_step):
    print(f'########## RUNNING TEST {id_test} --- n_tables {n_tables} ##########')
    batch_size = 500
    batch_size = batch_size if batch_size < n_tables else n_tables
    chroma_client = chromadb.PersistentClient(
                        path=DefaultPath.db_path.chroma + f'v_{n_tables}',
                        settings=chromadb.config.Settings(anonymized_telemetry=False)
                        )
    
    try: chroma_client.delete_collection('rows')
    except: pass
    finally: row_collection = chroma_client.create_collection('rows')
    
    try: chroma_client.delete_collection('columns')
    except: pass
    finally: column_collection = chroma_client.create_collection('columns')

    total_reading_time, total_rebuilding_time, total_embedding_time, total_storing_time = 0, 0, 0, 0
    total_read_tables = 0

    start_test_time = time()
    with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
        while total_read_tables < n_tables:
            batch = []
            
            # reading tables for the current batch
            start_reading_time = time()
            j = 0
            for table in reader:
                j += 1
                batch.append(table)
                if j >= batch_size:
                    break
            end_reading_time = time()
            
            if DEBUG: print(f'Reading done: len(batch)={len(batch)}')

            total_read_tables += len(batch)

            # rebuilding tables as dataframes
            start_rebuilding_time = time()
            batch_dataframes = list(
                map(
                    lambda table_json: rebuild_table(table_json),
                    batch
                )
            )
            end_rebuilding_time = time()
            if DEBUG:
                print(f'Rebuilding batch done: len(batch_dataframes)={len(batch_dataframes)}')
                print(f'Shape of dataframes: {[df.shape for df in batch_dataframes]}')

            # embedding the dataframes --> 3-dim list: <n_tables_in_batch, <row_embeddings, column_embeddings>>
            start_embedding_time = time()
            embeddings = \
                list(
                    map(
                        lambda df: tabenc.full_embedding(df, add_label, remove_numbers),
                        batch_dataframes
                    )
                )
            end_embedding_time = time()
            if DEBUG:
                print(f'Embedding done: len(embeddings)={len(embeddings)}')
                print(f'Shape of (row, col) embeddings: {[(emb[0].shape, emb[1].shape) for emb in embeddings]}')

            # storing
            start_storing_time = time()
            row_emb = [table_emb[0] for table_emb in embeddings]
            col_emb = [table_emb[1] for table_emb in embeddings]
            if DEBUG:
                print(f'len(row_emb)={len(row_emb)}')
                print(f'len(col_emb)={len(col_emb)}')

            metadatas = None #if not with_metadatas else [[] for i in range(len(embeddings))]

            ids = \
                [
                    [
                        [
                            f"{batch[batch_table_idx]['_id']}#{row_id}" 
                            for row_id in range(len(embeddings[batch_table_idx][0]))
                        ],
                        [
                            f"{batch[batch_table_idx]['_id']}#{column_id}" 
                            for column_id in range(len(embeddings[batch_table_idx][1]))
                        ]
                    ]
                    for batch_table_idx in range(len(batch))
                ]
            if DEBUG:
                print(f'ids built: len(ids)={len(ids)}')
                print(f'shape of ids: {[(len(id[0]), len(id[1])) for id in ids]}')
                print(f'len row id = {len([row_id for id in ids for row_id in id[0]])}')
                print(f'len col id = {len([col_id for id in ids for col_id in id[1]])}')
                print(f'len row embeddings = {len([row_emb for embedding in embeddings for row_emb in embedding[0]])}')
                print(f'len column embeddings = {len([row_emb for embedding in embeddings for row_emb in embedding[1]])}')

            row_collection.add(
                ids=[row_id for id in ids for row_id in id[0]],
                metadatas=metadatas,
                embeddings=[row_emb.tolist() for embedding in embeddings for row_emb in embedding[0]]
            )
            
            column_collection.add(
                ids=[col_id for id in ids for col_id in id[1]],
                metadatas=metadatas,
                embeddings=[col_emb.tolist() for embedding in embeddings for col_emb in embedding[1]]
            )

            end_storing_time = time()

            total_reading_time += end_reading_time - start_reading_time
            total_rebuilding_time += end_rebuilding_time - start_rebuilding_time
            total_embedding_time += end_embedding_time - start_embedding_time
            total_storing_time += end_storing_time - start_storing_time
    
    end_test_time = time()
    total_time = total_reading_time + total_rebuilding_time + total_embedding_time + total_storing_time
    print(f'Test completed.')
    print(f'\tTotal time:\t{end_test_time - start_test_time};')
    print(f'\texpected:\t{total_time}')
    print(f'\tdelta:\t\t{(end_test_time - start_test_time) - total_time}')

    stat.loc[len(stat)] = [
        id_test,
        n_tables,
        n_tables * 100 / N_TOTAL_SLOTH_TABLES,
        n_tables * 100 / N_TOTAL_WIKITABLES,
        batch_size,
        add_label,
        remove_numbers,
        with_metadatas,
        total_reading_time,
        total_rebuilding_time,
        total_embedding_time,
        total_storing_time,
        total_time,
        get_db_size(DefaultPath.db_path.chroma + f'v_{n_tables}' + os.sep + 'chroma.sqlite3') / (10 ** 9),
        get_index_size(DefaultPath.db_path.chroma + f'v_{n_tables}') / (10 ** 9)
    ]
    
    for c in [
        'n_tables (% sloth_tables)', 
        'n_tables (% train_tables)',
        'reading time (s)',
        'building table time (s)',
        'embedding time (s)',
        'storing time (s)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)']:
        stat[c] = stat[c].apply(lambda x: float(format(x, '.5f')))
    
    # avoid memory run out
    os.system(f'rm -rf {DefaultPath.db_path.chroma + f'v_{n_tables}'}')
    stat.to_csv(DefaultPath.data_path.wikitables + 'global_stat.csv', index=False)
