import os
import itertools
from time import time
from pprint import pprint

import jsonlines
import pandas as pd
from tqdm import tqdm

import chromadb

from code.utils.settings import DefaultPath
from code.fasttext.embedding_utils import TableEncoder
from code.utils.utils import rebuild_table


DEBUG = True

REMOVE_DATABASE_AT_EACH_STEP = False
KEEP_ONLY_DATABASES_WITH_n_tables = [2500, 5000]

STATISTICS_FILENAME = 'single_collection.csv'

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


add_label =         True
remove_numbers =    False
with_metadatas =    True
metadata_format = 'table:str,row:int,column:int'

n_tables_step = [100, 250, 500, 1000, 2500, 5000]#, 10000, 25000, 50000]

tabenc = TableEncoder()


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
        'metadata_format',
        'reading time (s)',
        'reading time (%)',
        'rebuilding time (s)',
        'rebuilding time (%)',
        'embedding time (s)',
        'embedding time (%)',
        'storing time (s)',
        'storing time (%)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)',
        'db-over-index size fraction'
        ]
    )


for id_test, (n_tables, add_label) in enumerate(itertools.product(n_tables_step, [False, True])):
    print(f'########## RUNNING TEST {id_test} --- n_tables {n_tables} --- add_label {add_label} ##########')
    db_name = f'v_{id_test}_{n_tables}'
    batch_size = 250
    batch_size = batch_size if batch_size < n_tables else n_tables
    chroma_client = chromadb.PersistentClient(
                        path=DefaultPath.db_path.chroma + db_name,
                        settings=chromadb.config.Settings(anonymized_telemetry=False)
                        )
    
    try: chroma_client.delete_collection('rows-columns')
    except: pass
    finally: 
        rows_columns_collection = chroma_client.create_collection(
            name='rows-columns',
            metadata={"hnsw:space": "cosine"}
        )
    
    total_reading_time, total_rebuilding_time, total_embedding_time, total_storing_time = 0, 0, 0, 0
    total_read_tables = 0
    total_processed_embeddings = 0

    start_test_time = time()
    with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
        while total_read_tables < n_tables:
            batch = []
            table_ids = []
            
            # reading tables for the current batch
            start_reading_time = time()
            j = 0
            for table in reader:
                j += 1
                batch.append(table)
                table_ids.append(table['_id'])
                if j >= batch_size:
                    break
            end_reading_time = time()
            
            if DEBUG: print(f'Reading done: len(batch)={len(batch)}')

            # rebuildings as dataframes
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

            if with_metadatas:
                metadatas = []
                final_embeddings = []
        
                for i_table in range(len(batch)):

                    metadatas.extend([
                        {
                            'table': table_ids[i_table], 
                            'is_row': True, 
                            'idx': i_row
                        } for i_row in range(embeddings[i_table][0].shape[0])])
                    
                    metadatas.extend([
                        {
                            'table': table_ids[i_table], 
                            'is_row': False, 
                            'idx': i_column
                        } for i_column in range(embeddings[i_table][1].shape[0])])
                    
                    final_embeddings.extend(
                        [emb for emb in embeddings[i_table][0].tolist() + embeddings[i_table][1].tolist()]
                    )
            ids = list(map(str, range(total_processed_embeddings, total_processed_embeddings + len(metadatas))))

            if DEBUG:
                print(f'metadatas built: len(metadatas)={len(metadatas)}')
                pprint(metadatas)
                print(f'shape final embeddings: {len(final_embeddings)}')
                print(f'len(final_embeddings[0]) = {len(final_embeddings[0])}')
                print(f'len(final_embeddings[40]) = {len(final_embeddings[40])}')                      

            assert len(metadatas) == len(final_embeddings)
            assert len(ids) == len(final_embeddings)
            assert all([len(emb) == 300 for emb in final_embeddings])

            rows_columns_collection.add(
                ids=ids,
                metadatas=metadatas,
                embeddings=final_embeddings
            )

            end_storing_time = time()

            total_read_tables += len(batch)
            total_processed_embeddings += len(final_embeddings)

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
    db_size = get_db_size(DefaultPath.db_path.chroma + db_name + os.sep + 'chroma.sqlite3')
    index_size = get_index_size(DefaultPath.db_path.chroma + db_name)

    stat.loc[len(stat)] = [
        id_test,
        tabenc.model_size,
        n_tables,
        n_tables * 100 / N_TOTAL_SLOTH_TABLES,
        n_tables * 100 / N_TOTAL_WIKITABLES,
        total_processed_embeddings,
        batch_size,
        add_label,
        remove_numbers,
        with_metadatas,
        metadata_format,
        total_reading_time,
        total_reading_time * 100/ total_time,
        total_rebuilding_time,
        total_rebuilding_time * 100 / total_time,
        total_embedding_time,
        total_embedding_time * 100 / total_time,
        total_storing_time,
        total_storing_time * 100 / total_time,
        total_time,
        db_size / (10 ** 9),
        index_size / (10 ** 9),
        db_size / index_size
    ]
    
    for c in [
        'n_tables (% sloth_tables)', 
        'n_tables (% train_tables)',
        'reading time (s)',
        'reading time (%)',
        'rebuilding time (s)',
        'rebuilding time (%)',
        'embedding time (s)',
        'embedding time (%)',
        'storing time (s)',
        'storing time (%)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)',
        'db-over-index size fraction']:
        stat[c] = stat[c].apply(lambda x: float(format(x, '.5f')))
    
    # avoid memory run out
    if REMOVE_DATABASE_AT_EACH_STEP and n_tables not in KEEP_ONLY_DATABASES_WITH_n_tables:
        os.system(f"rm -rf {DefaultPath.db_path.chroma + db_name}")
    stat.to_csv(DefaultPath.data_path.wikitables + STATISTICS_FILENAME, index=False)


