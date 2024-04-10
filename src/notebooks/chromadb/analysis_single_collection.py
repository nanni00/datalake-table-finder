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


DEBUG = False

REMOVE_DATABASE_AT_EACH_STEP = False
KEEP_ONLY_DATABASES_WITH_n_tables = [100, 2500, 5000, 10000]

STATISTICS_FILENAME = 'single_collection.csv'

# in the train_tables.jsonl file
N_TOTAL_WIKITABLES = 570171

N_TOTAL_SLOTH_TABLES = 113098

DB_ROOT_PATH = DefaultPath.db_path.chroma + 'single_collection/'

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



os.system(f'rm -rf {DB_ROOT_PATH}')
os.system(f'mkdir  {DB_ROOT_PATH}')


add_label =         True
remove_numbers =    False
with_metadatas =    True
metadata_format = 'table:str,row:int,column:int'

n_tables_step = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

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
        'embedding time (s)',
        'embedding time (%)',
        'pre-storing time (s)',
        'pre-storing time (%)',
        'storing time (s)',
        'storing time (%)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)',
        'db-over-index size fraction',
        'batch_proc_timestep'
        ]
    )


for id_test, (n_tables, add_label) in enumerate(itertools.product(n_tables_step[-2:], [False, True])):
    print(f'########## RUNNING TEST {id_test} --- n_tables {n_tables} --- add_label {add_label} ##########')
    db_name = f'v_{id_test}_{n_tables}_add_label_{add_label}'
    batch_size = 500
    batch_size = batch_size if batch_size < n_tables else n_tables
    chroma_client = chromadb.PersistentClient(
                        path=DB_ROOT_PATH + db_name,
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

    batch_processing_time = []
    start_test_time = time()
    batch, table_ids, embedding_ids, metadatas, embeddings, final_embeddings = [], [], [], [], [], []

    with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
        while total_read_tables < n_tables:
            
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

            batch_actual_size = len(batch)
            
            if DEBUG: print(f'Reading done: len(batch)={len(batch)}')

            # rebuildings as dataframes
            # start_rebuilding_time = time()
            batch = map(lambda table_json: rebuild_table(table_json), batch)
            # end_rebuilding_time = time()
            
            # embedding the dataframes --> 3-dim list: <n_tables_in_batch, <row_embeddings, column_embeddings>>
            start_embedding_time = time()
            embeddings = \
                list(
                    map(
                        lambda df: tabenc.full_embedding(df, add_label, remove_numbers),
                        batch
                    )
                )
            end_embedding_time = time()
            
            start_pre_storing_time = time()
            for i_table in range(batch_actual_size):
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
            embedding_ids = list(map(str, range(total_processed_embeddings, total_processed_embeddings + len(metadatas))))
            end_pre_storing_time = time()

            # storing
            start_storing_time = time()
            rows_columns_collection.add(
                ids=embedding_ids,
                metadatas=metadatas,
                embeddings=final_embeddings
            )
            end_storing_time = time()

            rows_columns_collection.update()

            total_read_tables += batch_actual_size
            total_processed_embeddings += len(final_embeddings)
            batch_processing_time.append(round(time() - start_reading_time, 2))
            mean_batch_proc_time = round(sum(batch_processing_time) / len(batch_processing_time), 2)
            
            print(f'Processed {total_read_tables} tables ({(total_read_tables * 100 / n_tables)}%), batch proc time={batch_processing_time[-1]}s, mean batch proc time={mean_batch_proc_time}s ...', end='\r')

            batch, embedding_ids, metadatas, embeddings, final_embeddings = [], [], [], [], []

            total_reading_time += (end_reading_time - start_reading_time)
            #total_rebuilding_time += (end_rebuilding_time - start_rebuilding_time)
            total_embedding_time += (end_embedding_time - start_embedding_time)
            total_pre_storing_time += (end_pre_storing_time - start_pre_storing_time)
            total_storing_time += (end_storing_time - start_storing_time)
    
    end_test_time = time()
    total_time = total_reading_time + total_embedding_time + total_pre_storing_time + total_storing_time
    print(f'Test completed.')
    print(f'\tTotal time:\t{end_test_time - start_test_time};')
    print(f'\texpected:\t{total_time}')
    print(f'\tdelta:\t\t{(end_test_time - start_test_time) - total_time}')

    db_size = get_db_size(DB_ROOT_PATH + db_name + os.sep + 'chroma.sqlite3')
    index_size = get_index_size(DB_ROOT_PATH + db_name)

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
        # total_rebuilding_time,
        # total_rebuilding_time * 100 / total_time,
        total_embedding_time,
        total_embedding_time * 100 / total_time,
        total_storing_time,
        total_storing_time * 100 / total_time,
        total_time,
        db_size / (10 ** 9),
        index_size / (10 ** 9),
        db_size / index_size,
        batch_processing_time
    ]
    
    for c in [
        'n_tables (% sloth_tables)', 
        'n_tables (% train_tables)',
        'reading time (s)',
        'reading time (%)',
        # 'rebuilding time (s)',
        # 'rebuilding time (%)',
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
        os.system(f"rm -rf {DB_ROOT_PATH + db_name}")
    stat.to_csv(DefaultPath.data_path.wikitables + STATISTICS_FILENAME, index=False)


