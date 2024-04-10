import os
import gc
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

REMOVE_DATABASE_AT_EACH_STEP = True
KEEP_ONLY_DATABASES_WITH_n_tables = [25000, 50000]

STATISTICS_FILENAME = 'single_collection_v2_3.csv'

# in the train_tables.jsonl file
N_TOTAL_WIKITABLES = 570171

N_TOTAL_SLOTH_TABLES = 113098

DB_ROOT_PATH = DefaultPath.db_path.chroma + 'single_collection_v2/'

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



# os.system(f'rm -rf {DB_ROOT_PATH}')
# os.system(f'mkdir  {DB_ROOT_PATH}')


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
        'reading-embedding time (s)',
        'reading-embedding time (%)',
        'storing time (s)',
        'storing time (%)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)',
        'db-over-index size fraction',
        'batch_proc_timestep'
        ]
    )


for id_test, (n_tables, batch_size, add_label) in enumerate(itertools.product(n_tables_step[-1:], [5000], [False, True])):
    #if id_test < 15: continue
    print(f'########## RUNNING TEST {id_test} --- n_tables {n_tables} --- batch {batch_size} --- add_label {add_label} ##########')
    db_name = f'v_{id_test}_{n_tables}_add_label_{add_label}'
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
    
    total_reading_embedding_time, total_storing_time = 0, 0
    
    total_read_tables = 0
    total_processed_embeddings = 0

    batch_processing_time = []
    start_test_time = time()

    with jsonlines.open(DefaultPath.data_path.wikitables + 'sloth_tables.jsonl') as reader:
        while total_read_tables < n_tables:
            embeddings, metadatas, embedding_ids = [], [], []

            # reading tables for the current batch
            start_reading_embedding_time = time()
            read_tables = 0
            for table in reader:
                read_tables += 1
                row_emb, col_emb = tabenc.full_embedding(rebuild_table(table))
                embeddings.extend(row_emb.tolist())
                embeddings.extend(col_emb.tolist())
                print(f'read tables: {round((total_read_tables + read_tables) * 100 / n_tables, 3)}%', end='\r')

                metadatas.extend([
                    {
                        'table_id': table['_id'],
                        'is_row': True,
                        'item_id': i_row
                    } for i_row in range(row_emb.shape[0])
                ])

                metadatas.extend([
                    {
                        'table_id': table['_id'],
                        'is_row': False,
                        'item_id': i_column
                    } for i_column in range(col_emb.shape[0])
                ])

                if len(embeddings) > 35000 or read_tables >= batch_size or read_tables + total_read_tables >= n_tables:
                    break
            end_reading_embedding_time = time()
            embedding_ids = [f'{total_processed_embeddings + i}' for i in range(len(metadatas))]

            # storing
            start_storing_time = time()
            rows_columns_collection.add(
                ids=embedding_ids,
                metadatas=metadatas,
                embeddings=embeddings # row_embeddings + column_embeddings
            )
            end_storing_time = time()

            total_read_tables += read_tables
            total_processed_embeddings += len(embeddings) # len(row_embeddings) + len(column_embeddings)
            batch_processing_time.append(round(time() - start_reading_embedding_time, 2))
            mean_batch_proc_time = round(sum(batch_processing_time) / len(batch_processing_time), 2)
            
            print(f'{round(time() - start_test_time, 3)}s: Processed {total_read_tables} tables ({(total_read_tables * 100 / n_tables)}%), batch proc time={batch_processing_time[-1]}s, mean batch proc time={mean_batch_proc_time}s ...')
                #end='\r')            

            total_reading_embedding_time += (end_reading_embedding_time - start_reading_embedding_time)
            total_storing_time += (end_storing_time - start_storing_time)
    
    end_test_time = time()
    total_time = total_reading_embedding_time + total_storing_time
    # print(f'{round(time() - start_test_time, 3)}s: Processed {total_read_tables} tables ({(total_read_tables * 100 / n_tables)}%), batch proc time={batch_processing_time[-1]}s, mean batch proc time={mean_batch_proc_time}s ...')            

    print(f'Test completed.')

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
        total_reading_embedding_time,
        total_reading_embedding_time * 100/ total_time,
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
        'reading-embedding time (s)',
        'reading-embedding time (%)',
        'storing time (s)',
        'storing time (%)',
        'total time (s)',
        'db size (GB)',
        'index size (GB)',
        'db-over-index size fraction']:
        stat[c] = stat[c].apply(lambda x: float(format(x, '.5f')))
    
    if REMOVE_DATABASE_AT_EACH_STEP:
        if n_tables not in KEEP_ONLY_DATABASES_WITH_n_tables or batch_size != 5000:
            os.system(f"rm -rf {DB_ROOT_PATH + db_name}")
    stat.to_csv(DefaultPath.data_path.wikitables + STATISTICS_FILENAME, index=False)
    chroma_client = None
    gc.collect()


