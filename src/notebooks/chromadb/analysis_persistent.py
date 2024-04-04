import os
import jsonlines
import pandas as pd
from time import time
from tqdm.notebook import tqdm
from itertools import product

from code.fasttext.embedding_utils import TableEncoder
from code.utils.settings import DefaultPath
from code.utils.utils import rebuild_table

import chromadb


def execute_test(dataset, batch_size, with_metadata, collection_column, collection_row):        
    def add_to_db(coll, ids, emb, metadatas=None):
        if metadatas: coll.add(ids=ids, embeddings=emb, metadatas=metadatas)
        else: coll.add(ids=ids, embeddings=emb)

    i = 0 
    batch_column_ids, batch_row_ids, batch_column_embeddings, batch_row_embeddings = [], [], [], []
    column_metadatas, row_metadatas = [], []

    with jsonlines.open(DefaultPath.data_path.wikitables + dataset, 'r') as reader:
        for wikitable in reader:            
            i += 1
            table_id, table = wikitable['_id'], rebuild_table(wikitable)
            row_embeddings, column_embeddings = tabenc.full_embedding(table, False, False)

            batch_column_embeddings.extend(column_embeddings.tolist())
            batch_row_embeddings.extend(row_embeddings.tolist())

            # ID - version 1
            batch_column_ids.extend([f'{table_id}#{idx}' for idx in range(column_embeddings.shape[0])])
            batch_row_ids.extend([f'{table_id}#{idx}' for idx in range(row_embeddings.shape[0])]) 
            
            if with_metadata:
                column_metadatas.extend([{'table_id': table_id}] * column_embeddings.shape[0])
                row_metadatas.extend([{'table_id': table_id}] * row_embeddings.shape[0])

            if i % batch_size == 0 and i != 0:
                add_to_db(collection_column, batch_column_ids, batch_column_embeddings, column_metadatas)
                add_to_db(collection_row, batch_row_ids, batch_row_embeddings, row_metadatas)      
                batch_column_ids, batch_row_ids, batch_column_embeddings, batch_row_embeddings = [], [], [], []
                column_metadatas, row_metadatas = [], []
                print(f"Loaded tables: {i}...", end='\r')
                
        if batch_column_ids:
            # if case the batch isn't empty but it hasn't been loaded previously
            add_to_db(collection_column, batch_column_ids, batch_column_embeddings)
            add_to_db(collection_row, batch_row_ids, batch_row_embeddings)  
            print(f"Loaded tables: {i}...", end='\r')



tabenc = TableEncoder()

batch_sizes = [500, 1000, 3500]
with_metadatas = [True, False]
datasets = ['medium_train_tables.jsonl']
cases = product(datasets, batch_sizes, with_metadatas)

stat = pd.DataFrame(columns=['test ID', 'VDB name', 'dataset', 'batch size', 'with metadata', 'total time (s)'])

for i, case in enumerate(cases):
    dataset, batch_size, with_metadata = case
    print(f'############### EXECUTING TEST {i} ###############')
    print(f'{dataset}\t{batch_size}\t{with_metadata}')

    chroma_client = chromadb.PersistentClient(DefaultPath.db_path.chroma + f'v_m_{i}')

    try:
        chroma_client.delete_collection("column-base")
        chroma_client.delete_collection("row-base")
        print('Collections deleted')
    except: pass
    finally:
        collection_column = chroma_client.create_collection(name='column-base')
        collection_row = chroma_client.create_collection(name='row-base')
        print('Collections created')

    start = time()
    execute_test(dataset, batch_size, with_metadata, collection_column, collection_row)
    stat.loc[len(stat)] = [
        i, f'v_m_{i}', dataset, batch_size, with_metadata, time() - start
    ]

stat.to_csv('src/notebooks/chromadb/medium_stat.csv', index=False)
