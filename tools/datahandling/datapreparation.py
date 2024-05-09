#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:00:42 2024

@author: giovanni
"""

import json
import faiss
import random
import jsonlines
import pandas as pd
from tqdm import tqdm

import tools.datahandling.lut as lut
import tools.utils.table as table
from tools.utils.utils import rebuild_table
from tools.table_encoding.table_encoder import TableEncoder




def convert_jsonl_to_csv_folder(
        path_to_jsonl_file:str,
        path_to_csv_folder:str,
        path_to_ids_file:str|None=None,
        only_first_n:int|None=None,
        ids_subset:set[str]|None=None,
        with_spec:dict[str:int]|None=None,
        to_index_perc:float=0.8):
    
    if not any([only_first_n, ids_subset, with_spec]):
        raise ValueError('At least one option among only_first_n, ids_subset and with_spec must be specified.')

    print('Opening JSONL file...')
    jreader = jsonlines.open(path_to_jsonl_file, 'r')
    read_ids = []
    
    if with_spec:
        min_rows = with_spec['min_rows']
        min_columns = with_spec['min_columns']
        min_area = with_spec['min_area']    
    
    print('Start reading tables...')
    for i, jtable in tqdm(enumerate(jreader), total=only_first_n if only_first_n else None):
        if only_first_n and i >= only_first_n:
            break
        if ids_subset and jtable['_id'] not in ids_subset:
            continue
        if with_spec:
            if len(jtable['tableData']) < min_rows \
                or len(jtable['tableHeaders'][0]) < min_columns \
                    or len(jtable['tableData']) * len(jtable['tableHeaders'][0]) < min_area:
                        continue
                    
        read_ids.append(jtable['_id'])
        rebuild_table(jtable).to_csv(f'{path_to_csv_folder}/{jtable["_id"]}', index=False)    
        
    print('All the requested tables have been read.')
    jreader.close()
    
    # only 80% of these IDs will be actually used for building the index
    # the remaining 20% is used for testing, as completely new tables
    index_ids = random.sample(read_ids, round(len(read_ids) * to_index_perc))
    test_ids = [tid for tid in read_ids if tid not in index_ids]
    
    
    if path_to_ids_file:
      with open(path_to_ids_file, 'w') as writer:
          json.dump(
              {
                  'only_first_n': only_first_n,
                  'with_spec': with_spec,
                  'ids_subset_only': ids_subset,
                  'index_tables_ids': index_ids,
                  'test_tables_ids': test_ids
              },
              writer
          )



def create_embeddings_for_tables(
        path_to_csv_folder:str, 
        path_to_index_folder:str,
        table_ids:list[str], 
        tabenc:TableEncoder,
        with_labels=False,
        normalize_embeddings=False):
    
    d = tabenc.get_encoding_dimension()
    row_index = faiss.IndexFlatL2(d)
    column_index = faiss.IndexFlatL2(d)
    
    row_lut = lut.LUT()
    column_lut = lut.LUT()
    print('Start embedding tables and loading them into the FAISS index...')
    for tid in tqdm(table_ids):
        df = pd.read_csv(f'{path_to_csv_folder}/{tid}')
        t = table.from_pandas(df)
        row_emb, col_emb = tabenc.encode_table(t, with_labels, normalize_embeddings)
        
        row_index.add(row_emb)
        column_index.add(col_emb)
        
        row_lut.insert_index(row_emb.shape[0], tid)
        column_lut.insert_index(col_emb.shape[0], tid)
        
    faiss.write_index(row_index, f'{path_to_index_folder}/row_index.index')
    faiss.write_index(column_index, f'{path_to_index_folder}/column_index.index')
    
    row_lut.save(f'{path_to_index_folder}/row_lut.json')
    column_lut.save(f'{path_to_index_folder}/column_lut.json')
    print('All the tables have been embedded and loaded into the index.')
