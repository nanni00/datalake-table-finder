#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:00:42 2024

@author: giovanni
"""

import json
import faiss
import jsonlines
import pandas as pd
from tqdm import tqdm

import lut
import tools.utils.table as table
from tools.utils.utils import rebuild_table
from tools.table_encoding.table_encoder import TableEncoder




def convert_jsonl_to_csv_folder(
        path_to_jsonl_file:str,
        path_to_csv_folder:str,
        path_to_ids_file:str|None=None,
        only_first_n:int|None=None,
        ids_subset:set[str]|None=None,
        with_spec:dict[str:int]|None=None):
    
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
    
    if path_to_ids_file:
        with open(path_to_ids_file, 'w') as writer:
            writer.writelines(read_ids)
            
    with open(f'{path_to_csv_folder}/info.json', 'w') as writer:
        json.dump(
            {
                'only_first_n': only_first_n,
                'with_spec': with_spec,
                'ids_subset': ids_subset                    
            }
        )



def create_embeddings_for_tables(
        path_to_csv_folder:str, 
        path_to_index_folder:str,
        table_ids:list[str], 
        tabenc:TableEncoder,
        with_labels=False,
        normalize_embeddings=False):
    
    d = tabenc.get_encoding_size()
    row_index = faiss.IndexFlatL2(d)
    column_index = faiss.IndexFlatL2(d)
    
    row_lut = lut.LUT()
    column_lut = lut.LUT()
    
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
        
