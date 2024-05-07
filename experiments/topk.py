import faiss
import random
import numpy as np
import pandas as pd
import polars as pl
from pprint import pprint

import tools.datahandling as datahandling
import tools.utils.table as table
from tools.utils.settings import DefaultPath as dp

from tools.table_encoding.table_encoder import SentenceTableEncoder


from sloth import sloth




def get_top_k_most_similar_tables(table:table.Table, tabenc, ridx, cidx, rlut, clut, k:int=3):
    """
    :param tabenc: an object which exposes the method encode_table that returns two
    numpy arrays, for row and column encodings
    """

    row_emb, col_emb = tabenc.encode_table(table, True)
    
    # the search value should be customizable
    rD, rI = ridx.search(row_emb, 5)
    cD, cI = cidx.search(col_emb, 5)

    rcnt = np.unique(np.vectorize(rlut.lookup)(rI), return_counts=True)
    ccnt = np.unique(np.vectorize(clut.lookup)(cI), return_counts=True)
    
    rtopk = dict(zip(rcnt[0], rcnt[1]))
    ctopk = dict(zip(ccnt[0], ccnt[1]))

    alpha = 0
    beta = 1 - alpha
    
    from collections import defaultdict
    
    cnt = defaultdict(int)
    
    for id in {*rtopk.keys(), *ctopk.keys()}:
        rv = 0 if id not in rtopk.keys() else alpha * rtopk[id]
        cv = 0 if id not in ctopk.keys() else beta * ctopk[id]
        
        cnt[id] = rv + cv
        
    res = sorted(list(zip(cnt, cnt.values())), key=lambda x: x[1], reverse=True)
    return res[:k] if k >= 0 else res





if __name__ == '__main__':
    path_to_jsonl_file =            dp.data_path.wikitables +   '/train_tables.jsonl'
    path_to_wiki_subset_folder =    dp.data_path.wikitables +   '/r5-c2-a50'
    path_to_tables_ids_file =       dp.data_path.wikitables +   '/r5-c2-a50/ids.txt'
    path_to_csv_folder =            dp.data_path.wikitables +   '/r5-c2-a50/csv'
    path_to_index_folder =          dp.db_path +                '/faiss/basic_index'
    
    
    # Convert the huge JSONL file in many small CSV file into the specified directory
    datahandling.datapreparation.convert_jsonl_to_csv_folder(
        path_to_jsonl_file, 
        path_to_csv_folder,
        path_to_tables_ids_file,
        with_spec={'min_rows':5, 'min_columns':2, 'min_area':50}
    )
    
    # Get the IDs of tables read and converted in the previous step
    with open(path_to_tables_ids_file) as reader:
        table_ids = reader.readlines()
    
    # only 80% of these IDs will be actually used for building the index
    # the remaining 20% is used for testing, as completely new tables
    samples_idx = random.sample(table_ids, round(len(table_ids) * 0.8))
    index_ids = [table_ids[i] for i in samples_idx]
    test_ids = [tid for i, tid in enumerate(table_ids) if i not in samples_idx]
    
    
    # Loading the table encoder
    tabenc = SentenceTableEncoder()
    dim = tabenc.get_encoding_dimension()
    
    # Create the embeddings for the sampled tables
    datahandling.datapreparation.create_embeddings_for_tables(
        path_to_csv_folder, 
        path_to_index_folder, 
        index_ids, 
        tabenc,
        with_labels=False,
        normalize_embeddings=True
    )
    
    k = -1
    
    # loading index and LUTs
    row_index = faiss.read_index(f'{path_to_index_folder}/row_index.index') 
    column_index = faiss.read_index(f'{path_to_index_folder}/column_index.index')
    
    row_lut = datahandling.lut.load_json(f'{path_to_index_folder}/row_lut.json')
    column_lut = datahandling.lut.load_json(f'{path_to_index_folder}/column_lut.json')

    # obtaining a sample table from the test ids set
    df = pd.read_csv(f'{path_to_csv_folder}/{test_ids[0]}')
    s_table = table.from_pandas(df)
    
    # computing the top-K 
    topk = get_top_k_most_similar_tables(table, tabenc, row_index, column_index, row_lut, column_lut)
    
    topk_with_sloth = []
    
    for tabid, tabsim in topk:
        rdf = pd.read_csv(f'{path_to_csv_folder}/{tabid}')
        r_table = table.from_pandas(rdf)
        res, metrics = sloth(r_table.columns, s_table.columns, verbose=False)
        
        largest_overlap_area = 0 if len(res) == 0 else len(res[0][0]) * len(res[0][1])
        topk_with_sloth.append((tabid, round(tabsim, 3), largest_overlap_area))
        
    pprint(topk_with_sloth)