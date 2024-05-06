import faiss
import numpy as np
import pandas as pd
import polars as pl
from pprint import pprint

from tools.faiss.lut import LUT
from tools.utils.table import Table
from tools.utils.settings import DefaultPath as dp

from sloth import sloth

from sentence_transformers import SentenceTransformer

import itertools
import functools


class TableEncoder:
    def __init__(self, model):
        self.model = model

    def encode_table(self, table:Table, normalize_embeddings=False):
        row_embs = self.model.encode([
                '|'.join([f'{h},{cell}' for (h, cell) in zip(table.headers, t)]) 
                for t in table.get_tuples()
            ],
            normalize_embeddings=normalize_embeddings
        )

        col_embs = self.model.encode([
                f"{h},{','.join(map(str, col))}"
                for h, col in zip(table.headers, table.columns)
            ],
            normalize_embeddings=normalize_embeddings
        )

        return row_embs, col_embs


def get_top_k_most_similar_tables(table:Table, tabenc, ridx, cidx, rlut, clut, k:int=3):
    """
    :param tabenc: an object which exposes the method encode_table that returns two
    numpy arrays, for row and column encodings
    """

    row_emb, col_emb = tabenc.encode_table(table, True)
    
    rD, rI = ridx.search(row_emb, k)
    cD, cI = cidx.search(col_emb, k)

    rcnt = np.unique(np.vectorize(rlut.lookup)(rI), return_counts=True)
    ccnt = np.unique(np.vectorize(clut.lookup)(cI), return_counts=True)
    
    # rtopk = sorted(list(zip(rcnt[0], rcnt[1])), key=lambda x: x[1], reverse=True)
    # ctopk = sorted(list(zip(ccnt[0], ccnt[1])), key=lambda x: x[1], reverse=True)
    # return rtopk, ctopk
    
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
        
    return sorted(list(zip(cnt, cnt.values())), key=lambda x: x[1], reverse=True)





if __name__ == '__main__':
    test_root_dir = dp.data_path.wikitables + 'threshold-r5-c2-a50'
    results = pl.scan_csv(f'{test_root_dir}/sloth-results-r5-c2-a50.csv')
    tables_path = f'{test_root_dir}/sloth-tables-r5-c2-a50.jsonl'
    k = 30
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    dim = model.get_sentence_embedding_dimension()
    
    dir_path = dp.root_project_path + 'notebooks/faiss/'
    row_index = faiss.read_index(dir_path + 'basic_index/row_index.index') 
    column_index = faiss.read_index(dir_path + 'basic_index/column_index.index')
    row_lut, column_lut = LUT(), LUT()

    row_lut.load(dir_path + 'basic_index/row_lut.json')
    column_lut.load(dir_path + 'basic_index/column_lut.json')

    index_ids = set(results.select(pl.col('r_id')).head(100).unique().collect()['r_id'].to_list())
    test_ids = list(set(results.select(pl.col('s_id')).head(100).unique().collect()['s_id'].to_list()).difference(index_ids))

    df = pl.read_csv(test_root_dir +  f'/csv/{test_ids[0]}')
    print(results.head(100).filter(pl.col('s_id') == test_ids[0]).collect())

    table = Table()
    table.from_polars(df)
    tabenc = TableEncoder(model)

    topk = get_top_k_most_similar_tables(table, tabenc, row_index, column_index, row_lut, column_lut)
    
    pprint(topk[:k])
        
    # Restructure the table as the list of its columns, ignoring the headers
    def parse_table(table, num_cols, num_headers):
        return [[row[i] for row in table[num_headers:]] for i in range(0, num_cols)]
    
    topk_with_sloth = []
    for tabid, tabsim in topk[:k]:
        rdf = pd.read_csv(test_root_dir +  f'/csv/{tabid}')
        r_table = Table()
        r_table.from_pandas(rdf)
        r_table = r_table.columns
        s_table = table.columns
        res, metrics = sloth(r_table, s_table, verbose=False)
        
        largest_overlap_area = 0 if len(res) == 0 else len(res[0][0]) * len(res[0][1])
        topk_with_sloth.append((tabid, round(tabsim, 3), largest_overlap_area))