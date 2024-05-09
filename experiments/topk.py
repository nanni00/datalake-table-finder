import os
import json
import faiss
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from collections import defaultdict


import tools.utils.table as table
from tools.datahandling.datapreparation import convert_jsonl_to_csv_folder, create_embeddings_for_tables
from tools.datahandling.lut import load_json
from tools.utils.settings import DefaultPath as dp
from tools.table_encoding.table_encoder import SentenceTableEncoder

from sloth import sloth



def get_top_k_most_similar_tables_count_only(table:table.Table, tabenc, ridx, cidx, rlut, clut, search_index_k:int=5, k:int=3, alpha:float=1, beta:float=1):
    """
    :param tabenc: an object which exposes the method encode_table that returns two
    numpy arrays, for row and column encodings
    """

    row_emb, col_emb = tabenc.encode_table(table, False, True)
    
    # the search value should be customizable
    rD, rI = ridx.search(row_emb, search_index_k)
    cD, cI = cidx.search(col_emb, search_index_k)

    rcnt = np.unique(np.vectorize(rlut.lookup)(rI), return_counts=True)
    ccnt = np.unique(np.vectorize(clut.lookup)(cI), return_counts=True)
    
    rtopk = dict(zip(rcnt[0], rcnt[1]))
    ctopk = dict(zip(ccnt[0], ccnt[1]))
    
    cnt = defaultdict(int)
    
    for id in {*rtopk.keys(), *ctopk.keys()}:
        rv = 0 if id not in rtopk.keys() else rtopk[id]
        cv = 0 if id not in ctopk.keys() else ctopk[id]
        
        cnt[id] = (rv, cv, round(rv * alpha + cv * beta, 3))
        
    res = sorted(list(zip(cnt, cnt.values())), key=lambda x: x[1][2], reverse=True)
    return res[:k] if k >= 0 else res




def get_top_k_most_similar_tables_count_and_dist(
        table:table.Table, tabenc, 
        ridx, cidx, rlut, clut, 
        search_index_k:int=5, k:int=3, 
        alpha:float=1, beta:float=1,
        gamma:float=1, delta:float=1):
    """
    Parameters
    ----------
    table : table.Table        
    tabenc : TYPE
        an object which exposes the method encode_table that returns two numpy arrays, for row and column encodings..    
    search_index_k : int, optional
        The default is 5.
    k : int, optional
        The default is 3.
    alpha : float, optional
        weight of number of rows for each found table. The default is 1.
    beta : float, optional
        weigth of mean distances of rows for each found table. The default is 1.
    gamma : float, optional
        weight of number of columns for each found table. The default is 1.
    delta : float, optional
        weigth of mean distances of columns for each found table. The default is 1.
    """

    row_emb, col_emb = tabenc.encode_table(table, True)
    
    # the search value should be customizable
    rD, rI = ridx.search(row_emb, search_index_k)
    cD, cI = cidx.search(col_emb, search_index_k)
    from itertools import groupby
    
    rI = np.vectorize(rlut.lookup)(rI)
    cI = np.vectorize(rlut.lookup)(cI)
    
    
    row_data = \
        map(
            # q: (groupID, group(distances))
            lambda q: (q[0], len(q[1]) * alpha, np.mean(q[1]) * beta, 'r'),
            map(
                # t: (groupID, groupCompact)
                lambda t: (t[0], [x[1] for x in t[1]]),
                groupby(
                    sorted(
                        [
                            (rI[i][j], rD[i][j], j) 
                            for i in range(rI.shape[0]) 
                            for j in range(rI.shape[1])
                            ]
                        ), 
                    # t: (ID, Distance, #i, #j)
                    key=lambda p: p[0]
                    )
                )
            )
    
    column_data = \
        map(
            # q: (groupID, group(distances))
            lambda q: (q[0], len(q[1]) * gamma, np.mean(q[1]) * delta),
            map(
                # t: (groupID, groupCompact)
                lambda t: (t[0], [x[1] for x in t[1]]),
                groupby(
                    sorted(
                        [
                            (cI[i][j], cD[i][j], j) 
                            for i in range(cI.shape[0]) 
                            for j in range(cI.shape[1])
                            ]
                        ), 
                    # t: (ID, Distance, #i, #j)
                    key=lambda p: p[0]
                    )
                )
            )
            
        
    row_data, column_data = list(row_data), list(column_data)
    
    res = \
        map(
            lambda w: (w[0], sum(ww[1] for ww in w[1]), sum(ww[2] for ww in w[1])), 
            map(
                # z: groupID, groups [(ID, cnt, dist, 'r'/'c')]
                lambda z: (z[0], list(z[1])),
                groupby(
                    sorted(
                        row_data + column_data, 
                        key=lambda x: x[0]
                        ),
                    key=lambda y: y[0]
                    ),
                )
            )

    return list(res)






def main1():
    only_first_n = 5000
    spec = {'min_rows':5, 'min_columns':2, 'min_area':50}
    dcode = f'/n{only_first_n}-r{spec["min_rows"]}-c{spec["min_columns"]}-a{spec["min_area"]}'    
    
    path_to_jsonl_file =            dp.data_path.wikitables +       '/train_tables.jsonl'
    path_to_wiki_subset_folder =    dp.data_path.wikitables +       dcode
    
    path_to_info_file =             path_to_wiki_subset_folder +    '/info.json'
    path_to_csv_folder =            path_to_wiki_subset_folder +    '/csv'
    path_to_index_folder =          path_to_wiki_subset_folder +    '/index'
    
    for d in [
            #path_to_wiki_subset_folder, 
            path_to_csv_folder, 
            #path_to_index_folder
            ]:
        
        if os.path.exists(d):
            os.system(f'rm -rf {d}')
        
        if not os.path.exists(d):
            os.mkdir(d)
        
    # Convert the huge JSONL file in many small CSV file into the specified directory
    convert_jsonl_to_csv_folder(
        path_to_jsonl_file, 
        path_to_csv_folder,
        path_to_info_file,
        only_first_n=only_first_n,
        with_spec=spec
    )
    return
    
    # Get the IDs of tables read and converted in the previous step
    with open(path_to_info_file) as reader:
        info = json.load(reader)
        table_ids = info['ids']
        
    # only 80% of these IDs will be actually used for building the index
    # the remaining 20% is used for testing, as completely new tables
    index_ids = random.sample(table_ids, round(len(table_ids) * 0.8))
    test_ids = [tid for tid in table_ids if tid not in index_ids]
    
    if input(f'{len(index_ids)} tables will be embedded and loaded into a new index (eta {len(index_ids) * 3.5}s): continue? (y/n) ') != 'y':
        return
    
    # Loading the table encoder
    tabenc = SentenceTableEncoder()
    
    # Create the embeddings for the sampled tables
    create_embeddings_for_tables(
        path_to_csv_folder, 
        path_to_index_folder, 
        index_ids, 
        tabenc,
        with_labels=False,
        normalize_embeddings=True
    )
    
    # sanity check, ensuring everything is setup correctly
    
    k = 3
    
    # loading index and LUTs
    row_index = faiss.read_index(f'{path_to_index_folder}/row_index.index') 
    column_index = faiss.read_index(f'{path_to_index_folder}/column_index.index')
    
    row_lut = load_json(f'{path_to_index_folder}/row_lut.json')
    column_lut = load_json(f'{path_to_index_folder}/column_lut.json')

    # obtaining a sample table from the test ids set
    n_id = 1
    df = pd.read_csv(f'{path_to_csv_folder}/{test_ids[n_id]}')
    s_table = table.from_pandas(df)
    
    # computing the top-K 
    topk = get_top_k_most_similar_tables_count_only(s_table, tabenc, row_index, column_index, row_lut, column_lut, k)
    
    topk_with_sloth = []
    
    for tabid, tabsim in topk:
        rdf = pd.read_csv(f'{path_to_csv_folder}/{tabid}')
        r_table = table.from_pandas(rdf)
        res, metrics = sloth(r_table.columns, s_table.columns, verbose=False)
        
        largest_overlap_area = 0 if len(res) == 0 else len(res[0][0]) * len(res[0][1])
        topk_with_sloth.append((tabid, round(tabsim, 3), largest_overlap_area))
        
    pprint(topk_with_sloth)
    

def main2():
    only_first_n = 5000
    spec = {'min_rows':5, 'min_columns':2, 'min_area':50}
    dcode = f'/n{only_first_n}-r{spec["min_rows"]}-c{spec["min_columns"]}-a{spec["min_area"]}'    
    
    path_to_wiki_subset_folder =    dp.data_path.wikitables +       dcode
    
    path_to_info_file =             path_to_wiki_subset_folder +    '/info.json'
    path_to_csv_folder =            path_to_wiki_subset_folder +    '/csv'
    path_to_index_folder =          path_to_wiki_subset_folder +    '/index'
    
    k = -1
          
    # Loading the table encoder
    tabenc = SentenceTableEncoder()
    
    # loading index and LUTs
    row_index = faiss.read_index(f'{path_to_index_folder}/row_index.index') 
    column_index = faiss.read_index(f'{path_to_index_folder}/column_index.index')
    
    row_lut = load_json(f'{path_to_index_folder}/row_lut.json')
    column_lut = load_json(f'{path_to_index_folder}/column_lut.json')

    # Get the IDs of tables read and converted in the previous step
    with open(path_to_info_file) as reader:
        info = json.load(reader)
        table_ids = info['ids']
        
    index_ids = row_lut.ids
    test_ids = [tid for tid in table_ids if tid not in index_ids]
     
    # obtaining a sample table from the test ids set
    n_id = 15
    df = pd.read_csv(f'{path_to_csv_folder}/{test_ids[n_id]}')
    s_table = table.from_pandas(df)

    topk = get_top_k_most_similar_tables_count_only(s_table, 
                                         tabenc, 
                                         row_index, column_index, 
                                         row_lut, column_lut, 
                                         k=k, search_index_k=10,
                                         alpha=0.01, beta=2
                                         )

    topk_with_sloth = []
    
    for tabid, tabsim in topk:
        rdf = pd.read_csv(f'{path_to_csv_folder}/{tabid}')
        r_table = table.from_pandas(rdf)
        res, metrics = sloth(r_table.columns, s_table.columns, bw=16, verbose=False)
        
        largest_overlap_area = 0 if len(res) == 0 else len(res[0][0]) * len(res[0][1])
        topk_with_sloth.append((tabid, *tabsim, largest_overlap_area))
    
    print(f"Analysis on table {test_ids[n_id]}:")
    topk_with_sloth = pd.DataFrame(topk_with_sloth, columns=['id', 'cnt_row', 'cnt_col', 'tabsim', 'overlap'])
    pprint(topk_with_sloth)
    
    import matplotlib.pyplot as plt
    
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['overlap'],   label='overlap')
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['tabsim'],    label='tabsim')
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['cnt_row'],   label='cnt_row')
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['cnt_col'],   label='cnt_col')
    plt.legend()
    plt.show()
    return topk_with_sloth
    


def main3():
    only_first_n = 5000
    spec = {'min_rows':5, 'min_columns':2, 'min_area':50}
    dcode = f'/n{only_first_n}-r{spec["min_rows"]}-c{spec["min_columns"]}-a{spec["min_area"]}'    
    
    path_to_wiki_subset_folder =    dp.data_path.wikitables +       dcode
    
    path_to_info_file =             path_to_wiki_subset_folder +    '/info.json'
    path_to_csv_folder =            path_to_wiki_subset_folder +    '/csv'
    path_to_index_folder =          path_to_wiki_subset_folder +    '/index'
    
    k = -1
          
    # Loading the table encoder
    tabenc = SentenceTableEncoder()
    
    # loading index and LUTs
    row_index = faiss.read_index(f'{path_to_index_folder}/row_index.index') 
    column_index = faiss.read_index(f'{path_to_index_folder}/column_index.index')
    
    row_lut = load_json(f'{path_to_index_folder}/row_lut.json')
    column_lut = load_json(f'{path_to_index_folder}/column_lut.json')

    # Get the IDs of tables read and converted in the previous step
    with open(path_to_info_file) as reader:
        info = json.load(reader)
        table_ids = info['ids']
        
    index_ids = row_lut.ids
    test_ids = [tid for tid in table_ids if tid not in index_ids]
     
    # obtaining a sample table from the test ids set
    n_id = 15
    r_id = test_ids[n_id]
    df = pd.read_csv(f'{path_to_csv_folder}/{r_id}')
    s_table = table.from_pandas(df)

    topk = get_top_k_most_similar_tables_count_and_dist(s_table, 
                                         tabenc, 
                                         row_index, column_index, 
                                         row_lut, column_lut, 
                                         k=k, search_index_k=7,
                                         alpha=1, beta=0.01,
                                         gamma=1, delta=0.5
                                         )
    
    
    topk_df = pd.DataFrame(topk, columns=['id', 'cnt', 'meandist'])
    topk_df = topk_df.sort_values(by=['cnt', 'meandist'], ascending=[False, True])
    topk = list(zip(*map(topk_df.get, topk_df)))
    print(topk_df)
    
    topk_with_sloth = []
    
    for tabid, cnt, meandist in topk:
        rdf = pd.read_csv(f'{path_to_csv_folder}/{tabid}')
        r_table = table.from_pandas(rdf)
        res, metrics = sloth(r_table.columns, s_table.columns, bw=16, verbose=False)
        
        largest_overlap_area = 0 if len(res) == 0 else len(res[0][0]) * len(res[0][1])
        topk_with_sloth.append((tabid, cnt, meandist, largest_overlap_area))
    
    print(f"Analysis on table {test_ids[n_id]}:")
    topk_with_sloth = pd.DataFrame(topk_with_sloth, columns=['id', 'cnt', 'meandist', 'overlap'])
    pprint(topk_with_sloth)
    
    
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['overlap'],   label='overlap')
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['meandist'],  label='meandist')
    plt.plot(range(len(topk_with_sloth)), topk_with_sloth['cnt'],       label='cnt')
    
    plt.legend()
    plt.show()
    
    for idx, row in topk_with_sloth.iterrows():
        if row[3] == 0:
            continue
        else:
            x = input(f'Table {row[0]} has an overlap of {row[3]}: see it? (y/n/exit) ')
            if x == 'y':        
                s_id = row[0]
                        
                r_df = pd.read_csv(f'{path_to_csv_folder}/{r_id}')
                s_df = pd.read_csv(f'{path_to_csv_folder}/{s_id}')
                
                r_table = table.from_pandas(r_df)
                s_table = table.from_pandas(s_df)
                
                res, metr = sloth(r_table.columns, s_table.columns, verbose=False)
                print(len(res), len(res[0][0]), len(res[0][1]), len(res[0][0]) * len(res[0][1]))
                print(res)
                
            elif x == 'n':
                continue
            else:
                break



if __name__ == '__main__':
    # main1()
    res = main2()
    # main3()