import re
import sys
import argparse

import polars as pl
import pandas as pd

from tools.josiedataprep.preparation_functions import _create_set_with_bag_semantic, _create_set_with_set_semantic
from tools.utils.settings import DefaultPath as defpath
from tools.sloth.sloth import sloth
from tools.utils.table import from_pandas


parser = argparse.ArgumentParser()
parser.add_argument('mode')
args = parser.parse_args()

mode = args.mode
print(mode)

root_dir = defpath.data_path.base + f'/josie-tests/n45673_m{mode}'

tables_csv_dir =        defpath.data_path.wikitables + '/tables-subset/csv'
josie_sloth_ids_file =  root_dir + '/josie_sloth_ids.csv'
tables_file =           root_dir + '/tables.set'

josie_sloth_ids = pd.read_csv(josie_sloth_ids_file)

josie_res = pd.read_csv(root_dir + '/result_k_5.csv')[['query_id', 'results']]

from pprint import pprint


print('idx', 'query_id', 'set_id ', 'josie_o', 'my_o', 'sloth')
for idx, row in enumerate(josie_res.itertuples(index=False)):
    qid = row[0]
    if type(row[1]) != str:
        continue
    res = row[1]
    sids, overlaps = re.findall(r'\d+', row[1])[::2], re.findall(r'\d+', row[1])[1::2]
    s_id1 = josie_sloth_ids[josie_sloth_ids['josie_id'] == qid]['sloth_id'].values[0]
    
    for sid, o in zip(sids, overlaps):                
        s_id2 = josie_sloth_ids[josie_sloth_ids['josie_id'] == int(sid)]['sloth_id'].values[0]

        df1 = pd.read_csv(tables_csv_dir + '/' + s_id1)
        df2 = pd.read_csv(tables_csv_dir + '/' + s_id2)
        
        if mode == 'set':
            set1 = _create_set_with_set_semantic(df1)
            set2 = _create_set_with_set_semantic(df2)
        elif mode == 'bag':
            set1 = _create_set_with_bag_semantic(df1)
            set2 = _create_set_with_bag_semantic(df2)

        tab1 = from_pandas(df1)
        tab2 = from_pandas(df2)

        mymetr = []
        res = sloth(tab1.columns, tab2.columns, verbose=False, metrics=mymetr)
        largest_ov_sloth = mymetr[-2]
        print(idx, '\t', qid, '\t', sid, '\t', o, '\t', len(set1.intersection(set2)), '\t', largest_ov_sloth)
        if abs(len(set1.intersection(set2)) - int(o)) not in (0, 1):
            print('Errore ', s_id1, s_id2)

