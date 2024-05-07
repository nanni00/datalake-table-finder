#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:26:08 2024

@author: giovanni
"""

"""
r_id,s_id,jsim,o_a,a%
15182906-1,33954290-1,0.7878787878787878,32,0.6153846153846154
14772277-1,32928438-1,0.9,36,0.6923076923076923
2198024-3,38093453-3,0.3082706766917293,16,0.1
"""

import pandas as pd
from sloth import sloth
from tools.utils.table import Table
from tools.utils.settings import DefaultPath as defpath

r_id, s_id = '15182906-1', '33954290-1'
# r_id, s_id = '14772277-1', '32928438-1'
# r_id, s_id = '2198024-3', '38093453-3'
r_id, s_id = '20941454-14', '37358124-1'

r_df = pd.read_csv(defpath.data_path.wikitables +
                      f'threshold-r5-c2-a50/csv/{r_id}')
s_df = pd.read_csv(defpath.data_path.wikitables +
                      f'threshold-r5-c2-a50/csv/{s_id}')

r_table = Table()
r_table.from_pandas(r_df)
s_table = Table()
s_table.from_pandas(s_df)

print(r_table.columns)
print()
print(s_table.columns)

res, metr = sloth(r_table.columns, s_table.columns)
print(len(res), len(res[0][0]), len(res[0][1]))
print(len(res[0][0]) * len(res[0][1]))

# Restructure the table as the list of its columns, ignoring the headers
def parse_table(table, num_cols, num_headers):
    return [[row[i] for row in table[num_headers:]] for i in range(0, num_cols)]


# r_table = parse_table(r_table, r_table.shape[1], r_table.shape[0])
# s_table = parse_table(s_table, s_table.shape[1], s_table.shape[0])
