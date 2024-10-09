import os
import pickle
from bidict import bidict
from tqdm import tqdm

import polars as pl

from thesistools.utils.misc import naive_detect_bad_columns

# TODO customize paths
data_dir = '/path/to/santos_large/datalake'
mapping_path = '/path/to/santos_large/mapping_id.pickle'
numcol_path = '/path/to/santos_large/numeric_columns.pickle'

ntables = len(os.listdir(data_dir))
ignored_tables = 0
nrows = 0
ncols = 0
counter = 0

mapping_id = bidict()
numeric_columns = dict()

for table_file in tqdm(os.listdir(data_dir), total=ntables):
    try:
        mapping_id[counter] = table_file.removesuffix('.csv')
        table = pl.read_csv(f'{data_dir}/{table_file}', infer_schema_length=0, encoding='latin8', has_header=False).rows()
        nrows += len(table)
        ncols += len(table[0])
        numeric_columns[counter] = naive_detect_bad_columns(table)
        counter += 1
    except:
        ignored_tables += 1

print(f'#tables={ntables}, #effectives={counter}, #ignored={ignored_tables}, #rows={nrows}, #cols={ncols}')

print('Saving mapping...')
with open(mapping_path, 'wb') as fw:
    pickle.dump(mapping_id, fw)

print('Saving numeric columns...')
with open(numcol_path, 'wb') as fw:
    pickle.dump(numeric_columns, fw)

print('Completed.')
