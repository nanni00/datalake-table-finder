import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.utils.utils import check_table_is_in_thresholds
from tools.utils.mongodb_utils import get_mongodb_collections
from tools.utils.settings import DefaultPath as defpath

size = 'standard'

statistics_path = f'{defpath.data_path.base}/dataset_stat/stat_{size}.csv'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.mkdir(os.path.dirname(statistics_path))


for dataset in ['wikipedia', 'gittables']:
    mongoclient, collections = get_mongodb_collections(dataset, size)

    tables_thresholds = {
        'min_row':      5,
        'min_column':   2,
        'min_area':     0,
        'max_row':      999999,
        'max_column':   999999,
        'max_area':     999999,
    }

    tables = sum(c.count_documents({}) for c in collections)
    thresh_tables = 0

    nrows = []
    nrows_final = []
    ncols = []
    ncols_final = []
    area = []
    area_final = []
    nan = []
    num_cols = []

    for collection in collections:
        print(f'Scanning documents from {collection.database.name}.{collection.name}...')
        for doc in tqdm(collection.find({}), total=collection.count_documents({})):
            table = doc['content']
            nrows.append(len(table))
            if check_table_is_in_thresholds(table, tables_thresholds) and not all(doc['numeric_columns']):
                thresh_tables += 1
                nrows_final.append(len(table))
                ncols.append(len(table[0]))
                num_cols.append(sum(doc['numeric_columns']) / len(doc['numeric_columns']))
                area.append(len(table) * len(table[0]))
                t = [[row[i] for i, x in enumerate(doc['numeric_columns']) if x == 0] for row in doc['content']]

                t = pd.DataFrame(t)
                ncols_final.append(t.shape[1])
                area_final.append(t.shape[0] * t.shape[1])
                # because there are many values like '' in data, which are actually NaN but once encoded into MongoDB
                # turn into empty string(?) 
                nan.append(int(t.replace('', pd.NA).isna().sum().sum()) / area_final[-1])

    stat = {
        'dataset': dataset,
        'size': size,
        'tables': tables,
        'tables_in_thresholds': thresh_tables,
        
        'rows_tot': np.sum(nrows),
        'rows_max': np.amax(nrows),
        'rows_mean': np.mean(nrows),
        'rows_stdev': np.std(nrows),
        
        'final_rows_tot': np.sum(nrows_final),
        'final_rows_max': np.amax(nrows_final),
        'final_rows_mean': np.mean(nrows_final),
        'final_rows_stdev': np.std(nrows_final),

        'columns_tot': np.sum(ncols),
        'columns_max': np.amax(ncols),
        'columns_mean': np.mean(ncols),
        'columns_stdev': np.std(ncols),
        
        'final_columns_tot': np.sum(ncols_final),
        'final_columns_max': np.amax(ncols_final),
        'final_columns_mean': np.mean(ncols_final),
        'final_columns_stdev': np.std(ncols),
        
        'area_mean': np.mean(area),
        'area_max': np.amax(area),
        'area_stdev': np.std(area),
        
        'final_area_mean': np.mean(area_final),
        'final_area_max': np.amax(area_final),
        'final_area_stdev': np.std(area_final),

        '%numeric_columns_per_table_mean': np.mean(num_cols),
        '%numeric_columns_per_table_stdev': np.std(num_cols),
        
        '%nan_per_table_mean': np.mean(nan),
        '%nan_per_table_stdev': np.std(nan),
    }
    stat = {k: [round(v, 5) if k not in {'dataset', 'size'} else v] for k, v in stat.items()}

    append = os.path.exists(statistics_path)
    stat = pd.DataFrame.from_dict(stat, orient='columns')
    stat.to_csv(statistics_path, index=False, mode='a' if append else 'w', header=False if append else True)



