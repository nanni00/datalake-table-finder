import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.utils.misc import is_valid_table
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.settings import DefaultPath as defpath
from tools.utils.parallel_worker import chunks
import multiprocessing as mp

# dataset = 'santoslarge'
# size = 'standard'
# datalake_location = "/data4/nanni/data/santos_large/datalake"
dataset = "wikiturlsnap"
size = "standard"
datalake_location = "mongodb"

mapping_id_file = "/data4/nanni/data/santos_large/mapping_id.pickle"
numeric_columns_file = "/data4/nanni/data/santos_large/numeric_columns.pickle"

statistics_path = f'{defpath.data_path.base}/dataset_stat/stat_{size}_2.csv'

if not os.path.exists(os.path.dirname(statistics_path)):
    os.mkdir(os.path.dirname(statistics_path))

dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)

num_cpu = 72
ntables = dlh.get_number_of_tables()


# for table_obj in tqdm(dlh.scan_tables(), total=ntables):
def task(data):
    global datalake_location, dataset, size, mapping_id_file, numeric_columns_file, num_cpu
    # if os.getpid() % num_cpu == 0:
    #     print(data)

    accepted_tables = 0
    nrows = []
    nrows_final = []
    ncols = []
    ncols_final = []
    area = []
    area_final = []
    nan = []
    n_numeric_cols = []

    table_ids = data[0]
    dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)

    for table_id in tqdm(table_ids, disable=False if os.getpid() % num_cpu == 0 else True):
        table_obj = dlh.get_table_by_numeric_id(table_id)
        table = table_obj['content']
        numeric_columns = table_obj['numeric_columns']

        nrows.append(len(table))        
        ncols.append(len(table[0]) if len(table) > 0 else 0)
        area.append(len(table[0]) * len(table) if len(table) > 0 else 0)
        
        if not is_valid_table(table, numeric_columns):
            continue

        # filter the numerical columns
        table = [[row[i] for i, x in enumerate(numeric_columns) if x == 0] for row in table]
        
        t = pd.DataFrame(table)
        r, c = t.shape
        accepted_tables += 1

        nrows_final.append(r)
        ncols_final.append(c)
        area_final.append(t.shape[0] * t.shape[1])
        
        # because there are many values like '' in data, which are actually NaN but once encoded into MongoDB
        # turn into empty string(?) 
        nan.append(int(t.replace('', pd.NA).isna().sum().sum()) / area_final[-1])
        n_numeric_cols.append(sum(numeric_columns) / len(numeric_columns))
    return accepted_tables, nrows, nrows_final, ncols, ncols_final, area, area_final, n_numeric_cols, nan


work = range(ntables)
with mp.Pool(num_cpu) as pool:
    results = pool.map(task, chunks(work, ntables // num_cpu))
accepted_tables = sum(r[0] for r in results)
nrows =         [r for rr in results for r in rr[1]]
nrows_final =   [r for rr in results for r in rr[2]]
ncols =         [r for rr in results for r in rr[3]]
ncols_final =   [r for rr in results for r in rr[4]]
area =          [r for rr in results for r in rr[5]]
area_final =    [r for rr in results for r in rr[6]]
n_numeric_cols =[r for rr in results for r in rr[7]]
nan =           [r for rr in results for r in rr[8]]


stat = {
    'dataset': dataset,
    'size': size,
    'tables': ntables,
    'accepted_tables': accepted_tables,
    
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

    '%numeric_columns_per_table_mean': np.mean(n_numeric_cols),
    '%numeric_columns_per_table_stdev': np.std(n_numeric_cols),
    
    '%nan_per_table_mean': np.mean(nan),
    '%nan_per_table_stdev': np.std(nan),
}

# stat = {k: [round(v, 5) if k not in {'dataset', 'size'} else v] for k, v in stat.items()}
stat = [[k, round(v, 5) if k not in {'dataset', 'size'} else v] for k, v in stat.items()]
# stat = pd.DataFrame.from_dict(stat, orient='columns')

append = os.path.exists(statistics_path)
if os.path.exists(statistics_path):
    statdf = pd.read_csv(statistics_path)
    statdf[len(statdf.columns) + 1] = [s[1] for s in stat]
else:
    statdf = pd.DataFrame.from_dict(stat, orient='columns')

statdf.to_csv(statistics_path, index=False)



