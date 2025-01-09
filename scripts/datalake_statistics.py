import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

from dltf.utils.parallel import chunks
from dltf.utils.settings import DefaultPath as defpath
from dltf.utils.datalake import DataLakeHandlerFactory
from dltf.utils.tables import is_valid_table, table_rows_to_rows


def task(data):
    global dlhconfig, num_cpu
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
    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)

    for table_id in tqdm(table_ids, disable=False if os.getpid() % num_cpu == 0 else True):
        table_obj = dlh.get_table_by_numeric_id(table_id)
        if not table_obj:
            print(table_id, table_obj)
        table = table_obj['content']
        valid_columns = table_obj['valid_columns']
        
        nrows.append(len(table))        
        ncols.append(len(table[0]) if len(table) > 0 else 0)
        area.append(len(table[0]) * len(table) if len(table) > 0 else 0)
        
        if not is_valid_table(table, valid_columns):
            continue

        # filter the numerical columns
        table = table_rows_to_rows(table, 0, len(table[0]), valid_columns)
        
        t = pd.DataFrame(table)
        r, c = t.shape
        accepted_tables += 1

        nrows_final.append(r)
        ncols_final.append(c)
        area_final.append(t.shape[0] * t.shape[1])
        
        # because there are many values like '' in data, which are actually NaN but once encoded into MongoDB
        # turn into empty string(?) 
        nan.append(int(t.replace('', pd.NA).isna().sum().sum()) / area_final[-1])
        n_numeric_cols.append(sum(valid_columns) / len(valid_columns))
    return accepted_tables, nrows, nrows_final, ncols, ncols_final, area, area_final, n_numeric_cols, nan


def initializer(_dlhconfig, _num_cpu):
    global dlhconfig, num_cpu
    dlhconfig = _dlhconfig
    num_cpu = _num_cpu


def main(dlhconfig, num_cpu=os.cpu_count()):
    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
    ntables = dlh.count_tables()

    statistics_path = f'{defpath.data_path.base}/statistics_{dlhconfig[1]}.csv'
    if not os.path.exists(os.path.dirname(statistics_path)):
        os.mkdir(os.path.dirname(statistics_path))

    work = range(ntables)
    with mp.get_context('spawn').Pool(num_cpu, initializer, (dlhconfig, num_cpu)) as pool:
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
        'dataset': dlhconfig[1],
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


if __name__ == '__main__':
    # main(['mongodb', 'wikitables', ['datasets.wikitables']])
    # main(['mongodb', 'wikiturlsnap', ['optitab.turl_training_set', 'sloth.latest_snapshot_tables']])
    main(['mongodb', 'latestsnapshot', ['sloth.latest_snapshot_tables']])
    # main(['mongodb', 'wikilatestsnapshot', ['sloth.demo']])
    # main(['mongodb', 'wikiturl', ['optitab.turl_training_set']])
    # main(dlhconfig=['mongodb', 'gittables', ['sloth.gittables']])
    # main(dlhconfig=[f'{os.environ["HOME"]}/datasets_datalakes/SantosLarge', 'santoslarge', []])


