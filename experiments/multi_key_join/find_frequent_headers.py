import re
import os
import json
import multiprocessing as mp

from tqdm import tqdm

from dltftools.utils.datalake import DataLakeHandlerFactory
from dltftools.utils.parallel import chunks
from dltftools.utils.settings import DefaultPath as dp
from dltftools.utils.tables import _is_number, is_valid_table

from datamining.utils.misc import create_items_table
from datamining.frequent_itemsets.apriori import apriori


def task(data):
    global dlhconfig
    _range = data[0]
    headers = []
    dlh = dlh = DataLakeHandlerFactory.create_handler(dlhconfig)
    for i in _range:
        table_obj = dlh.get_table_by_numeric_id(i)
        if not is_valid_table(table_obj['content'], table_obj['numeric_columns']) or 'headers' not in table_obj or not table_obj['headers']: 
            continue
        headers.append(sorted([h.lower().strip() for i, h in enumerate(table_obj['headers']) 
                            if table_obj['numeric_columns'][i] == 0 
                            and not _is_number(h) 
                            and 'nnamed: ' not in str(h)
                            and not re.match(r'.\d+|\d+%', str(h).lower().strip)]))
    return headers


def initializer(_dlhconfig):
    global dlhconfig
    dlhconfig = _dlhconfig


def find_frequent_headers(dlhconfig, k=4, s=500, num_cpu=10):
    """
    Find the the frequent headers in the given tables collection
    It uses A-Priori, so tune parameters k and s to your needs
    :param dlhconfig: A list of configuration parameters used to access the tables repository
    :param k: The maximum size of the frequent itemsets to find
    :param s: Threshold search parameter of A-Priori
    :param num_cpu: Number of CPUs that can be used
    """
    read_headers =          True
    headers_list_file =     f'{dp.root_project_path}/experiments/multi_key_join/all_headers/{dlhconfig[1]}.json'
    frequent_headers_file = f'{dp.root_project_path}/experiments/multi_key_join/frequent_headers/{dlhconfig[1]}.json'

    for f in [frequent_headers_file, headers_list_file]:
        if not os.path.exists(os.path.dirname(f)):
            os.makedirs(os.path.dirname(f))

    dlh = DataLakeHandlerFactory.create_handler(dlhconfig)
    ntables = dlh.count_tables()

    print('Reading headers...')
    headers = []
    if read_headers:
        with mp.Pool(num_cpu, initializer, [dlhconfig]) as pool:
            for headers_list in tqdm(pool.map(task, chunks(range(ntables), ntables // os.cpu_count())), leave=False):
                for h in headers_list:
                    headers.append(h)
            with open(headers_list_file, 'w') as fw:
                json.dump(headers, fw)
    else:
        with open(headers_list_file) as fr:
            headers = json.load(fr)

    headers, items_mapping = create_items_table(headers)

    print('Running A-Priori algorithm...')
    results = apriori(headers, k, s, len(headers), verbose=True)

    results = {
        k: {
            tuple(sorted(str(items_mapping.inverse[item]) for item in itemset)) if not isinstance(itemset, int) else items_mapping.inverse[itemset]: freq
            for itemset, freq in k_res.items()
        } for k, k_res in results.items() 
    }

    results = {
        k: {
            ' : '.join(sorted(itemset)) if isinstance(itemset, tuple) else itemset: freq
            for itemset, freq in sorted(k_res.items(), key=lambda x: x[0])
        } for k, k_res in results.items() 
    }

    print(f'Saving results in {frequent_headers_file}...')
    with open(frequent_headers_file, 'w') as fw:
        json.dump(results, fw, indent=4)


def main_wikitables():
    dlhconfig = ['mongodb', 'wikitables', ['datasets.wikitables']]
    find_frequent_headers(dlhconfig)


if __name__ == '__main__':
    main_wikitables()
