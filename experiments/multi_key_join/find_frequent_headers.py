import os
import json
import multiprocessing as mp
from collections import defaultdict
import re

from tqdm import tqdm

from dltftools.myutils.parallel import chunks
from dltftools.myutils.misc import is_number_tryexcept, is_valid_table
from dltftools.myutils.settings import DefaultPath as dp
from dltftools.myutils.datalake import SimpleDataLakeHelper

from datamining.frequent_itemsets.apriori import apriori
from datamining.utils.misc import create_items_table


test_name =             'main'
size =                  'standard'
mode =                  'bag'
dataset =               'gittables'
dbname =                'nanni'
read_headers =          True
headers_list_file =     f'{dp.root_project_path}/experiments/multi_key_join/all_headers/{dataset}.json'
frequent_headers_file = f'{dp.root_project_path}/experiments/multi_key_join/frequent_headers/{dataset}.json'

# A-Priori parameters
k = 4
s = 500

for f in [frequent_headers_file, headers_list_file]:
    if not os.path.exists(os.path.dirname(f)):
        os.makedirs(os.path.dirname(f))


match dataset:
    case 'wikiturlsnap' | 'gittables':
        datalake_location = "mongodb"
        mapping_id_file = numeric_columns_file = None
    case 'santoslarge':
        datalake_location = "/data4/nanni/data/santos_large/datalake"
        mapping_id_file = "/data4/nanni/data/santos_large/mapping_id.pickle"
        numeric_columns_file = "/data4/nanni/data/santos_large/numeric_columns.pickle"
    case _:
        raise ValueError()
    

header_cnt = defaultdict(int)

dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)
ntables = dlh.get_number_of_tables()


def task(data):
    _range = data[0]
    headers = []
    dlh = SimpleDataLakeHelper(datalake_location, dataset, size, mapping_id_file, numeric_columns_file)
    for i in _range:
        table_obj = dlh.get_table_by_numeric_id(i)
        if not is_valid_table(table_obj['content'], table_obj['numeric_columns']) or 'headers' not in table_obj or not table_obj['headers']: 
            continue
        headers.append(sorted([h.lower().strip() for i, h in enumerate(table_obj['headers']) 
                               if table_obj['numeric_columns'][i] == 0 
                               and not is_number_tryexcept(h) 
                               and 'nnamed: ' not in str(h)
                               and not re.match(r'.\d+|\d+%', str(h).lower().strip)]))
    return headers

print('Reading headers...')
headers = []
if read_headers:
    with mp.Pool() as pool:
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
