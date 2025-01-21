import re
import os
import json
import multiprocessing as mp

from tqdm import tqdm

from dltf.utils.datalake import DataLakeHandlerFactory
from dltf.utils.misc import chunks
from dltf.utils.settings import DefaultPath as dp
from dltf.utils.tables import _is_number, is_valid_table

# from datamining.utils.misc import create_items_table
# from datamining.frequent_itemsets.apriori import apriori
from efficient_apriori import apriori


def task(data):
    global dlhconfig
    _range = data[0]
    headers = []
    dlh = dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
    for i in _range:
        table_obj = dlh.get_table_by_numeric_id(i)
        if not is_valid_table(table_obj['content'], table_obj['valid_columns']) or 'headers' not in table_obj or not table_obj['headers']: 
            continue
        
        # for wikitables there could be num_header_rows>=2, 
        # but for simplicity we use only those tables with num_header_rows==1
        if isinstance(table_obj['headers'][0], list) and len(table_obj['headers']) > 1:
            continue
        
        headers.append(sorted([h.lower().strip() for i, h in enumerate(table_obj['headers'][0]) 
                            if table_obj['valid_columns'][i] == 1 
                            and not _is_number(h) 
                            and 'nnamed: ' not in str(h)]))
                            # and not re.match(r'.\d+|\d+%', str(h).lower().strip)]))
    return headers


def initializer(_dlhconfig):
    global dlhconfig
    dlhconfig = _dlhconfig


def find_frequent_headers(dlhconfig, k, min_support, num_cpu=os.cpu_count()):
    """
    Find the the frequent headers in the given tables collection
    It uses A-Priori, so tune parameters k and s to your needs
    :param dlhconfig: A list of configuration parameters used to access the tables repository
    :param k: The maximum size of the frequent itemsets to find
    :param s: Threshold search parameter of A-Priori
    :param num_cpu: Number of CPUs that can be used
    """
    headers_list_file       = f'{os.path.dirname(__file__)}/headers/{dlhconfig[1]}_all.json'
    frequent_headers_file   = f'{os.path.dirname(__file__)}/headers/{dlhconfig[1]}_frequent.json'
    read_headers            = not os.path.exists(headers_list_file)

    for f in [frequent_headers_file, headers_list_file]:
        if not os.path.exists(os.path.dirname(f)):
            print(f'Creating folder {os.path.dirname(f)}...')
            os.makedirs(os.path.dirname(f))

    dlh = DataLakeHandlerFactory.create_handler(*dlhconfig)
    ntables = dlh.count_tables()

    print('Reading headers...')
    headers = []
    if read_headers:
        with mp.get_context('spawn').Pool(num_cpu, initializer, [dlhconfig]) as pool:
            for headers_list in tqdm(pool.map(task, chunks(range(ntables), ntables // os.cpu_count())), leave=False):
                for h in headers_list:
                    headers.append(h)
            with open(headers_list_file, 'w') as fw:
                json.dump(headers, fw)
    else:
        print('Reading from file...')
        with open(headers_list_file) as fr:
            headers = json.load(fr)

    print('Running A-Priori algorithm...')
    min_support = min_support / len(headers) 
    itemsets, _ = apriori(transactions=headers, max_length=k, min_support=min_support, verbosity=True)

    itemsets = {
        k : {
            ' : '.join(sorted(itemset)) if isinstance(itemset, tuple) else itemset: freq
            for itemset, freq in sorted(k_its.items(), key=lambda x: x[0])
        }
        for k, k_its in itemsets.items()
    }


    print(f'Saving results in {frequent_headers_file}...')
    with open(frequent_headers_file, 'w') as fw:
        json.dump(itemsets, fw, indent=4)


def main_wikitables():
    find_frequent_headers(['mongodb', 'wikiturlsnap', ['optitab.turl_training_set']], k=3, min_support=1000)


def main_gittables():
    find_frequent_headers(['mongodb', 'gittables', ['sloth.gittables']], k=3, min_support=1000)


def main_demo():
    find_frequent_headers(['mongodb', 'demo', ['sloth.demo']], k=3, min_support=50)



if __name__ == '__main__':
    # main_wikitables()
    # main_gittables()
    main_demo()
