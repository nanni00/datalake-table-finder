import os
import random
import re
import pandas as pd
from tqdm import tqdm
from tqdm import tqdm
import binascii
from collections import defaultdict
import multiprocessing as mp

from tools.josie import JosieDB
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.misc import is_valid_table, create_token_set, apply_sloth
from tools.utils.classes import ResultDatabase
from tools.utils.metrics import ndcg_at_p


get_result_ids = lambda s: list(map(int, re.findall(r'\d+', s)[::2]))
get_result_overlaps = lambda s: list(map(int, re.findall(r'\d+', s)[1::2]))

parse_results = lambda r: list(zip(get_result_ids(r), get_result_overlaps(r)))


def create_query_table(queries:list[int:set[int]], dbname, tables_prefix):    
    josiedb.open()
    josiedb.clear_query_table()
    for table_id, tokens_ids in queries:
        josiedb._dbconn.execute(f"INSERT INTO {tables_prefix}_queries VALUES ({table_id}, ARRAY[{','.join(map(str, tokens_ids))}]);")
    
    josiedb._dbconn.commit()
    GOPATH = os.environ['GOPATH']
    josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)
    
    # if cost sampling tables already exist we assume they are correct and won't recreate them
    sample_costs_tables_exist = josiedb.cost_tables_exist()
    josiedb.close()

    if not sample_costs_tables_exist:
        os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                    --pg-database={dbname} \
                    --test_tag={tables_prefix} \
                    --pg-table-queries={tables_prefix}_queries')

def query(results_file, k, results_directory, dbname, tables_prefix):
    # we are not considering the query preparation steps, since in some cases this will 
    # include also the cost sampling phase and in other cases it won't
    token_table_on_memory = False
    
    GOPATH = os.environ['GOPATH']
    josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)
    
    x = 'true' if token_table_on_memory else 'false'

    os.system(f'go run {josie_cmd_dir}/topk/main.go \
                --pg-database={dbname} \
                --test_tag={tables_prefix} \
                --outputDir={results_directory} \
                --resultsFile={results_file} \
                --useMemTokenTable={x} \
                --k={k} \
                --verbose=false')

def josie_multi_query(queries:list[int:set[int]], k, results_file, dbname, tables_prefix) -> list[tuple[int, list[tuple[int, int]]]]:
    results_directory = os.path.dirname(results_file)
    create_query_table(queries, dbname, tables_prefix)
    query(results_file, k, results_directory, dbname, tables_prefix)
    df = pd.read_csv(results_file)
    return [[row[0], parse_results(row[1])] for row in df[['query_id', 'results']].itertuples(index=False)]

def josie_single_query(set_id, token_set, k, results_file, dbname, tables_prefix):
    results_directory = os.path.dirname(results_file)
    create_query_table({set_id: token_set}, dbname, tables_prefix)
    query(results_file, k, results_directory, dbname, tables_prefix)
    return parse_results(pd.read_csv(results_file)['results'].values[0])

sim = lambda s, b: 1 - abs(s - b) / max(s, b)

def compute_largest_overlap_for_the_lists(id1, id2, dlh, resultsdb, bag_overlap):
    id1, id2 = (id1, id2) if id1 <= id2 else (id2, id1)
    largest_overlap = resultsdb.lookup_result_table(id1, id2)
    if largest_overlap:
        return largest_overlap
    tobj1 = dlh.get_table_by_numeric_id(id1)
    tobj2 = dlh.get_table_by_numeric_id(id2)
    largest_overlap = apply_sloth(tobj1['content'], tobj2['content'], tobj1['numeric_columns'], tobj2['numeric_columns'])[0]
    resultsdb.insert_results([[id1, id2, largest_overlap]])
    return -1 if largest_overlap == -1 or sim(largest_overlap, bag_overlap) < threhsold else largest_overlap

def task(data):
    qid, res_list = data
    dlh = SimpleDataLakeHelper(datalake_location, dataset, size)
    resultsdb = ResultDatabase(dbname, results_table)
    resultsdb.open()
    x = []
    for rid, bag_overlap in tqdm(res_list, leave=False, disable=False if os.getpid() % N == 0 else True):
        if largest_overlap := compute_largest_overlap_for_the_lists(qid, rid, dlh, resultsdb, bag_overlap) != -1:
            x.append([rid, bag_overlap, largest_overlap])
    return qid, x




test_name, datalake_location, dataset, size, mode = 'main', 'mongodb', 'wikiturlsnap', 'standard', 'bag'

dlh = SimpleDataLakeHelper(datalake_location, dataset, size)
ntables = dlh.get_number_of_tables()
dbname = 'nanni'
tables_prefix = f'{test_name}_d{dataset}_m{mode}'
results_table = f'results_d{dataset}_s{size}'
test_directory = ''
threshold = 0.7
k = 10
N = 500


josiedb = JosieDB(dbname, tables_prefix)

list_names = [['Team', 'Country'], ['Country', 'Location'], ['Athlete', 'Nationality'], ['Athlete', 'Nationality', 'Location']]

for i, names in enumerate(list_names):
    josie_results_file = '/results.csv'
    if not josiedb.is_open():
        josiedb.open()

    print(f' Working with names {names} '.center(60, '-'))
    queries = []
    ids = set()
    for table_obj in dlh.scan_tables(ignore_firsts=10**5):
        if is_valid_table(table_obj['content'], table_obj['numeric_columns']):
            tabset = set(create_token_set(table_obj['content'], 'set', table_obj['numeric_columns']))
            if sum(token in tabset for token in names) == len(names):
                queries.append(table_obj)
                # ids.add(id)
        if len(queries) >= N:
            break
        print(f'Found {len(queries)}/{N}', end='\r')

    print(f'Found {len(queries)}/{N}')

    query_columns = defaultdict(list)
    for q in queries:
        table, numeric_columns = q['content'], q['numeric_columns']
        table = [[row[i] for row in table] for i in range(len(table[0])) if numeric_columns[i] == 0]
        for column in table:
            if any(token in column for token in names):
                query_columns[q['_id_numeric']].append(column)

    print(f'Found {len(queries)} query tables')


    print('Apply JOSIE with single-column queries (baseline)')
    josiedb.open()
    single_column_bags = {qid: [create_token_set([column], 'bag', [0] * len(column)) for column in query_columns[qid]] for qid in query_columns.keys()}
    queries = []

    print('> Create query sets...')
    for qid, qbag_list in tqdm(single_column_bags.items(), leave=False):
        result = josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
        for qbag in qbag_list:
            integer_tokens = set()
            for id in result:
                raw_token = josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {id}").fetchone()[0]
                if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                    integer_tokens.add(id)
            queries.append([qid, integer_tokens])

    d = josie_multi_query(queries, k, results_file, dbname, tables_prefix)
    single_column_results = defaultdict(set)

    print('> Compute JOSIE overlaps, merge single column results...')
    for qid, josie_res in tqdm(d, leave=False):
        for jr in josie_res:
            single_column_results[qid].add(jr)
    print('> Sort ')
    for qid in single_column_results.keys():
        single_column_results[qid] = sorted(single_column_results[qid], key=lambda r: r[1], reverse=True)[:k]


    print('Compute largest overlap for single-column query results...')
    with mp.Pool(72) as pool:
        x = pool.map(task, single_column_results.items(), 1)
        for qid, res_list in x:
            result[qid] = res_list


    print('Filtering queries where results have all the same largest overlap (no tail)')
    for qid, res_list in single_column_results.items():
        if len(set(res_list)) == 1:
            del single_column_results[qid]
    print(f'Dropped {len(queries) - len(single_column_results)} queries')


    print('Create multi-column results (only for queries from the previous filtering)')
    multi_column_bags = {qid: create_token_set(columns, 'bag', [0] * len(columns[0])) for qid, columns in query_columns.items() if qid in single_column_results.keys()}
    queries = []
    for qid, qbag in tqdm(multi_column_bags.items(), leave=False):
        q = set()
        tokens = josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
        for id in tokens:
            raw_token = josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {id}").fetchone()[0]
            if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                q.add(id)
        queries.append([qid, q])

    multi_columns_results = dict(josie_multi_query(queries, k, results_file, dbname, tables_prefix))
    josiedb.close()




    print('Create silver standard...')
    silver_standard = defaultdict(list)
    for result in [multi_columns_results, single_column_results]:
        for qid, r in result.items():
            silver_standard[qid].extend(r)

    for qid in silver_standard.keys():
        silver_standard[qid] = sorted(silver_standard[qid], key=lambda t: t[2], reverse=True)[:k]


    print('Compute ndcg@10...')
    ndcg = []

    for qid, silstd in silver_standard.items():
        true_rel = [x[2] for x in silstd]
        p = k
        
        pred_rel = [x[2] for x in multi_columns_results[qid]]
        multi_col_ndcg, _ = ndcg_at_p(true_rel, pred_rel, p)

        pred_rel = [x[2] for x in single_column_results[qid]]
        single_col_ndcg, one_p = ndcg_at_p(true_rel, pred_rel, p)

        ndcg.append([qid, multi_col_ndcg, single_col_ndcg, k])

    pd.DataFrame(ndcg, columns=['qid', '2-col-ndcg', '1-col-ndcg', str(k)]).to_csv(f"/data4/nanni/tesi-magistrale/experiments/ndcg_res-{'-'.join(names)}.csv", index=False)
    print()
