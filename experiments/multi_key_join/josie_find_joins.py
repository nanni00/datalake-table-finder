import os
import re
import logging
import binascii
import multiprocessing as mp
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import matplotlib.colors
from matplotlib import pyplot as plt

from tools.josie import JosieDB
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.misc import is_valid_table, create_token_set, apply_sloth, logging_setup
from tools.utils.classes import ResultDatabase
from tools.utils.metrics import ndcg_at_p, relevance_precision
from tools.utils.settings import DefaultPath as dp


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
    return -1 if largest_overlap == -1 or sim(largest_overlap, bag_overlap) < threshold else largest_overlap

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


# 1 calcolo con singole colonne
# 2 cercare coppie in cui l'overlap SLOTH e vicino a quello di JOSIE
# 2.1 controllare il discorso tail: se con k=10 tutti i risultati sono ottimi, magari occorre prendere k=20 per avere una coda di valori meno buoni da confrontare
# 2.2 considerare le coppie di query-result con differenza tra bag intersection e largest overlap bassa
# 3 calcolo multi column sulle query filtrate dal passaggio precedente
# 4 verifica quale dei due va meglio




test_name, datalake_location, dataset, size, mode = 'main', 'mongodb', 'wikiturlsnap', 'standard', 'bag'

dbname = 'nanni'
tables_prefix = f'{test_name}_d{dataset}_m{mode}'
results_table = f'results_d{dataset}_s{size}'
threshold = 0.7

k = 10
N = 500


josiedb = JosieDB(dbname, tables_prefix)
dlh = SimpleDataLakeHelper(datalake_location, dataset, size)
ntables = dlh.get_number_of_tables()

list_names = [['Team', 'Country'], ['Country', 'Location'], ['Athlete', 'Nationality'], ['Athlete', 'Nationality', 'Location']]
multi_key_join_directory =  f"{dp.root_project_path}/experiments/multi_key_join"
logfile =                   f"{multi_key_join_directory}/{N}.log"

for i, names in enumerate(list_names):
    test_directory =            f"{multi_key_join_directory}/{'-'.join(names)}/{N}"
    queries_file =              f"{test_directory}/queries.txt"
    josie_single_results_file = f"{test_directory}/results_single_josie.csv"
    josie_multi_results_file =  f"{test_directory}/results_multi_josie.csv"
    final_results_file =        f"{test_directory}/final_results.csv"
    final_results_plot =        f"{test_directory}/final_results.png"
    

    if not os.path.exists(test_directory):
        os.makedirs(test_directory)

    if i == 0: logging_setup(logfile)
    logging.getLogger('TestLog').info(f' Working with names {names} '.center(60, '-'))
    info = lambda msg: logging.getLogger('TestLog').info(msg)
    
    info(f"N = {N}, K = {k}, threshold = {threshold}")
    
    queries = []
    ids = set()
    
    if not os.path.exists(queries_file):
        for table_obj in dlh.scan_tables(ignore_firsts=10**5):
            if is_valid_table(table_obj['content'], table_obj['numeric_columns']):
                tabset = set(create_token_set(table_obj['content'], 'set', table_obj['numeric_columns']))
                if sum(token in tabset for token in names) == len(names):
                    queries.append(table_obj)
                    ids.add(table_obj['_id_numeric'])
            if len(queries) >= N:
                break
            print(f'Found {len(queries)}/{N}', end='\r')
        with open(queries_file, 'w') as fw:
            fw.writelines(map(lambda x: str(x) + '\n', ids))
    else:
        with open(queries_file) as fr:
            ids = fr.readlines()
        for tid in ids:
            table_obj = dlh.get_table_by_numeric_id(int(tid))
            queries.append(table_obj)
        
    query_columns = defaultdict(list)
    for q in queries:
        table, numeric_columns = q['content'], q['numeric_columns']
        table = [[row[i] for row in table] for i in range(len(table[0])) if numeric_columns[i] == 0]
        for column in table:
            if any(token in column for token in names):
                query_columns[q['_id_numeric']].append(column)

    info(f'Found {len(queries)}/{N} query tables')


    info('Apply JOSIE with single-column queries (baseline)')
    if not os.path.exists(josie_single_results_file):
        josiedb.open()
        single_column_bags = {qid: [create_token_set([column], 'bag', [0] * len(column)) for column in query_columns[qid]] for qid in query_columns.keys()}
        queries_set = []

        info('\tCreate query sets...')
        for qid, qbag_list in tqdm(single_column_bags.items(), leave=False):
            result = josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
            for qbag in qbag_list:
                integer_tokens = set()
                for tid in result:
                    raw_token = josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {tid}").fetchone()[0]
                    if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                        integer_tokens.add(tid)
                queries_set.append([qid, integer_tokens])

        info('\tCompute JOSIE overlaps...')
        d = josie_multi_query(queries_set, k, josie_single_results_file, dbname, tables_prefix)
        josiedb.close()
    else:
        d = [[row[0], parse_results(row[1])] for row in pd.read_csv(josie_single_results_file)[['query_id', 'results']].itertuples(index=False)]

    info('\tMerge distinct results for the same queries')
    # if JOSIE found for query 1234 the two results (444, 23) e (444, 5)
    # the the upper bound is set to 23+5=8
    single_column_results = defaultdict(lambda: defaultdict(int))
    for qid, josie_res in tqdm(d, leave=False):
        for jr in josie_res:
            single_column_results[qid][jr[0]] += jr[1]

    single_column_results = {qid: sorted([[rid, ub] for rid, ub in r.items()],
                                         key=lambda x: x[1], reverse=True)[:k] for qid, r in single_column_results.items()}

    info('\tCompute largest overlap for single-column query results...')
    result = defaultdict(list)
    with mp.Pool(72) as pool:
        x = pool.map(task, single_column_results.items(), 1)
        for qid, res_list in x:
            result[qid] = res_list
            
    info('\tFiltering queries where results have all the same largest overlap (no tail) or there are less than k values')
    to_drop = set()
    for qid, res_list in single_column_results.items():
        top = res_list[0][1]
        if len(res_list) < k or len(set(x[0] for x in res_list[1:])) <= 1 or res_list[-1][1] == top - 1:
            to_drop.add(qid)
    for qid in to_drop:
        del single_column_results[qid]
    info(f'\tDropped {len(queries) - len(single_column_results)} queries')


    info('Apply JOSIE with multi-column queries (our)')
    if not os.path.exists(josie_multi_results_file):
        info('\tCreate multi-column results (only for queries from the previous filtering)')
        josiedb.open()
        multi_column_bags = {qid: create_token_set(columns, 'bag', [0] * len(columns[0])) for qid, columns in query_columns.items() if qid in single_column_results.keys()}
        queries_set = []
        for qid, qbag in tqdm(multi_column_bags.items(), leave=False):
            q = set()
            tokens = josiedb._dbconn.execute(f"SELECT tokens FROM {josiedb._SET_TABLE_NAME} WHERE id = {qid}").fetchall()[0][0]
            for id in tokens:
                raw_token = josiedb._dbconn.execute(f"SELECT raw_token FROM {josiedb._INVERTED_LISTS_TABLE_NAME} WHERE token = {id}").fetchone()[0]
                if binascii.unhexlify(raw_token).decode('utf-8') in qbag:
                    q.add(id)
            queries_set.append([qid, q])

        info('\tCompute JOSIE overlaps...')
        multi_columns_results = dict(josie_multi_query(queries_set, k, josie_multi_results_file, dbname, tables_prefix))
        josiedb.close()
    else:
        multi_columns_results = dict([[row[0], parse_results(row[1])] for row in pd.read_csv(josie_multi_results_file)[['query_id', 'results']].itertuples(index=False)])

    info('\fFiltering columns with less than k values...')

    info('Create silver standard...')
    silver_standard = defaultdict(list)
    for result in [single_column_results, multi_columns_results]:
        for qid, r in result.items():
            silver_standard[qid].extend(r)

    for qid in silver_standard.keys():
        silver_standard[qid] = sorted(silver_standard[qid], key=lambda t: t[1], reverse=True)[:k]


    info('Compute final results...')
    ndcg = []

    for qid, silstd in silver_standard.items():
        true_rel = [x[1] for x in silstd]
        p = k
        
        pred_rel = [x[1] for x in single_column_results[qid]]
        single_col_ndcg, _ = ndcg_at_p(true_rel, pred_rel, p)
        single_rel_prec = relevance_precision(true_rel, pred_rel, p)
        
        pred_rel = [x[1] for x in multi_columns_results[qid]]
        multi_col_ndcg, _ = ndcg_at_p(true_rel, pred_rel, p)
        multi_rel_prec = relevance_precision(true_rel, pred_rel, p)

        ndcg.append([qid, single_col_ndcg, multi_col_ndcg, single_rel_prec, multi_rel_prec, k])

    results = pd.DataFrame(ndcg, columns=['qid', '1-col-ndcg', 'multi-col-ndcg', '1-col-RP', 'multi-col-RP', str(k)])
    results.to_csv(final_results_file, index=False)
    
    labels = results.columns[1:5]

    bplot = plt.boxplot(
        results.to_numpy()[:, 1:5],
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        showfliers=True,
        patch_artist=True
    )
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())[:len(bplot['boxes']) + 2]
    colors = colors[:1] + colors[3:] # just to drop green and orange, mean and median line aren't clear on it

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.suptitle(f"Results for {'-'.join(names)}; used {len(silver_standard)} query")
    plt.savefig(final_results_plot)
    plt.close()
    print()
