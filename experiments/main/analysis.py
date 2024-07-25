import os
import argparse
import statistics
from time import time
from math import log2
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numerize_denumerize.numerize import numerize

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_mongodb_collections, get_local_time



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
    parser.add_argument('--num-cpu', 
                        type=int, required=False, default=min(os.cpu_count(), 96),
                        help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
    parser.add_argument('--num-query-samples',
                        type=int, required=False, default=1000,
                        help='extract results only for the given result set size (e.g. 1000)')
    parser.add_argument('--small', 
                        required=False, action='store_true',
                        help='works on small collection versions (only for testing)')
        
    args = parser.parse_args()
    test_name =         args.test_name
    nsamples =          args.num_query_samples
    num_cpu =           args.num_cpu
    small =             args.small
    q = numerize(nsamples, asint=True)

    small = False
    mongoclient, collections = get_mongodb_collections(small)


    ROOT_TEST_DIR =         defpath.data_path.tests + '/' + test_name
    results_extr_dir =      ROOT_TEST_DIR + '/results/extracted'
    analyses_dir =          ROOT_TEST_DIR + '/results/analyses'
    analyses_query_dir =    analyses_dir + f'/{q}'
    statistics_dir =        ROOT_TEST_DIR  + '/statistics'

    runtime_stat_file =     statistics_dir + '/runtime.csv'     

    runtime_metrics = []

    if not os.path.exists(analyses_dir):
        os.mkdir(analyses_dir)
    
    if not os.path.exists(analyses_query_dir):
        os.mkdir(analyses_query_dir)
    
    analyses_dir = analyses_query_dir

    solvers = [('josie', 'set'), ('josie', 'bag'), ('lshforest', 'set'), ('lshforest', 'bag'), ('embedding', 'fasttext')]

    results = pd.read_csv(f'{results_extr_dir}/final_results_q{q}.csv')
    results = results.dropna()

    results['difference_overlap'] = results['algorithm_overlap'] - results['sloth_overlap']
    results['algorithm_overlap_norm'] = results['algorithm_overlap'] / (results['sloth_overlap'] + 1)



    ########## Zero Ratio ##########
    x = []
    print(get_local_time(), ' Computing Zero Ratio...')
    start_zr = time()
    for am, am_group in results.groupby(by=["algorithm", "mode"]):
        for query_id, q_group in am_group.groupby(by=["query_id"]):
            cnt = ((q_group['sloth_overlap'] == 0)).sum()
            num_query_results = q_group.count().values.tolist()[0]
            x.append([am[0], am[1], query_id[0], num_query_results, cnt, cnt / num_query_results])
    end_zr = time()
    print(get_local_time(), ' Finished. Total time: ', round(end_zr - start_zr, 3), 's')

    x = pd.DataFrame(x, columns=['algorithm', 'mode', 'query_id', 'query_size', 'zero_overlap_cnt', 'zero_overlap_ratio'])
    null_ratio_pivot = pd.pivot_table(x, values=['zero_overlap_ratio'], index=['algorithm', 'mode'], aggfunc=['mean', 'std', 'min', 'max'])
    null_ratio_pivot.to_csv(analyses_dir + f'/null_ratio_q{q}.csv')

    runtime_metrics.append((get_local_time(), 'zero_ratio', round(end_zr-start_zr, 3)))



    ########## Algorithm Overlap VS Real Overlap ##########
    data = [(am[0], am[1], group) for am, group in results.groupby(by=['algorithm', 'mode'])]

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = -200, 300, 10

    ax.hist([d[2]['difference_overlap'] for d in data], 
            bins=np.arange(xmin, xmax, step), alpha=0.8, 
            label=[f'{a}-{m}' for a, m, _ in data],
            align='mid')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax, step))
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)
    ax.grid()
    ax.set_xlabel('ALGORITHM overlap - SLOTH overlap')
    ax.set_ylabel('frequency')

    plt.legend()
    plt.savefig(analyses_dir + '/graph_difference.png')
    plt.close()



    ########## Algorithm Overlap VS Real Overlap (Normalised) ##########

    # In this graph it may seems that JOSIE-bag underestimate the overlap, but this is due to the +1;
    # TODO handle this in some better way?
    data = [(am[0], am[1], group) for am, group in results.groupby(by=['algorithm', 'mode'])]

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = 0, 10, 0.2

    ax.hist([d[2]['algorithm_overlap_norm'] for d in data], 
            bins=np.arange(xmin, xmax, step), alpha=0.8, 
            label=[f'{a}-{m}' for a, m, _ in data],
            align='mid')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax, step))
    ax.grid()
    ax.set_xlabel('ALGORITHM_overlap / (SLOTH_overlap + 1)')
    ax.set_ylabel('frequency')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)

    plt.legend()
    plt.savefig(analyses_dir + '/graph_difference_norm.png')
    plt.close()



    ########## Create Silver Standards ##########
    print(get_local_time(), ' Creating Silver Standards...')
    start_ss = time()
    silver_standard = defaultdict(set)
    results_ids = results.convert_dtypes().groupby(by='query_id')[['result_id', 'sloth_overlap']]

    for query_id, ids_overlaps in results_ids:
        for i in ids_overlaps.values:
            _id, _overlap = i
            silver_standard[query_id].add((_id, _overlap))

    for query_id in silver_standard.keys():
        silver_standard[query_id] = sorted(list(silver_standard[query_id]), key=lambda x: x[1], reverse=True)
    end_ss = time()
    print(get_local_time(), ' Finished. Total time: ', round(end_ss - start_ss, 3), 's')
    runtime_metrics.append((get_local_time(), 'create_silver_standard', round(end_ss - start_ss, 3)))



    ########### Precision at K ###########
    def worker_precision(query_id):
        global p
        qss = [x[1] for x in silver_standard[query_id]]
        prec_results = []

        try: avg_overlap = round(statistics.mean(qss), 3)
        except statistics.StatisticsError: avg_overlap = 0

        # here errors may be given by single-result queries; 
        # standard deviation cannot be computed for single values (very uncommon cases...)
        try: stdev_overlap = round(statistics.stdev(qss))
        except statistics.StatisticsError: stdev_overlap = 0
            
        for (algorithm, mode), data in results.groupby(by=["algorithm", "mode"]):
            ids = data[data['query_id'] == query_id]['result_id'].values.tolist()
            for _p in p:
                real_topk = [x[0] for x in silver_standard[query_id][:_p]]
                precision_at_k = set(real_topk).intersection(ids)
                prec_results.append([query_id, len(qss), avg_overlap, stdev_overlap, algorithm, mode, _p, len(precision_at_k)])
        return prec_results


    p = [1, 3, 5, 10]
    precision_at_k_results = []
    work = list(silver_standard.keys())

    # Parallel version needed for large query sets (>100K)
    with mp.Pool(processes=num_cpu) as pool:
        print(get_local_time(), ' Computing precision@p...')
        start_prec = time()
        precision_at_k_results = pool.map(worker_precision, work, chunksize=len(work) // num_cpu)
        end_prec = time()
        print(get_local_time(), ' Finished. Total time: ', round(end_prec - start_prec, 3), 's')
        precision_at_k_results = [x for qres in precision_at_k_results for x in qres]

    runtime_metrics.append((get_local_time(), 'precision', round(end_prec - start_prec, 3)))

    columns = [
        'query_id',
        'silver_std_size',
        'silver_std_ov_mean',
        'silver_std_ov_stdev',
        'algorithm',
        'mode',
        'k',
        'precision_at_k'
    ]

    precision_at_k_results = pd.DataFrame(precision_at_k_results, columns=columns)

    patk_pivot = pd.pivot_table(precision_at_k_results, values=['precision_at_k'], index=['algorithm', 'mode'], columns=['k'], aggfunc=['mean', 'std', 'max'])
    patk_pivot.to_csv(analyses_dir + f'/precision@p_q{q}.csv')

    for row, label in zip(patk_pivot['mean', 'precision_at_k'].values, patk_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel('p')
    plt.ylabel('mean precision@p')

    plt.legend()
    plt.savefig(analyses_dir + f'/graph_precision@p_q{q}.png')
    plt.close()

    # scaling pivot table to [0, 1]
    scaled_patk_pivot = patk_pivot['mean']['precision_at_k'] / np.array([1, 3, 5, 10])
    scaled_patk_pivot.to_csv(analyses_dir + f'/precision@p_norm_q{q}.csv')

    for row, label in zip(scaled_patk_pivot.values, patk_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel('p')
    plt.ylabel('mean precision@p normalised')

    plt.legend()
    plt.savefig(analyses_dir + f'/graph_precision@p_norm_q{q}.png')
    plt.close()



    ########### Normalised Discounted Cumulative Gain at P (nDCG@p) ###########

    def ndcg_at_p(true_relevances, scores, p):
        p = min(p, len(true_relevances), len(scores))
        if p <= 0: # because computing nDCG is meaningful only if there is more than one document 
            return 0, 1
        idcg = sum(rel / log2(i + 1) for i, rel in enumerate(true_relevances[:p], start=1))
        dcg = sum(rel / log2(i + 1) for i, rel in enumerate(scores[:p], start=1))
        if idcg < dcg:
            raise ZeroDivisionError()

        return dcg / idcg, p

    def worker_ndcg(query_id):
        global p
        ndcg_res = []
        true_relevances = [x[1] for x in silver_standard[query_id]]
        max_silver_standard = true_relevances[0]

        for (algorithm, mode), data in results.groupby(by=['algorithm', 'mode']):
            r = data[data['query_id'] == query_id][['result_id', 'sloth_overlap']]
            result_relevances = [min(max_silver_standard, x[1]) for x in r.values.tolist()]
            for _p in p:
                try: ndcg, _actual_p = ndcg_at_p(true_relevances, result_relevances, _p)
                except ZeroDivisionError: continue
                ndcg_res.append([query_id, len(true_relevances), algorithm, mode, _p, _p - _actual_p, ndcg])
        return ndcg_res


    # same work list of precision@p
    with mp.Pool(num_cpu) as pool:
        print(get_local_time(), ' Computing nDCG@p...')
        start_ndcg = time()
        ndcg_results = pool.map(worker_ndcg, work, chunksize=len(work) // num_cpu)
        end_ndcg = time()
        print(get_local_time(), ' Finished. Total time: ', round(end_ndcg - start_ndcg, 3), 's')
        ndcg_results = [x for qres in ndcg_results for x in qres]

    runtime_metrics.append([get_local_time(), 'ndcg', round(end_ndcg - start_ndcg, 3)])

    df = pd.DataFrame(ndcg_results, columns=['query_id', 'silver_standard_size', 'algorithm', 'mode', 'p', 'missing_p', 'ndcg_p'])

    # consider only those groups that have more than X elements
    silver_standard_size_threshold = 5
    df_thr = df[df['silver_standard_size'] >= silver_standard_size_threshold]

    ndcg_pivot = df_thr.pivot_table(index=['algorithm', 'mode'], columns=['p'], values=['ndcg_p', 'missing_p'], aggfunc=['mean', 'max']).convert_dtypes()
    ndcg_pivot.to_csv(analyses_dir + f'/ndcg@p_q{q}.csv')

    for row, label in zip(ndcg_pivot['mean', 'ndcg_p'].values, ndcg_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel("p")
    plt.ylabel("mean nDCG@p")

    plt.legend()
    plt.grid()
    plt.savefig(analyses_dir + f'/graph_ndcg@p_q{q}.png')
    plt.close()




    # Saving time statistics
    add_header = not os.path.exists(runtime_stat_file)
    with open(runtime_stat_file, 'a') as rfw:
        if add_header:
            rfw.write("local_time,algorithm,mode,task,time(s)\n")

        for (t_loctime, t_task, t_time) in runtime_metrics:
            rfw.write(f"{t_loctime},analysis,,{t_task}_{q},{t_time}\n")



