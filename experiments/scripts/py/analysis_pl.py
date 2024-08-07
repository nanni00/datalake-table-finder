import os
import sys
import logging
import argparse
from time import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
from numerize_denumerize.numerize import numerize

from tools.utils.settings import DefaultPath as defpath
from tools.utils.utils import get_local_time, logging_setup
from tools.utils.parallel_worker import worker_fp_per_query, worker_ndcg, worker_precision
from tools.utils.mongodb_utils import get_mongodb_collections



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-name', required=True, type=str, help='a user defined test name, used instead of the default one m<mode>')
    parser.add_argument('--num-cpu', 
                        type=int, required=False, default=min(os.cpu_count(), 72),
                        help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
    parser.add_argument('--num-query-samples',
                        type=int, required=False, default=1000,
                        help='extract results only for the given result set size (e.g. 1000)')
    parser.add_argument('--dataset', 
                        required=True, choices=['wikipedia', 'gittables'])
    parser.add_argument('--size', 
                        required=False, default='standard', choices=['small', 'standard'],
                        help='works on small collection versions (only for testing)')
        
    args = parser.parse_args()
    test_name =         args.test_name
    nsamples =          args.num_query_samples
    num_cpu =           args.num_cpu
    dataset =           args.dataset
    size =             args.size
    
    test_name = test_name.lower()
    q = numerize(nsamples, asint=True)

    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=size)

    ROOT_TEST_DIR =         defpath.data_path.tests + f'/{test_name}'
    TEST_DATASET_DIR =      ROOT_TEST_DIR + f'/{dataset}'
    results_extr_dir =      TEST_DATASET_DIR + '/results/extracted'
    logfile =               TEST_DATASET_DIR + '/logging.log'
    
    analyses_dir =          TEST_DATASET_DIR + '/results/analyses'
    analyses_query_dir =    analyses_dir + f'/{q}'
    
    statistics_dir =        TEST_DATASET_DIR  + '/statistics'
    runtime_stat_file =     statistics_dir + '/runtime.csv'     

    runtime_metrics = []

    if not os.path.exists(analyses_dir):
        os.mkdir(analyses_dir)
    
    if not os.path.exists(analyses_query_dir):
        os.mkdir(analyses_query_dir)
    
    analyses_dir = analyses_query_dir

    logging_setup(logfile)
    logging.info(f'{"#" * 10} {test_name.upper()} - {dataset.upper()} - {size.upper()} - ANALYSES {"#" * 10}')

    solvers = [('josie', 'set'), ('josie', 'bag'), ('lshforest', 'set'), ('lshforest', 'bag'), ('embedding', 'fasttext')]

    results = pl.read_csv(f'{results_extr_dir}/final_results_q{q}.csv')
    
    # results.dropna()
    results = results.drop_nulls() 

    # results = results[results['sloth_overlap'] != -1]
    results = results.filter(pl.col('sloth_overlap') != -1)

    # results['difference_overlap'] = results['algorithm_overlap'] - results['sloth_overlap']
    # results['algorithm_overlap_norm'] = results['algorithm_overlap'] / (results['sloth_overlap'] + 1)

    results = results.with_columns((pl.col('algorithm_overlap') - pl.col('sloth_overlap')).alias('difference_overlap'))
    results = results.with_columns((pl.col('algorithm_overlap') / (pl.col('sloth_overlap') + 1)).alias('algorithm_overlap_norm'))


    ##########################################################
    ##################### False positive #####################
    ##########################################################
    
    logging.info('Computing False Positives...')
    
    # False Positives per single query result
    start_zr = time()
    with mp.get_context('spawn').Pool(len(solvers)) as pool:
        r = pool.map(worker_fp_per_query, 
                     results.select(['algorithm', 'mode', 'query_id', 'result_id', 'sloth_overlap'])
                     .to_pandas().groupby(['algorithm', 'mode'], group_keys=True))
    
    x = pd.DataFrame([z for y in r for z in y], columns=['algorithm', 'mode', 'FP_count', 'FP_rate'])
    fp_per_query_pivot = x.pivot_table(values=['FP_rate'], index=['algorithm', 'mode'], aggfunc=['mean', 'std'])

    # False Positives per algorithm-mode
    z = []
    for am, am_group in results.select(['algorithm', 'mode', 'query_id', 'result_id', 'sloth_overlap']).group_by('algorithm', 'mode'):
        false_positives = am_group.filter(pl.col('sloth_overlap') == 0).shape[0]
        ntot = am_group.shape[0]
        z.append((am[0], am[1], false_positives, ntot, round(false_positives / ntot, 5)))
    fp_per_algmode = pd.DataFrame(z, columns=['algorithm', 'mode', 'FP_count', 'total_results', 'FP_rate'])
    end_zr = time()

    # Saving the results
    logging.info(f'Finished. Total time: {round(end_zr - start_zr, 3)}s')
    runtime_metrics.append((get_local_time(), 'false_positives', round(end_zr-start_zr, 3)))
    fp_per_query_pivot.to_csv(analyses_dir + f'/false_positives_per_group_q{q}.csv')
    fp_per_algmode.to_csv(analyses_dir + f'/false_positives_per_alg_mode_q{q}.csv')
    



    #############################################################################
    ##################### Algorithm Overlap VS Real Overlap #####################
    #############################################################################
    data = [(am[0], am[1], group) for am, group in results.group_by('algorithm', 'mode')]

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = -500, 505, 25

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
    fig.suptitle(f'Algorithm and Real Overlap comparison for dataset {dataset}')

    plt.legend()
    plt.savefig(analyses_dir + '/graph_difference.png', bbox_inches='tight')
    plt.close()


    ##########################################################################################
    ##################### Algorithm Overlap VS Real Overlap (Normalised) #####################
    ##########################################################################################

    # In this graph it may seems that JOSIE-bag underestimate the overlap, but this is due to the +1;
    # TODO handle this in some better way?
    data = [(am[0], am[1], group) for am, group in results.group_by('algorithm', 'mode')]

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = 0, 20.04, 0.5

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
    fig.suptitle(f'Algorithm and Real Overlap Normalizes comparison for dataset {dataset}')

    plt.legend()
    plt.savefig(analyses_dir + '/graph_ratio_norm.png', bbox_inches='tight')
    plt.close()


    ###################################################################
    ##################### Create Silver Standards #####################
    ###################################################################

    logging.info('Creating Silver Standards...')
    
    start_ss = time()
    silver_standard = defaultdict(set)
    nqueries = results.select('query_id').unique().shape[0]
    results_ids = results.select(['query_id', 'result_id', 'sloth_overlap']).unique().group_by(['query_id'])

    # for each query, create its Silver Standard:
    # take all the result IDs from all the methods, then create a sorted list 
    # with pairs <result_ID, sloth_overlap>, taking only those pair with sloth_overlap>0,
    # since they are actually relevant
    for query_id, ids_overlaps in tqdm(results_ids, total=nqueries):
        s = set()
        s.update(map(tuple, ids_overlaps.to_numpy()[:, 1:].tolist()))
        silver_standard[query_id[0]] = sorted([x for x in list(s) if x[1] > 0], key=lambda x: x[1], reverse=True)
    end_ss = time()

    logging.info(f'Finished. Total time: {round(end_ss - start_ss, 3)}s')
    runtime_metrics.append((get_local_time(), 'create_silver_standard', round(end_ss - start_ss, 3)))


    ##########################################################
    ##################### Precision at K #####################
    ##########################################################


    p_values = [1, 3, 5, 10]
    precision_at_p_results = []
    query_groups = results.select('query_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)

    # Parallel version needed for large query sets
    with mp.Pool(processes=num_cpu) as pool:
        logging.info('Computing precision@p...')
        start_prec = time()
        precision_at_p_results = pool.map(
            worker_precision, 
            ((name, data, p_values, silver_standard[name]) for name, data in query_groups), 
        )
    
        end_prec = time()
        logging.info(f'Finished. Total time: {round(end_prec - start_prec, 3)}s')

    precision_at_p_results = [x for qres in precision_at_p_results for x in qres]
    runtime_metrics.append((get_local_time(), 'precision', round(end_prec - start_prec, 3)))

    columns = ['query_id', 'silver_std_size', 'algorithm', 'mode', 'p', 'precision_at_p']

    precision_at_p_results = pd.DataFrame(precision_at_p_results, columns=columns)

    patp_pivot = pd.pivot_table(precision_at_p_results, values=['precision_at_p'], index=['algorithm', 'mode'], columns=['p'], aggfunc=['mean', 'std', 'max'])
    patp_pivot.to_csv(analyses_dir + f'/precision@p_q{q}.csv')

    for row, label in zip(patp_pivot['mean', 'precision_at_p'].values, patp_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel('p')
    plt.ylabel('mean precision@P')
    plt.title(f"Precision@P graph for dataset {dataset}")
    plt.legend()
    plt.grid()
    plt.savefig(analyses_dir + f'/graph_precision@p_q{q}.png', bbox_inches='tight')
    plt.close()

    # scaling pivot table to [0, 1]
    scaled_patp_pivot = patp_pivot['mean']['precision_at_p'] / np.array([1, 3, 5, 10])
    scaled_patp_pivot.to_csv(analyses_dir + f'/precision@p_norm_q{q}.csv')

    for row, label in zip(scaled_patp_pivot.values, patp_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel('p')
    plt.ylabel('mean precision@P normalised')
    plt.grid()
    plt.title(f"Precision@P normalised graph for dataset {dataset}")
    plt.legend()
    plt.savefig(analyses_dir + f'/graph_precision@p_norm_q{q}.png', bbox_inches='tight')
    plt.close()

    # boxplots with precision@p
    precision_at_p_results['precision_norm'] = precision_at_p_results['precision_at_p'] / precision_at_p_results['p']
    data = [(amp[0], amp[1], amp[2], group) for amp, group in precision_at_p_results.groupby(by=['algorithm', 'mode', 'p'])]
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, p in enumerate(p_values):
        x = i % 4 # here assuming we are working just with [1, 3, 5, 10]
        x, y = x // 2, x % 2
        labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == p]
        colors = list(mcolors.TABLEAU_COLORS)[:len(labels)]

        bplot = axes[x][y].boxplot([d[3]['precision_norm'] for d in data if d[2] == p],
                    patch_artist=True,
                    labels=labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        axes[x][y].set_title(f'p = {p}')
    fig.savefig(analyses_dir + f'/graph_boxplot_precision@p_q{q}.png', bbox_inches='tight')
    plt.close()

    

    ###########################################################################
    ########### Normalised Discounted Cumulative Gain at P (nDCG@p) ###########
    ###########################################################################

    query_groups = results.select('query_id', 'result_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)
    work = ((name, data, p_values, silver_standard[name]) for name, data in query_groups)

    with mp.Pool(num_cpu) as pool:
        logging.info('Computing nDCG@p...')
        start_ndcg = time()
        ndcg_results = pool.map(worker_ndcg, work, chunksize=results.select('query_id').unique().shape[0] // num_cpu)
        end_ndcg = time()
        logging.info(f'Finished. Total time: {round(end_ndcg - start_ndcg, 3)}s')
        ndcg_results = [x for qres in ndcg_results for x in qres]
    runtime_metrics.append([get_local_time(), 'ndcg', round(end_ndcg - start_ndcg, 3)])

    df = pd.DataFrame(ndcg_results, columns=['query_id', 'silver_standard_size', 'algorithm', 'mode', 'p', 'missing_p', 'ndcg_p'])

    # consider only those groups that have more than X elements
    silver_standard_size_threshold = 0
    df_thr = df[df['silver_standard_size'] >= silver_standard_size_threshold]

    ndcg_pivot = df_thr.pivot_table(index=['algorithm', 'mode'], columns=['p'], values=['ndcg_p', 'missing_p'], aggfunc=['mean', 'max']).convert_dtypes()
    ndcg_pivot.to_csv(analyses_dir + f'/ndcg@p_q{q}.csv')

    for row, label in zip(ndcg_pivot['mean', 'ndcg_p'].values, ndcg_pivot.index):
        plt.plot([1, 3, 5, 10], row, 'o-', label=f'{label[0]}-{label[1]}')
    plt.xticks([1, 3, 5, 10], [1, 3, 5, 10])
    plt.xlabel("p")
    plt.ylabel("mean nDCG@P")

    plt.title(f"nDCG@P graph for dataset {dataset}")
    plt.legend()
    plt.grid()
    plt.savefig(analyses_dir + f'/graph_ndcg@p_q{q}.png', bbox_inches='tight')
    plt.close()
    
    # boxplots with nDCG@p
    data = [(amp[0], amp[1], amp[2], group) for amp, group in df.groupby(by=['algorithm', 'mode', 'p'])]
    
    import matplotlib.colors as mcolors
    
    # here assuming we are working just with [1, 3, 5, 10]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, p in enumerate(p_values):
        x = i % 4
        x, y = x // 2, x % 2
        labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == p]
        colors = list(mcolors.TABLEAU_COLORS)[:len(labels)]

        bplot = axes[x][y].boxplot([d[3]['ndcg_p'] for d in data if d[2] == p],
                    patch_artist=True,
                    labels=labels)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        axes[x][y].set_title(f'p = {p}')
    fig.savefig(analyses_dir + f'/graph_boxplot_ndcg@p_q{q}.png', bbox_inches='tight')
    plt.close()



    # Saving time statistics
    add_header = not os.path.exists(runtime_stat_file)
    with open(runtime_stat_file, 'a') as rfw:
        if add_header:
            rfw.write("local_time,algorithm,mode,task,time(s)\n")

        for (t_loctime, t_task, t_time) in runtime_metrics:
            rfw.write(f"{t_loctime},analysis,,{t_task}_{q},{t_time}\n")



