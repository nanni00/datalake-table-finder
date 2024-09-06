import os
import pickle
import logging
from time import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numerize_denumerize.numerize import numerize

from tools.utils import basicconfig
from tools.utils.settings import make_parser, get_all_paths
from tools.utils.misc import get_local_time, logging_setup
from tools.utils.parallel_worker import worker_fp_per_query, worker_ndcg, worker_precision




def analyses(test_name, k, num_query_samples, num_cpu, dataset, size, p_values, 
             save_silver_standard_to:str=None, load_silver_standard_from:str=None, *args, **kwargs):
    assert int(k) > 0
    assert int(num_cpu) > 0
    assert dataset in basicconfig.DATASETS
    assert size in basicconfig.DATASETS_SIZES
    
    test_name = test_name.lower()
    q = numerize(num_query_samples, asint=True)

    TEST_DATASET_DIR, _, logfile, _, _, \
        _, results_extr_dir, \
            _, runtime_stat_file, _ = get_all_paths(test_name, dataset, k, num_query_samples)
    
    analyses_dir =          f'{TEST_DATASET_DIR}/results/analyses'
    analyses_query_dir =    f'{analyses_dir}/k{k}_q{q}'
    

    runtime_metrics = []

    if not os.path.exists(analyses_dir):
        os.mkdir(analyses_dir)
    
    if not os.path.exists(analyses_query_dir):
        os.mkdir(analyses_query_dir)
    
    analyses_dir = analyses_query_dir

    logging_setup(logfile)
    logging.getLogger('TestLog').info(f' {test_name.upper()} - {dataset.upper()} - {size.upper()} - ANALYSES - {k} - {q} '.center(150, '-'))

    all_colors = colors = list(mcolors.TABLEAU_COLORS.keys())
    methods = basicconfig.ALGORITHM_MODE_CONFIG
    markers = {m: 'o' if m[0] == 'josie' else 'x' if m[0] == 'lshforest' else 'd' for m in methods}
    methods = {m: c for m, c in zip(methods, colors[:len(methods)])}

    alpha = 1
    showfliers = False

    results = pl.read_csv(f'{results_extr_dir}/final_results_k{k}_q{q}.csv')
    results = results.filter(pl.struct(['algorithm', 'mode']).is_in(list(map(lambda am: {'algorithm': am[0], 'mode': am[1]}, methods.keys()))))

    results = results.drop_nulls() # queries without any results
    results = results.filter(pl.col('sloth_overlap') != -1) # pairs that have had a SLOTH failure while computing the overlap

    results = results.with_columns((pl.col('algorithm_overlap') - pl.col('sloth_overlap')).alias('difference_overlap'))
    # results = results.with_columns((pl.col('algorithm_overlap') / (pl.col('sloth_overlap') + 1)).alias('algorithm_overlap_norm'))

    logging.getLogger('TestLog').info(f'Filtering those groups where any method has returned less than K={k} values...')
    start_filtering = time()
    bad_groups = []
    for query_id, q_group in tqdm(results.to_pandas().groupby('query_id'), total=results.select('query_id').unique().shape[0]):
        for (alg, mode), data in q_group.groupby(['algorithm', 'mode']):
            if data.shape[0] < k:
                bad_groups.append(query_id)
                break
    results = results.filter(~pl.col('query_id').is_in(bad_groups))
    end_filtering = time()
    print(len(bad_groups), num_query_samples, results.select('query_id').unique().shape[0])
    
    runtime_metrics.append((get_local_time(), 'filtering_groups', round(end_filtering - start_filtering, 3)))
    logging.getLogger('TestLog').info(f'Filtered {len(bad_groups)} groups in {round(end_filtering - start_filtering, 3)}s')



    ##########################################################
    ##################### False positive #####################
    ##########################################################
    
    logging.getLogger('TestLog').info('Computing False Positives...')
    
    # False Positives per single query result
    start_zr = time()
    with mp.get_context('spawn').Pool(len(methods)) as pool:
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
    logging.getLogger('TestLog').info(f'Finished. Total time: {round(end_zr - start_zr, 3)}s')
    runtime_metrics.append((get_local_time(), 'false_positives', round(end_zr-start_zr, 3)))
    fp_per_query_pivot.to_csv(analyses_dir + f'/false_positives_per_group.csv')
    fp_per_algmode.to_csv(analyses_dir + f'/false_positives_per_alg_mode.csv')
    


    #############################################################################
    ##################### Algorithm Overlap VS Real Overlap #####################
    #############################################################################
    data = [(am[0], am[1], group) for am, group in results.group_by('algorithm', 'mode')]
    labels = [f'{a}-{m}' for a, m, _ in data]
    colors = [methods[(a, m)] for a, m, _ in data]

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = -500, 505, 25

    ax.hist([d[2]['difference_overlap'] for d in data], 
            bins=np.arange(xmin, xmax, step), alpha=alpha, 
            label=labels,
            color=colors,
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

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(15, 5))
    xmin, xmax, step = 0, 20.04, 0.5
    positive_overlaps = results.filter(pl.col('sloth_overlap') > 0).with_columns((pl.col('algorithm_overlap') / (pl.col('sloth_overlap'))).alias('algorithm_overlap_norm'))
    data = [(am[0], am[1], group) for am, group in positive_overlaps.group_by('algorithm', 'mode')]
    
    logging.getLogger('TestLog').info(f'Total results: {results.shape[0]}; results with positive overlaps: {positive_overlaps.shape[0]}')
    
    ax.hist([d[2]['algorithm_overlap_norm'] for d in data], 
            bins=np.arange(xmin, xmax, step), alpha=alpha, 
            label=labels, color=colors,
            align='mid')
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(np.arange(xmin, xmax, step))
    ax.grid()
    ax.set_xlabel(r'$\frac{overlap_{algoritmo}}{largestOverlap}$')
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
    
    if load_silver_standard_from:
        logging.getLogger('TestLog').info(f'Loading silver standard from {load_silver_standard_from}...')
        with open(load_silver_standard_from, 'rb') as fp:
            silver_standard = pickle.load(fp)
    else:
        logging.getLogger('TestLog').info('Creating Silver Standards...')
        
        start_ss = time()
        silver_standard = defaultdict(set)
        nqueries = results.select('query_id').unique().shape[0]
        results_ids = results.select(['query_id', 'result_id', 'sloth_overlap']).unique().group_by(['query_id'])

        # for each query, create its Silver Standard:
        # take all the result IDs from all the methods, then create a sorted list 
        # with pairs <result_ID, sloth_overlap>, taking only those pair with sloth_overlap>0,
        # because they are actually relevant
        # if a Silver Standard has less than the maximum P value, it isn't considered in the 
        # next analyses, just to avoid corner cases with groups with just N<<P values (however, this
        # shouldn't happen since we've already filtered on the returned list size of each algorithm...)
        for query_id, ids_overlaps in tqdm(results_ids, total=nqueries):
            if ids_overlaps.shape[0] < max(p_values):
                continue
            s = set()
            s.update(map(tuple, ids_overlaps.to_numpy()[:, 1:].tolist()))
            silver_standard[query_id[0]] = sorted([x for x in list(s) if x[1] > 0], key=lambda x: x[1], reverse=True)
        end_ss = time()

        logging.getLogger('TestLog').info(f'Finished. Total time: {round(end_ss - start_ss, 3)}s')
        runtime_metrics.append((get_local_time(), 'create_silver_standard', round(end_ss - start_ss, 3)))

    if save_silver_standard_to:
        logging.getLogger('TestLog').info(f'Saving silver standard to {save_silver_standard_to}...')
        with open(save_silver_standard_to, 'wb') as wp:
            pickle.dump(silver_standard, wp)

    ##########################################################
    ############### Precision and Recall at K ################
    ##########################################################

    query_groups = results.select('query_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)

    # Parallel version
    with mp.Pool(processes=num_cpu) as pool:
        logging.getLogger('TestLog').info('Computing precision@p...')
        start_prec = time()
        prec_rec_results = pool.map(
            worker_precision, 
            ((name, data, p_values, silver_standard[name]) for name, data in query_groups))
        end_prec = time()
        logging.getLogger('TestLog').info(f'Finished. Total time: {round(end_prec - start_prec, 3)}s')
    runtime_metrics.append((get_local_time(), 'prec-rec-f1', round(end_prec - start_prec, 3)))

    prec_rec_results = [x for qres in prec_rec_results for x in qres]
    prec_rec_results = pd.DataFrame(prec_rec_results, columns=['query_id', 'silver_std_size', 'algorithm', 'mode', 'p', 'precision', 'RP', 'recall', 'f1'])

    res_pivot = pd.pivot_table(prec_rec_results, values=['precision', 'RP', 'recall', 'f1'], index=['algorithm', 'mode'], columns=['p'], aggfunc=['mean', 'std', 'max'])
    res_pivot.to_csv(analyses_dir + f'/precision_recall@p.csv')

    # basic plot
    # for measure in ['precision', 'precision_v2', 'recall', 'f1']:
    measures = ['RP']
    for measure in measures:
        for row, label in zip(res_pivot['mean', measure].values, res_pivot.index):
            plt.plot(p_values, row, f'{markers[(label)]}-', label=f'{label[0]}-{label[1]}', color=methods[(label[0], label[1])])
        plt.xticks(p_values, p_values)
        plt.xlabel('P')
        plt.ylabel(f'mean {measure}@P')
        plt.title(f"{measure}@P graph for dataset {dataset}")
        plt.legend()
        plt.grid()
        plt.savefig(f'{analyses_dir}/graph_{measure}@p.png', bbox_inches='tight')
        plt.close()

    # boxplots with precision@p
    data = [(amp[0], amp[1], amp[2], group) for amp, group in prec_rec_results.groupby(by=['algorithm', 'mode', 'p'])]
    for measure in measures:
        fig, axes = plt.subplots(1, len(p_values), figsize=(15, 7), sharey=True)
        for i, p in enumerate(p_values):
            labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == p]
            colors = [methods[(d[0], d[1])] for d in data if d[2] == p]

            bplot = axes[i].boxplot([d[3][measure] for d in data if d[2] == p],
                        patch_artist=True,
                        showfliers=showfliers,
                        showmeans=True,
                        meanline=True,
                        labels=labels)
            
            for median in bplot['medians']:
                median.set_color(all_colors[-1])
            for median in bplot['means']:
                median.set_color(all_colors[-2])                

            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)

            axes[i].set_title(f'P = {p}')
            axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45)
        fig.savefig(f'{analyses_dir}/boxplot_{measure}@p.png', bbox_inches='tight')
        plt.close()


    ###########################################################################
    ########### Normalised Discounted Cumulative Gain at P (nDCG@p) ###########
    ###########################################################################

    query_groups = results.select('query_id', 'result_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)
    work = ((query_id, data, p_values, silver_standard[query_id]) for query_id, data in query_groups)

    with mp.Pool(num_cpu) as pool:
        logging.getLogger('TestLog').info('Computing nDCG@p...')
        start_ndcg = time()
        ndcg_results = pool.map(worker_ndcg, work, chunksize=results.select('query_id').unique().shape[0] // num_cpu)
        end_ndcg = time()
        logging.getLogger('TestLog').info(f'Finished. Total time: {round(end_ndcg - start_ndcg, 3)}s')
        ndcg_results = [x for qres in ndcg_results for x in qres]
    runtime_metrics.append([get_local_time(), 'ndcg', round(end_ndcg - start_ndcg, 3)])

    df = pd.DataFrame(ndcg_results, columns=['query_id', 'silver_standard_size', 'algorithm', 'mode', 'p', 'missing_p', 'ndcg_p'])

    # consider only those groups that have more than X elements
    silver_standard_size_threshold = max(p_values)
    df_thr = df[df['silver_standard_size'] >= silver_standard_size_threshold]

    ndcg_pivot = df_thr.pivot_table(index=['algorithm', 'mode'], columns=['p'], values=['ndcg_p', 'missing_p'], aggfunc=['mean', 'max']).convert_dtypes()
    ndcg_pivot.to_csv(analyses_dir + f'/ndcg@p.csv')

    for row, label in zip(ndcg_pivot['mean', 'ndcg_p'].values, ndcg_pivot.index):
        plt.plot(p_values, row, f'{markers[(label)]}-', label=f'{label[0]}-{label[1]}', color=methods[(label[0], label[1])])
    plt.xticks(p_values, p_values)
    plt.xlabel("P")
    plt.ylabel("mean nDCG@P")

    plt.title(f"nDCG@P graph for dataset {dataset}")
    plt.legend()
    plt.grid()
    plt.savefig(f'{analyses_dir}/graph_ndcg@p.png', bbox_inches='tight')
    plt.close()
    
    # boxplots with nDCG@p
    data = [(amp[0], amp[1], amp[2], group) for amp, group in df.groupby(by=['algorithm', 'mode', 'p'])]
    
    fig, axes = plt.subplots(1, len(p_values), figsize=(15, 7), sharey=True)
    for i, p in enumerate(p_values):
        labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == p]
        colors = [methods[(d[0], d[1])] for d in data if d[2] == p]

        bplot = axes[i].boxplot([d[3]['ndcg_p'] for d in data if d[2] == p],
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=showfliers,
                    labels=labels)

        for median in bplot['medians']:
            median.set_color(all_colors[-1])
        for median in bplot['means']:
            median.set_color(all_colors[-2])                

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)

        axes[i].set_title(f'P = {p}')
        axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45)

    fig.savefig(f'{analyses_dir}/boxplot_ndcg@p.png', bbox_inches='tight')
    plt.close()



    # Saving time statistics
    add_header = not os.path.exists(runtime_stat_file)
    with open(runtime_stat_file, 'a') as rfw:
        if add_header:
            rfw.write("local_time,algorithm,mode,task,k,num_queries,time(s)\n")

        for (t_loctime, t_task, t_time) in runtime_metrics:
            rfw.write(f"{t_loctime},analysis,,{t_task},{k},{q},{t_time}\n")





if __name__ == '__main__':
    args = make_parser('test_name', 'num_cpu', 'num_query_samples', 'k', 'dataset', 'size', 'p_values')
    test_name =         args.test_name
    num_query_samples = args.num_query_samples
    k =                 args.k
    num_cpu =           args.num_cpu
    dataset =           args.dataset
    size =              args.size
    p_values =          args.p_values


    analyses(test_name, k, num_query_samples, num_cpu, dataset, size, p_values)
