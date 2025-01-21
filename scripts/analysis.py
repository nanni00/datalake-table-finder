import os
import pickle
from time import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import polars as pl
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dltf.utils.metrics import *
from dltf.utils.misc import numerize
from dltf.utils.loghandler import logging_setup, info
from dltf.utils.settings import get_all_paths, ALGORITHM_MODE_CONFIG, DATALAKES




def worker_fp_per_query(inp):
    am, am_group = inp
    y = []
    for _, q_group in am_group.groupby(['query_id']):
        cnt = q_group[q_group['sloth_overlap'] == 0].shape[0]
        num_query_results = q_group.shape[0]
        
        y.append([am[0], am[1], cnt, cnt / num_query_results])
    return y


def worker_precision(inp):
    query_id, data, k_values, query_silver_standard = inp

    results = []
    true_relevances = [x[1] for x in query_silver_standard]    
    
    for (algorithm, mode), data in data.groupby(['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()    
        for k in k_values:
            if k > len(true_relevances):
                continue            
            rec = recall_at_k(true_relevances, result_relevances, k)
            prec = precision_at_k(true_relevances, result_relevances, k)
            rel_prec = relevance_precision_at_k(true_relevances, result_relevances, k)
            f1 = f_score(prec, rec)
            results.append([query_id, len(query_silver_standard), algorithm, mode, k, prec, rel_prec, rec, f1])
    return results


def worker_ndcg(inp):
    query_id, query_res, k_values, query_silver_standard = inp
    all_ndcg_results = []
    true_relevances = [x[1] for x in query_silver_standard]

    for (algorithm, mode), data in query_res.groupby(by=['algorithm', 'mode']):
        result_relevances = data['sloth_overlap'].values.tolist()
        for k in k_values:
            if len(true_relevances) == 0:
                all_ndcg_results.append([query_id, len(true_relevances), algorithm, mode, k, 0])
                continue
            # there could be errors in those cases where there isn't any actual relevant documents
            # i.e. all the retrivied documents doesn't have a non-zero SLOTH overlap                
            ndcg = ndcg_at_k(true_relevances, result_relevances, k)

            all_ndcg_results.append([query_id, len(true_relevances), algorithm, mode, k, ndcg])
    return all_ndcg_results




def analyses(test_name, k, num_query_samples, num_cpu, 
             datalake_name, 
             k_values, 
             save_silver_standard_to:str=None, load_silver_standard_from:str=None, *args, **kwargs):
    
    assert int(k) > 0
    assert int(num_cpu) > 0
    assert datalake_name in DATALAKES
    
    # General parameters
    test_name           = test_name.lower()
    k_search            = k
    q                   = numerize(num_query_samples)

    # Define paths and folders
    paths               = get_all_paths(test_name, datalake_name, k_search, num_query_samples)
    TEST_DATASET_DIR    = paths['TEST_DATASET_DIR']
    analyses_dir        = f'{TEST_DATASET_DIR}/results/analyses/k{k_search}_q{q}'
    
    # matplotlib graphics options
    alpha               = 1
    showfliers          = False

    # If the analysis directory doesn't exist, create it
    if not os.path.exists(analyses_dir):
        os.makedirs(analyses_dir)
    
    # Logging just on stdout
    logging_setup(on_stdout=True)

    info(f' {test_name.upper()} - {datalake_name.upper()} - ANALYSES - {k_search} - {q} '.center(150, '-'))

    all_colors = colors = list(mcolors.TABLEAU_COLORS.keys())
    methods = ALGORITHM_MODE_CONFIG
    markers = {m: 'o' if m[0] == 'josie' else 'x' if m[0] == 'lshforest' else 'd' for m in methods}
    methods = {m: c for m, c in zip(methods, colors[:len(methods)])}

    results = pl.read_csv(f"{paths['results_extr_dir']}/k{k_search}_q{q}.csv")
    results = results.filter(pl.struct(['algorithm', 'mode']).is_in(list(map(lambda am: {'algorithm': am[0], 'mode': am[1]}, methods.keys()))))

    results = results.drop_nulls()
    # This should no longer happen
    # results = results.filter(pl.col('sloth_overlap') != -1) # pairs that have had a SLOTH failure while computing the overlap

    results = results.with_columns((pl.col('algorithm_overlap') - pl.col('sloth_overlap')).alias('difference_overlap'))

    info(f'Filtering those groups where any method has returned less than K={k_search} values...')
    start_filtering = time()
    bad_groups = []
    for query_id, q_group in tqdm(results.to_pandas().groupby('query_id'), total=results.select('query_id').unique().shape[0], leave=False):
        for _, data in q_group.groupby(['algorithm', 'mode']):
            if data.shape[0] < k_search:
                bad_groups.append(query_id)
                break
    results = results.filter(~pl.col('query_id').is_in(bad_groups))
    end_filtering = time()
    
    info(f'Filtered {len(bad_groups)} groups in {round(end_filtering - start_filtering, 3)}s')


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
    fig.suptitle(f'Algorithm and Real Overlap comparison for dataset {datalake_name}')

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
    
    info(f'Total results: {results.shape[0]}; results with positive overlaps: {positive_overlaps.shape[0]}')
    
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
    fig.suptitle(f'Algorithm and Real Overlap Normalizes comparison for dataset {datalake_name}')

    plt.legend()
    plt.savefig(analyses_dir + '/graph_ratio_norm.png', bbox_inches='tight')
    plt.close()

    ###################################################################
    ##################### Create Silver Standards #####################
    ###################################################################
    
    if load_silver_standard_from:
        info(f'Loading silver standard from {load_silver_standard_from}...')
        with open(load_silver_standard_from, 'rb') as fp:
            silver_standard = pickle.load(fp)
    else:
        info('Creating Silver Standards...')
        
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
        for query_id, ids_overlaps in tqdm(results_ids, total=nqueries, leave=False):
            if ids_overlaps.shape[0] < max(k_values):
                continue
            s = set()
            s.update(map(tuple, ids_overlaps.to_numpy()[:, 1:].tolist()))
            silver_standard[query_id[0]] = sorted([x for x in list(s) if x[1] > 0], key=lambda x: x[1], reverse=True)
        end_ss = time()

        info(f'Finished. Total time: {round(end_ss - start_ss, 3)}s')

    if save_silver_standard_to:
        info(f'Saving silver standard to {save_silver_standard_to}...')
        with open(save_silver_standard_to, 'wb') as wp:
            pickle.dump(silver_standard, wp)

    ##########################################################
    ############### Precision and Recall at K ################
    ##########################################################

    query_groups = results.select('query_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)

    # Parallel version
    with mp.get_context('spawn').Pool(processes=num_cpu) as pool:
        info('Computing precision@K...')
        start_prec = time()
        prec_rec_results = pool.map(
            worker_precision, 
            ((name, data, k_values, silver_standard[name]) for name, data in query_groups))
        end_prec = time()
        info(f'Finished. Total time: {round(end_prec - start_prec, 3)}s')

    prec_rec_results = [x for qres in prec_rec_results for x in qres]
    prec_rec_results = pl.DataFrame(
        prec_rec_results, 
        schema=['query_id', 'silver_std_size', 'algorithm', 'mode', 'k', 'precision', 'RP', 'recall', 'f1'],
        orient='row'
    ).to_pandas()

    res_pivot = prec_rec_results.pivot_table(
        values=['precision', 'RP', 'recall', 'f1'], 
        index=['algorithm', 'mode'], 
        columns=['k'], 
        aggfunc=['mean', 'std', 'max'])
    res_pivot.to_csv(analyses_dir + f'/precision_recall@K.csv')

    # basic plot
    # for measure in ['precision', 'precision_v2', 'recall', 'f1']:
    measures = ['RP']
    for measure in measures:
        for row, label in zip(res_pivot['mean', measure].values, res_pivot.index):
            plt.plot(k_values, row, f'{markers[(label)]}-', label=f'{label[0]}-{label[1]}', color=methods[(label[0], label[1])])
        plt.xticks(k_values, k_values)
        plt.xlabel('K')
        plt.ylabel(f'mean {measure}@K')
        plt.title(f"{measure}@K graph for dataset {datalake_name}")
        plt.legend()
        plt.grid()
        plt.savefig(f'{analyses_dir}/graph_{measure}@k.png', bbox_inches='tight')
        plt.close()

    # boxplots with precision@p
    data = [(amp[0], amp[1], amp[2], group) for amp, group in prec_rec_results.groupby(by=['algorithm', 'mode', 'k'])]
    for measure in measures:
        fig, axes = plt.subplots(1, len(k_values), figsize=(15, 7), sharey=True)
        for i, k in enumerate(k_values):
            labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == k]
            colors = [methods[(d[0], d[1])] for d in data if d[2] == k]

            bplot = axes[i].boxplot([d[3][measure] for d in data if d[2] == k],
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

            axes[i].set_title(f'K = {k}')
            axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45)
        fig.savefig(f'{analyses_dir}/boxplot_{measure}@k.png', bbox_inches='tight')
        plt.close()


    ###########################################################################
    ########### Normalised Discounted Cumulative Gain at K (nDCG@K) ###########
    ###########################################################################

    query_groups = results.select('query_id', 'result_id', 'algorithm', 'mode', 'sloth_overlap').to_pandas().groupby("query_id", group_keys=True)
    work = ((query_id, data, k_values, silver_standard[query_id]) for query_id, data in query_groups)

    with mp.get_context('spawn').Pool(num_cpu) as pool:
        info('Computing nDCG@K...')
        start_ndcg = time()
        chunk_size = max(results.select('query_id').unique().shape[0] // num_cpu, 1)
        
        ndcg_results = pool.map(worker_ndcg, work, chunksize=chunk_size)
        end_ndcg = time()
        info(f'Finished. Total time: {round(end_ndcg - start_ndcg, 3)}s')
        try:
            ndcg_results = [x for qres in ndcg_results for x in qres]
        except:
            print(ndcg_results)

    df = pl.DataFrame(ndcg_results, schema=['query_id', 'silver_standard_size', 'algorithm', 'mode', 'k', 'ndcg@k'], orient='row')

    # consider only those groups that have more than X elements
    silver_standard_size_threshold = max(k_values)
    df_thr = df.filter(pl.col('silver_standard_size') >= silver_standard_size_threshold).to_pandas()

    ndcg_pivot = df_thr.pivot_table(
        index=['algorithm', 'mode'], 
        columns=['k'], 
        values=['ndcg@k'], 
        aggfunc=['mean', 'max']
    ).convert_dtypes()
    ndcg_pivot.to_csv(analyses_dir + f'/ndcg@k.csv')

    for row, label in zip(ndcg_pivot['mean', 'ndcg@k'].values, ndcg_pivot.index):
        plt.plot(k_values, row, f'{markers[(label)]}-', label=f'{label[0]}-{label[1]}', color=methods[(label[0], label[1])])
    plt.xticks(k_values, k_values)
    plt.xlabel("K")
    plt.ylabel("mean nDCG@K")

    plt.title(f"nDCG@K graph for dataset {datalake_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f'{analyses_dir}/graph_ndcg@k.png', bbox_inches='tight')
    plt.close()
    
    # boxplots with nDCG@K
    data = [(amp[0], amp[1], amp[2], group) for amp, group in df.group_by('algorithm', 'mode', 'k')]
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 7), sharey=True)
    for i, k in enumerate(k_values):
        labels = [f'{d[0]}\n{d[1]}' for d in data if d[2] == k]
        colors = [methods[(d[0], d[1])] for d in data if d[2] == k]

        bplot = axes[i].boxplot([d[3]['ndcg@k'] for d in data if d[2] == k],
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

        axes[i].set_title(f'K = {k}')
        axes[i].set_xticks(axes[i].get_xticks(), axes[i].get_xticklabels(), rotation=45)

    fig.savefig(f'{analyses_dir}/boxplot_ndcg@k.png', bbox_inches='tight')
    plt.close()
