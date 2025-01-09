import json
import os
from pprint import pprint
import random
import logging
from itertools import product
import multiprocessing as mp
from time import time

from datasketch.lshforest import MinHashLSHForest

from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from tqdm import tqdm

from dltf.utils.parallel import chunks
from dltf.utils.datalake import SimpleDataLakeHelper
from dltf.utils.misc import (
    is_valid_table, table_rows_to_columns, are_joinable_columns, create_minhash,
    whitespace_translator, lowercase_translator, punctuation_translator
)

info = lambda msg: logging.getLogger('TestLog').info(msg)



def task_columns(data):
    range_search, translators = data
    columns = []
    dlh = SimpleDataLakeHelper('mongodb', 'wikitables', 'standard')

    for tid in range_search:
        table_obj = dlh.get_table_by_numeric_id(tid)
        if not is_valid_table(table_obj['content'], table_obj['numeric_columns']): 
            continue
        for column in map(tuple, table_rows_to_columns(table_obj['content'], 0, len(table_obj['content'][0]), table_obj['numeric_columns'])):
            column = tuple(tokenize_column(column, *translators))
            if len(column) > 0:
                columns.append((column, create_minhash(column), hash(column)))
    return columns


def task_check_join(data):
    column_pairs, t, metric = data
    valid_pairs = set()
    for pair in column_pairs:
        if are_joinable_columns(*pair, t, metric):
            valid_pairs.add((' '.join(pair[0]), ' '.join(pair[1])))
    return valid_pairs



def extract_train_data(train_size:int=30000, t:float=0.8, metric='containment', translators:list=[]):
    num_workers = int(os.cpu_count() * 0.7)
    num_columns = 10 ** 6
    columns = []
    column_hashes = set()
    train_set = set()
    i = 0
    
    with mp.Pool(num_workers) as pool:
        while len(columns) < num_columns:
            work = range(i, i+num_workers*10)
            i += num_workers * 10
            for cols in pool.map(task_columns, chunks(work, len(work) // num_workers, translators)):
                for column, column_minhash, column_hash in cols:
                    if len(columns) >= num_columns:
                        break

                    if column_hash not in column_hashes:
                        columns.append([column, column_minhash])
                        column_hashes.add(column_hash)

            print(f"> Found {len(columns)}/{num_columns} columns ({round(100 * len(columns) / num_columns)}%)", end='\r')        
        print()

        info('Creating LSH Forest...')
        forest = MinHashLSHForest(l=16)
        for column, column_hash in tqdm(columns):
            forest.add(column, column_hash)
        info('> Indexing LSH Forest...')
        forest.index()

        step_work_size = 1000
        s = 0
        info('Searching candidate pairs...')
        while len(train_set) < train_size and s < len(columns):
            work = [[column, r] for column, column_hash in columns[s:s+step_work_size] for r in forest.query(column_hash, 10) if r != column]
            for rv in pool.map(task_check_join, chunks(work, len(work) // num_workers, t, metric)):
                for column_pair in rv:
                    if len(train_set) > train_size:
                        break
                    train_set.add(column_pair)
            s += step_work_size
            print(f"> Found {len(train_set)}/{train_size} pairs ({round(100 * len(train_set) / train_size)}%), scanned {s} pairs", end='\r')
        print()
    
    return list(train_set)



def create_inbatch_negatives(data:list[tuple[str, str]]):
    negatives = []
    for i, d in enumerate(data):
        while (j := random.randint(0, len(data) - 1)) == i:
            continue
        
        negatives.append([*d, data[j][1]])
    return negatives


def save_data_as_json(filepath:str, data):
    with open(filepath, 'w') as fw:
        json.dump(data, fw, indent=1)


def load_data(filepath):
    with open(filepath, 'r') as fr:
        return json.load(fr)


def triplets_to_three_dicts(triplets:list[tuple[str, str, str]]):
    return {
        'anchor': [t[0] for t in triplets],
        'positive': [t[1] for t in triplets],
        'negative': [t[2] for t in triplets]
    }





if __name__ == '__main__':
    dataset_size = 30_000
    train_frac = 0.95
    eval_frac = 0.025
    test_frac = 0.025

    assert train_frac + eval_frac + test_frac == 1

    # dataset_path = 'experiments/deepjoin/deepjoin_train.json'
    dataset_path = 'deepjoin_dataset.json'
    logfile = 'deepjoin.log'
    logging_setup(logfile)
    t = 0.7
    metric = 'jaccard'
    translators = [whitespace_translator, punctuation_translator, lowercase_translator]
    model = None

    create_dataset = True
    create_model = True

    # experiments with original wikitables dataset
    dlh = SimpleDataLakeHelper('mongodb', 'wikitables', 'standard')

    if create_dataset:
        info('Find positive random samples from data lake')
        # dataset = extract_random_positive_samples_from_datalake_p(dlh, dataset_size, t, metric=metric, translators=translators, verbose=True)
        dataset = extract_train_data(metric='jaccard', translators=translators)
        info('Create in-batch negative examples')
        dataset = create_inbatch_negatives(dataset)
        random.shuffle(dataset)
        dataset = {
            'train':    dataset[:int(len(dataset) * train_frac)], 
            'dev':      dataset[int(len(dataset) * train_frac):int(len(dataset) * (train_frac + eval_frac))], 
            'test':     dataset[int(len(dataset) * (train_frac + eval_frac)):]
            }
        info('Saving dataset as JSON')
        save_data_as_json(dataset_path, dataset)

    if create_model:
        model = SentenceTransformer("microsoft/mpnet-base", device='cuda')
        dataset = load_data(dataset_path)
        train_dataset = Dataset.from_dict(triplets_to_three_dicts(dataset['train']))
        eval_dataset =  Dataset.from_dict(triplets_to_three_dicts(dataset['dev']))
        test_dataset =  Dataset.from_dict(triplets_to_three_dicts(dataset['test']))

        info(f'Train size: {train_dataset.num_rows}')
        info(f'Evaluation size: {eval_dataset.num_rows}')
        info(f'Test size: {test_dataset.num_rows}')

        info('Initialising loss function...')
        loss = losses.MultipleNegativesRankingLoss(model)

        info('SentenceTransformerTrainingArguments init...')
        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir="models/mpnet-base-all-nli-triplet",
            # Optional training parameters:
            num_train_epochs=1,
            # per_device_train_batch_size=32,
            # per_device_eval_batch_size=32,
            # per_gpu_train_batch_size=32,
            # per_gpu_eval_batch_size=32,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=10000,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            logging_steps=100,
            run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
        )

        info('Init triplet evaluator...')
        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="all-nli-dev"
        )
        info('Evaluate basic model...')
        results = dev_evaluator(model)
        info('Evaluation results:')
        info(results)

        info('Training sentence transformer model...')
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        trainer.train()

        info('Testing model...')
        test_evaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            name="all-nli-test",
        )
        results = test_evaluator(model)
        info('Testing results:')
        info(results)
        info('Save model...')
        model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

    dlh.close()
