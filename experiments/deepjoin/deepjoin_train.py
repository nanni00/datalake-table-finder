import json
import random
from itertools import product
import multiprocessing as mp
from time import time

from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers

from tools.utils.deepjoin_utils import *
from tools.utils.parallel_worker import chunks
from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.misc import is_valid_table



def task(data):
    candidates, t, metric, translators = data
    train_data = []

    dlh = SimpleDataLakeHelper('mongodb', 'wikiturlsnap', 'standard')

    for r_id, s_id in candidates:
        r_table_obj = dlh.get_table_by_numeric_id(r_id)
        if not is_valid_table(r_table_obj['content'], r_table_obj['numeric_columns']):
            continue

        s_table_obj = dlh.get_table_by_numeric_id(s_id)
        if not is_valid_table(s_table_obj['content'], s_table_obj['numeric_columns']):
            continue

        r_columns = table_rows_to_columns(r_table_obj['content'], 0, len(r_table_obj['content'][0]), r_table_obj['numeric_columns'])
        s_columns = table_rows_to_columns(s_table_obj['content'], 0, len(s_table_obj['content'][0]), s_table_obj['numeric_columns'])

        for r_col, s_col in product(r_columns, s_columns):
            r_tok_col, s_tok_col = tokenize_column(r_col, *translators), tokenize_column(s_col, *translators)
            if are_joinable_columns(r_tok_col, s_tok_col, t, metric):
                train_data.append([' '.join(r_tok_col), ' '.join(s_tok_col)])
    dlh.close()
    return train_data


def extract_random_positive_samples_from_datalake_p(dlh:SimpleDataLakeHelper, train_size:int=500, t:float=0.8, metric='containment', translators:list=[], verbose=False):
    train_data = []
    datalake_size = dlh.get_number_of_tables()
    id_pairs = set()
    train_data = []
    start = time()

    while len(train_data) < train_size:
        search_step_size = train_size * 10
        candidates = set(map(tuple, [sorted(random.sample(range(0, datalake_size - 1), 2)) for _ in range(search_step_size)])).difference(id_pairs)
        for c in candidates: id_pairs.add(c)

        with mp.Pool(72) as pool:
            for rv in pool.map(task, chunks(list(candidates), len(candidates) // 72, t, metric, translators)):
                if len(rv) > 0:
                    train_data += rv
            if len(train_data) > 0:
                d_per_t = len(train_data) / (time() - start)
                eta = round((train_size - len(train_data)) / d_per_t, 3)
                print(f"Found {len(train_data)}/{train_size} samples\t({100 * len(train_data) / train_size}%)\teta: {eta}s".rjust(50), end='\r')        
    return train_data


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
    dataset_size = 1000
    train_frac = 0.6
    eval_frac = 0.2
    test_frac = 0.2
    assert train_frac + eval_frac + test_frac == 1

    # dataset_path = 'experiments/deepjoin/deepjoin_train.json'
    dataset_path = 'deepjoin_dataset.json'

    t = 0.7
    metric = 'containment'
    translators = [whitespace_translator, punctuation_translator, lowercase_translator]
    model = None

    create_dataset = False
    create_model = False

    dlh = SimpleDataLakeHelper('mongodb', 'wikiturlsnap', 'standard')

    if create_dataset:
        dataset = extract_random_positive_samples_from_datalake_p(dlh, dataset_size, t, translators, True)
        dataset = create_inbatch_negatives(dataset)
        random.shuffle(dataset)
        dataset = {
            'train':    dataset[:int(len(dataset) * train_frac)], 
            'dev':      dataset[int(len(dataset) * train_frac):int(len(dataset) * (train_frac + eval_frac))], 
            'test':     dataset[int(len(dataset) * (train_frac + eval_frac)):]
            }
        save_data_as_json(dataset_path, dataset)

    if create_model:
        model = SentenceTransformer("microsoft/mpnet-base")
        dataset = load_data(dataset_path)
        train_dataset = Dataset.from_dict(triplets_to_three_dicts(dataset['train']))
        eval_dataset =  Dataset.from_dict(triplets_to_three_dicts(dataset['dev']))
        test_dataset =  Dataset.from_dict(triplets_to_three_dicts(dataset['test']))

        print(f'Train size: {train_dataset.num_rows}')
        print(f'Evaluation size: {eval_dataset.num_rows}')
        print(f'Test size: {test_dataset.num_rows}')

        loss = losses.MultipleNegativesRankingLoss(model)


        args = SentenceTransformerTrainingArguments(
            # Required parameter:
            output_dir="models/mpnet-base-all-nli-triplet",
            # Optional training parameters:
            num_train_epochs=2,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
            bf16=False,  # Set to True if you have a GPU that supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
            # Optional tracking/debugging parameters:
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            logging_steps=100,
            run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
        )

        dev_evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
            name="all-nli-dev"
        )
        dev_evaluator(model)

        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
        )
        trainer.train()

        test_evaluator = TripletEvaluator(
            anchors=test_dataset["anchor"],
            positives=test_dataset["positive"],
            negatives=test_dataset["negative"],
            name="all-nli-test",
        )
        test_evaluator(model)

        model.save_pretrained("models/mpnet-base-all-nli-triplet/final")

    dlh.close()
