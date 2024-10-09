import os
import random
import argparse
import multiprocessing as mp

import pymongo
from tqdm import tqdm

from thesistools.sloth.utils import parse_table
from thesistools.utils.misc import naive_detect_bad_columns
from thesistools.utils.datalake import get_mongodb_collections
from thesistools.utils.basicconfig import DATALAKES, DATALAKE_SIZES


NUMERICAL_NER_TAGS = {
    # 'DATE', 
    # 'TIME', 
    # 'PERCENT', 
    # 'MONEY', 
    'QUANTITY', 
    'CARDINAL'
    }


def infer_column_type_from_sampled_value(sample_value_entry: list[list[str]]) -> str:
    # given the tags detected in the sampled cells, return 1 if this column is numeric,
    # i.e. if at least one cell has more numeric than other tags.
    cells = []

    for sampled_cell_tags in sample_value_entry:
        num_numerical_tags = 0
        num_other_tags = 0

        for tag in sampled_cell_tags:
            if tag in NUMERICAL_NER_TAGS:
                num_numerical_tags += 1
            else:
                num_other_tags += 1
        cells.append(1 if num_numerical_tags > num_other_tags else 0)
    return 1 if sum(cells) > 0 else 0


def spacy_detect_table_numeric_columns(table: list[list], nlp, nsamples: int) -> list[int]:
    """
    given a table, as a list of lists in a row view, returns a list
    of integer, where if the ith element is 1 this means that the ith column is
    numeric, 0 otherwise
    """
    return [
        infer_column_type_from_sampled_value(
            [
                [
                    token.ent_type_
                    for token in doc
                ]
                for doc in nlp.pipe(random.sample(column, min(nsamples if nsamples != -1 else len(table), len(table))))
            ]
        )
        for column in parse_table(table, len(table[0]), 0)
    ]





def worker(t: tuple[str, list[list]]):
    if mode == 'naive':
        return (t[0], naive_detect_bad_columns(t[1]))
    elif mode == 'spacy':
        return (t[0], spacy_detect_table_numeric_columns(t[1], nlp[os.getpid() % ncpu], nsamples))
    else:
        raise Exception('Unknown mode: ' + mode)


def init_pool(nlp_array, ncpu, nsamples, mode):
    global nlp
    nlp = nlp_array
    ncpu = ncpu
    nsamples = nsamples
    mode = mode


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', 
                        choices=['set', 'unset'], 
                        help='if "set", add a new field "numeric_columns" to each document, i.e. table,  \
                            an array of lenght #columns where if the ith value of the array is 1 then the \
                            ith column of the table is numeric, 0 otherwise. \
                            If "unset", delete the field "numeric_columns".')
    parser.add_argument('-m', '--mode', 
                        required=False, choices=['naive', 'spacy'], default='naive',
                        help='defines how to detect numeric columns')
    parser.add_argument('-w', '--num-cpu', 
                        type=int, required=False, default=min(os.cpu_count(), 64),
                        help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
    parser.add_argument('-s', '--sample-size', 
                        type=int, required=False, default=3,
                        help='defines how many cell(s) must be sampled in spacy mode from each column to detect wheter or not the column is numeric, \
                            default is 3. If set to -1 analyses the whole column.')
    parser.add_argument('--dataset', 
                        required=True, choices=DATALAKES)
    parser.add_argument('--size',
                        required=False, default='standard', choices=DATALAKE_SIZES,
                        help='works on small collection versions (only for testing)')

    args = parser.parse_args()
    mode = args.mode
    task = args.task
    ncpu = args.num_cpu
    nsamples = args.sample_size
    dataset = args.dataset
    size = args.size

    if os.cpu_count() < ncpu:
        print(f"Using {os.cpu_count()} cores")
    ncpu = min(os.cpu_count(), ncpu)

    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=size)

    if task == 'set':
        if mode == 'spacy':
            import spacy
            print('Loading spacy models...')
            nlp = [spacy.load('en_core_web_sm') for _ in range(ncpu)] # perhaps not the best way
        else:
            nlp = []

        with mp.Pool(processes=ncpu, initializer=init_pool, initargs=(nlp, ncpu, nsamples, mode)) as pool:
            for collection in collections:
                collsize = collection.count_documents({})
                batch_update = []
                batch_size = 10000

                print(f'Starting pool working on {collection.database.name}.{collection.name}...')
                for res in tqdm(pool.imap(
                    worker, ((t['_id_numeric'], t['content'], mode) for t in collection.find({}, projection={"_id_numeric": 1, "content": 1})), chunksize=100), 
                    total=collsize
                    ):
                    batch_update.append(pymongo.UpdateOne({"_id_numeric": res[0]}, {"$set": {"numeric_columns": res[1]}}))
                    if len(batch_update) == batch_size:
                        collection.bulk_write(batch_update, ordered=False)
                        batch_update = []
                
                if len(batch_update) > 0:
                    collection.bulk_write(batch_update, ordered=False)
                print(f'{collection.database.name}.{collection.name} updated.')
    else:
        for collection in collections:
            print(f'Start unsetting field "numeric_columns" from {collection.database.name}.{collection.name}...')
            collection.update_many({}, {"$unset": {"numeric_columns": 1}})
            print(f'{collection.database.name}.{collection.name} updated.')
            
        