import os
import spacy
import random
import pymongo
import argparse
import multiprocessing as mp

from tools.sloth.utils import parse_table


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
    # i.e. the majority of the cells has more numeric tags then other tags, 0 otherwise
    cells = []

    for sampled_cell_tags in sample_value_entry:
        num_numerical_tags = 0
        num_other_tags = 0

        for tag in sampled_cell_tags:
            if tag in NUMERICAL_NER_TAGS:
                num_numerical_tags += 1
            else:
                num_other_tags += 1
        cells.append(1 if num_numerical_tags > num_other_tags else -1)
    
    return 1 if sum(cells) >= 0 else 0


def detect_table_numeric_columns(table: list[list], nlp: spacy.Language, nsamples: int) -> tuple[str, list[int]]:
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
                    for token in doc if token.ent_type_ != ''
                ]
                for doc in nlp.pipe(random.sample(column, min(nsamples, len(table))))
            ]
        )
        for column in parse_table(table, len(table[0]), 0)
    ]


def worker(t: tuple[str, list[list]]):
    # given a pair (_id, content) detect the numeric columns
    return (t[0], detect_table_numeric_columns(t[1], nlp[os.getpid() % ncpu], nsamples))


def init_pool(nlp_array, ncpu, nsamples):
    global nlp
    nlp = nlp_array
    ncpu = ncpu
    nsamples = nsamples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', 
                        required=True, choices=['set', 'unset'], 
                        help='if "set", add a new field "numeric_columns" to each document, i.e. table,  \
                            an array of lenght #columns where if the ith value of the array is 1 then the \
                            ith column of the table is numeric, 0 otherwise. \
                            If "unset", delete the field "numeric_columns".')
    parser.add_argument('-w', '--num-workers', 
                        type=int, required=False, default=min(os.cpu_count(), 64),
                        help='number of CPU(s) to use for processing, default is the minimum between computer CPUs and 64.')
    parser.add_argument('-s', '--sample-size', 
                        type=int, required=False, default=3,
                        help='defines how many cell(s) must be sampled from each column to detect wheter or not the column is numeric, default is 3.')
    parser.add_argument('--small', required=False, action='store_true',
                        help='works on small collection versions (only for testing)')

    args = parser.parse_args()
    mode = args.mode
    ncpu = args.num_workers
    nsamples = args.sample_size
    small = args.small

    mongoclient = pymongo.MongoClient()
    if not small:
        turlcoll = mongoclient.optitab.turl_training_set
        snapcoll = mongoclient.sloth.latest_snapshot_tables
    else:
        turlcoll = mongoclient.optitab.turl_training_set_small
        snapcoll = mongoclient.sloth.latest_snapshot_tables_small

    if mode == 'set':
        print('Loading spacy models...')
        nlp = [spacy.load('en_core_web_sm') for _ in range(ncpu)]

        turlcoll_size = 570000 if not small else 10000
        snapshot_size = 2100000 if not small else 10000

        with mp.Pool(processes=ncpu, initializer=init_pool, initargs=(nlp, ncpu, nsamples)) as pool:
            for (collection, collsize, collname) in [(snapcoll, snapshot_size, 'sloth.latest_snapshot_tables'), 
                                                     (turlcoll, turlcoll_size, 'optitab.turl_training_set')]:
                print(f'Starting pool working on {collname}...')
                for i, res in enumerate(pool.imap(worker, ((t['_id'], t['content']) for t in collection.find({}, projection={"_id": 1, "content": 1})), chunksize=100)):
                    if i % 1000 == 0:
                        print(round(100 * i / collsize, 3), '%', end='\r')
                    collection.update_one({"_id": res[0]}, {"$set": {"numeric_columns": res[1]}})
                print(f'{collname} updated.')
    else:
        print('Start unsetting field "numeric_columns" from optitab.turl_training_set...')
        turlcoll.update_many({}, {"$unset": {"numeric_columns": 1}})
        print('optitab.turl_training_set updated.')
        
        print('Start unsetting field "numeric_columns" from sloth.latest_snapshot_tables...')
        snapcoll.update_many({}, {"$unset": {"numeric_columns": 1}})
        print('sloth.latest_snapshot_tables updated.')
        