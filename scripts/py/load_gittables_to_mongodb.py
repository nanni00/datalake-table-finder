"""
Loads all the GitTables stored as CSVs into the MongoDB collection sloth.gittables
Assumes that the collection and the index on the _id_numeric field already exist
"""

import os
import pymongo
import pandas as pd
from tqdm import tqdm

from tools.utils.mongodb_utils import get_mongodb_collections


gittables_csv_folder = '/data3/zecca/projects/sloth/armadillo/gittables/dataset/tables_csv'

mongoclient, collections = get_mongodb_collections(dataset='gittables', size='standard')
gittables_coll = collections[0]

batch_size = 1000
batch_tables = []

for table_id in tqdm(os.listdir(gittables_csv_folder)):
    if not gittables_coll.find_one({'_id': table_id}):
        table_df = pd.read_csv(gittables_csv_folder + '/' + table_id, lineterminator="\n")
        table_obj = dict()
        table_obj["_id"] = table_id
        table_obj["content"] = table_df.values.tolist()
        table_obj["headers"] = list(table_df.columns)
        table_obj["num_header_rows"] = 0
        table_obj["num_columns"] = len(table_obj["content"][0])
        table_obj['num_rows'] = len(table_obj["content"])
        batch_tables.append(pymongo.InsertOne(table_obj))

        if len(batch_tables) == batch_size:
            try:
                nwriteop = gittables_coll.bulk_write(batch_tables, ordered=False)
            except OverflowError:
                # there are integer stored with a number of bytes that MongoDB doesn't support.
                # in case of error it seems that even with the "ordered=False" option
                # all the remaining tables in the bulk write aren't written to the database,
                # so scan the batch and try to insert each single table one by one
                for table in batch_tables:
                    try: gittables_coll.insert_one(table)
                    except: continue
            finally:
                batch_tables = []

if batch_tables:
    try:
        gittables_coll.bulk_write(batch_tables, ordered=False)
    except:
        for table in batch_tables:
            try: gittables_coll.insert_one(table)
            except: continue

mongoclient.close()
