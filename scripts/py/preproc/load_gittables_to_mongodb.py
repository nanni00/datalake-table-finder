"""
Loads all the GitTables stored as CSVs into MongoDB
"""

import os

import pymongo
import polars as pl
from tqdm import tqdm

from tools.utils.datalake import get_mongodb_collections
from tools.utils.settings import DefaultPath as dp


gittables_csv_folder = f'{dp.data_path.base}/raw_dataset/gittables_parquet'

mongoclient, collections = get_mongodb_collections(dataset='gittables', size='standard')
gittables_coll = collections[0]

batch_size = 10000
batch_tables = []
counter = 0
errors = 0


for subdir in os.listdir(gittables_csv_folder):
    print(f'Working on {subdir}...')
    for table_id in tqdm(os.listdir(os.path.join(gittables_csv_folder, subdir)), leave=False):
        try:
            # table_df = pd.read_csv(os.path.join(gittables_csv_folder, subdir, table_id), sep=None, engine='python')
            table_df = pl.read_parquet(os.path.join(gittables_csv_folder, subdir, table_id))
        except Exception:
            print(table_id)
            errors += 1
            # raise Exception()
            continue
        
        table_obj = dict()
        table_obj["_id"] = f"{subdir.replace('_csv', '').replace('_licensed', '')}.{table_id}"
        table_obj["_id_numeric"] = counter
        # table_obj["content"] = table_df.values.tolist()
        table_obj["content"] = table_df.rows()
        table_obj["headers"] = list(table_df.columns)
        table_obj["num_header_rows"] = 0
        table_obj["columns"] = len(table_obj["content"][0])
        table_obj['rows'] = len(table_obj["content"])

        counter += 1
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
                    except: 
                        errors += 1
                        continue
            finally:
                batch_tables = []

if batch_tables:
    try:
        gittables_coll.bulk_write(batch_tables, ordered=False)
    except:
        for table in batch_tables:
            try: gittables_coll.insert_one(table)
            except: 
                errors += 1
                continue

print(f"Total tables that have not been loaded due to errors: {errors}")
mongoclient.close()
