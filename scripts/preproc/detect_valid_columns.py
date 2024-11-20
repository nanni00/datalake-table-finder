import os
import multiprocessing as mp

import pymongo
from tqdm import tqdm

from thesistools.utils.misc import naive_detect_valid_columns
from thesistools.utils.datalake import DataLakeHandlerFactory



def worker(t: tuple[str, list[list]]):
    return (t[0], naive_detect_valid_columns(t[1]))
    

def mongodb_update_numeric_columns(task, num_cpu, *datalake_args):
    if os.cpu_count() < num_cpu:
        print(f"Using {os.cpu_count()} cores")
    num_cpu = min(os.cpu_count(), num_cpu)
    
    dlh = DataLakeHandlerFactory.create_handler(*datalake_args)
    collections = dlh._collections

    if task == 'set':
        with mp.Pool(processes=num_cpu) as pool:
            for collection in collections:
                collsize = collection.count_documents({})
                batch_update = []
                batch_size = 10000

                print(f'Starting pool working on {collection.database.name}.{collection.name}...')
                for res in tqdm(pool.imap(
                    worker, ((t['_id_numeric'], t['content']) for t in collection.find({}, projection={"_id_numeric": 1, "content": 1})), chunksize=100), 
                    total=collsize
                    ):
                    batch_update.append(pymongo.UpdateOne({"_id_numeric": res[0]}, {"$set": {"valid_columns": res[1]}}))
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
            

if __name__ == '__main__':
    mongodb_update_numeric_columns('set', 8, 'mongodb', 'wikitables', ['datasets.wikitables'])
