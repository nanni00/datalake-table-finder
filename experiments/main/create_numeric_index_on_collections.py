import argparse
import pymongo
from tqdm import tqdm

from tools.utils.utils import get_mongodb_collections



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
                        required=True, choices=['set', 'unset'], 
                        help='if "set", add a new field "_id_numeric" to each document, i.e.  \
                            a numeric identifier to each document (in addition to the string _id already present). \
                            If "unset", delete the field "_id_numeric".')
    parser.add_argument('--dataset', 
                        required=True, choices=['wikipedia', 'gittables'])
    parser.add_argument('--small', 
                        required=False, action='store_true',
                        help='works on small collection versions (only for testing)')

    args = parser.parse_args()
    task = args.task
    dataset = args.dataset
    small = args.small

    mongoclient, collections = get_mongodb_collections(dataset=dataset, size=small)
    batch_update = []
    batch_size = 1000

    if task == 'set':
        _id_numeric = 0
        for collection in collections:
            print(f'Scanning documents from {collection.database.name}.{collection.name}...')
            for doc in tqdm(collection.find({}, projection={"_id": 1}), total=collection.count_documents({})):
                batch_update.append(pymongo.UpdateOne({"_id": doc["_id"]}, {"$set": {"_id_numeric": _id_numeric}}))
                _id_numeric += 1
                if len(batch_update) == batch_size:
                    collection.bulk_write(batch_update, ordered=False)
                    batch_update = []
            collection.bulk_write(batch_update, ordered=False)
    else:
        for collection in collections:
            print(f'Start unsetting field "_id_numeric" from {collection.database.name}.{collection.name}...')
            collection.update_many({}, {"$unset": {"_id_numeric": 1}})
            print(f'{collection.database.name}.{collection.name} updated.')
        
    mongoclient.close()
