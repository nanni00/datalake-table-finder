import argparse
import pymongo



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', 
                        required=True, choices=['set', 'unset'], 
                        help='if "set", add a new field "_id_numeric" to each document, i.e.  \
                            a numeric identifier to each document (in addition to the string _id already present). \
                            If "unset", delete the field "_id_numeric".')
    parser.add_argument('--small', required=False, action='store_true',
                        help='works on small collection versions (only for testing)')

    args = parser.parse_args()
    task = args.task
    small = args.small

    mongoclient = pymongo.MongoClient()
    if not small:
        turlcoll = mongoclient.optitab.turl_training_set
        snapcoll = mongoclient.sloth.latest_snapshot_tables
    else:
        turlcoll = mongoclient.optitab.turl_training_set_small
        snapcoll = mongoclient.sloth.latest_snapshot_tables_small

    if task == 'set':
        turlcoll_size = 570000 if not small else 10000
        snapshot_size = 2100000 if not small else 10000

        _id_numeric = 0
        for (collection, collsize, collname) in [(snapcoll, snapshot_size, 'sloth.latest_snapshot_tables'), 
                                                (turlcoll, turlcoll_size, 'optitab.turl_training_set')]:
            print(f"Starting creating \"_id_numeric\" field on collection {collname}...")
            for i, doc in enumerate(collection.find({}, projection={"_id": 1})):
                collection.update_one({"_id": doc["_id"]}, {"$set": {"_id_numeric": _id_numeric}})
                _id_numeric += 1
                if i % 1000 == 0:
                    print(round((i * 100) / collsize, 3), '%', end='\r')
            print(f'{collname} updated.')
            
    else:
        print('Start unsetting field "_id_numeric" from optitab.turl_training_set...')
        turlcoll.update_many({}, {"$unset": {"_id_numeric": 1}})
        print('optitab.turl_training_set updated.')
        
        print('Start unsetting field "_id_numeric" from sloth.latest_snapshot_tables...')
        snapcoll.update_many({}, {"$unset": {"_id_numeric": 1}})
        print('sloth.latest_snapshot_tables updated.')