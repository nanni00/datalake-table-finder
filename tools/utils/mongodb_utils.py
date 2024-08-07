import logging
import pymongo
import pymongo.collection


def get_mongodb_collections(dataset:str='wikipedia', size:str='standard') -> tuple[pymongo.MongoClient, list[pymongo.collection.Collection]]:
    mongoclient = pymongo.MongoClient(directConnection=True)
    collections = []
    if size not in ['small', 'standard']:
        logging.error('Unknown dataset size: ' + str(size))
        raise ValueError('Unknown dataset size: ' + str(size))

    if dataset == 'wikipedia':
        if size == 'small':
            collections.append(mongoclient.optitab.turl_training_set_small)
            collections.append(mongoclient.sloth.latest_snapshot_tables_small)
        else:
            collections.append(mongoclient.optitab.turl_training_set)
            collections.append(mongoclient.sloth.latest_snapshot_tables)
    elif dataset == 'gittables':
        if size == 'small':
            collections.append(mongoclient.sloth.gittables_small)
        else:
            collections.append(mongoclient.sloth.gittables)
    else:
        logging.error('Unknown dataset: ' + str(dataset))
        raise ValueError('Unknown dataset: ' + str(dataset))

    return mongoclient, collections



def get_one_document_from_mongodb_by_key(key, value, *collections:tuple[pymongo.collection.Collection]):
    for collection in collections:
        document = collection.find_one({key: value})
        if document:
            return document
