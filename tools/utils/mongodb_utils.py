import logging
import pymongo
import pymongo.collection

from tools.utils.basicconfig import datasets, datasets_size

def get_mongodb_collections(dataset:str='wikitables', size:str='standard') -> tuple[pymongo.MongoClient, list[pymongo.collection.Collection]]:
    mongoclient = pymongo.MongoClient(directConnection=True)
    collections = []
    if size not in datasets_size:
        logging.error('Unknown dataset size: ' + str(size))
        raise ValueError('Unknown dataset size: ' + str(size))

    if dataset == 'wikitables':
        if size == 'small':
            # collections.append(mongoclient.dataset.turl_training_set_small)
            collections.append(mongoclient.datasets.wikitables)
        else:
            # collections.append(mongoclient.dataset.turl_training_set)
            collections.append(mongoclient.datasets.wikitables)
    elif dataset == 'gittables':
        if size == 'small':
            collections.append(mongoclient.datasets.gittables_small)
        else:
            collections.append(mongoclient.datasets.gittables)
    else:
        logging.error('Unknown dataset: ' + str(dataset))
        raise ValueError('Unknown dataset: ' + str(dataset))

    return mongoclient, collections



def get_one_document_from_mongodb_by_key(key, value, *collections:tuple[pymongo.collection.Collection]):
    for collection in collections:
        document = collection.find_one({key: value})
        if document:
            return document


def get_document_from_mongodb_by_numeric_id(id_numeric, *collections:tuple[pymongo.collection.Collection]):
    return get_one_document_from_mongodb_by_key('_id_numeric', id_numeric, *collections)
