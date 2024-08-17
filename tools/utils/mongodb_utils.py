import logging
import sys
import pymongo
import pymongo.collection

from tools.utils.basicconfig import datasets, datasets_size


def get_mongodb_collections(dataset:str='wiki_snap_turl', size:str='standard') -> tuple[pymongo.MongoClient, list[pymongo.collection.Collection]]:
    if dataset not in datasets:
        logging.error('Unknown dataset: ' + str(dataset))
        raise ValueError('Unknown dataset: ' + str(dataset))
    if size not in datasets_size:
        logging.error('Unknown dataset size: ' + str(size))
        raise ValueError('Unknown dataset size: ' + str(size))

    mongoclient = pymongo.MongoClient(directConnection=True)
    collections = []

    if 'optitab' in mongoclient.list_database_names():
        match dataset:
            case 'wikitables':
                collections.append('mongoclient.optitab.turl_training_set')
                collections.append('mongoclient.sloth.latest_snapshot_tables')
            case 'gittables':
                collections.append('mongoclient.sloth.gittables')
    elif 'datasets' in mongoclient.list_database_names():
        match dataset:
            case 'wikitables':
                collections.append('mongoclient.datasets.wikitables')
            case 'gittables':
                collections.append('mongoclient.datasets.gittables')
    else:
        logging.error('Current MongoDB not configured')
        raise KeyError('Current MongoDB not configured')
    
    collections = [eval(c + '_small' if size == 'small' else c, {'mongoclient': mongoclient}) for c in collections]
    return mongoclient, collections



def get_one_document_from_mongodb_by_key(key, value, *collections:tuple[pymongo.collection.Collection]):
    for collection in collections:
        document = collection.find_one({key: value})
        if document:
            return document


def get_document_from_mongodb_by_numeric_id(id_numeric, *collections:tuple[pymongo.collection.Collection]):
    return get_one_document_from_mongodb_by_key('_id_numeric', id_numeric, *collections)
