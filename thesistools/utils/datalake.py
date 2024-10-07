import pickle

import polars as pl
import pymongo
import pymongo.collection

from thesistools.utils.basicconfig import MONGODB_DATALAKES, DATALAKE_SIZES


def get_mongodb_collections(dataset:str, size:str='standard') -> tuple[pymongo.MongoClient, list[pymongo.collection.Collection]]:
    assert dataset in MONGODB_DATALAKES
    assert size in DATALAKE_SIZES
    
    mongoclient = pymongo.MongoClient(directConnection=True)
    collections = []

    if 'optitab' in mongoclient.list_database_names():
        match dataset:
            case 'wikiturlsnap':
                collections.append('mongoclient.optitab.turl_training_set')
                collections.append('mongoclient.sloth.latest_snapshot_tables')
            case 'gittables':
                collections.append('mongoclient.sloth.gittables')
            case _:
                raise ValueError(f'Unknown dataset: {dataset}')
    elif 'datasets' in mongoclient.list_database_names():
        match dataset:
            case 'wikitables':
                collections.append('mongoclient.datasets.wikitables')
            case 'gittables':
                collections.append('mongoclient.datasets.gittables')
            case _:
                raise ValueError(f'Unknown dataset: {dataset}')
    else:
        raise ValueError('Current MongoDB not configured')
    
    collections = [eval(c + '_small' if size == 'small' else c, {'mongoclient': mongoclient}) for c in collections]
    return mongoclient, collections



def get_document_from_mongodb_by_numeric_id(id_numeric, *collections:pymongo.collection.Collection):
    for collection in collections:
        if (document := collection.find_one({'_id_numeric': id_numeric})) != None:
            return document


def get_one_document_from_mongodb_by_key(*args):
    raise DeprecationWarning()



def format_wikitables_header(header:list):
    txt_headers = [[h['text'] for h in header_row] for header_row in header]
    return [' '.join([h[i] for h in txt_headers]) for i in range(len(txt_headers[0]))]



class SimpleDataLakeHelper:
    """
    A simple class that helps to manage different data lake sources.
    Since the original datasets GitTables and WikiTurlSnap are stored in MongoDB
    and the SANTOS Large data lake as a CSVs folder, this structure avoids boiler plates code (sperem) """
    def __init__(self, datalake_location:str, *args):
        self.datalake_location = datalake_location
        self.datalake_name = None
        self.size = 'standard'
        self.mapping_id_path = None
        self.numeric_columns_path = None
        self.mapping_id = None
        self.numeric_columns = None
        
        match self.datalake_location:
            case 'mongodb':
                self.datalake_name = args[0]
                self.size = args[1] if len(args) > 1 else 'standard'
                self._mongoclient, self._collections = get_mongodb_collections(self.datalake_name, self.size)
            case _:
                self.mapping_id_path = args[2]
                self.numeric_columns_path = args[3]
                mapping_id_path, numeric_columns_path = args[2:]
                with open(mapping_id_path, 'rb') as fr:
                    self.mapping_id = pickle.load(fr)
                with open(numeric_columns_path, 'rb') as fr:
                    self.numeric_columns = pickle.load(fr)
                
    def get_table_by_numeric_id(self, numeric_id):
        """Return a dictionary with fields:
            - _id_numeric
            - content
            - numeric columns
            - headers """
        match self.datalake_location:
            case 'mongodb':
                if doc := get_document_from_mongodb_by_numeric_id(numeric_id, *self._collections):
                    content = doc['content']
                    numeric_columns = doc['numeric_columns']
                    headers = doc['headers'] if 'headers' in doc else None
                    if self.datalake_name == 'wikitables':
                        headers = format_wikitables_header(headers)  # on MongoDB WikiTables these headers aren't formatted as the WikiTurlSnap ones
                else:
                    return None
            case _:
                try:
                    # print('Ci prova...')
                    content = pl.read_csv(f'{self.datalake_location}/{self.mapping_id[numeric_id]}.csv', has_header=False, infer_schema_length=0, encoding='latin1').rows()
                    # print('...e ci riesce!')
                    numeric_columns = self.numeric_columns[numeric_id]
                    headers = content[0]
                except KeyError:
                    print('Nada')
                    return None
        try:
            return {'_id_numeric': numeric_id, 'content': content, 'numeric_columns': numeric_columns, 'headers': headers}
        except UnboundLocalError:
            print(numeric_id)
            raise UnboundLocalError()

    def get_number_of_tables(self):
        match self.datalake_location:
            case 'mongodb':
                return sum(c.count_documents({}, hint='_id_') for c in self._collections)
            case _:
                return len(self.mapping_id)

    def scan_tables(self, ignore_firsts:int=None):
        match self.datalake_location:
            case 'mongodb':
                query = {} if not ignore_firsts else {'_id_numeric': {'$gt': ignore_firsts}}
                for collection in self._collections:
                    for doc in collection.find(query):
                        headers = doc['headers'] if 'headers' in doc else doc['content'][0] if len(doc['content']) > 0 else None
                        if self.datalake_name == 'wikitables':
                            headers = format_wikitables_header(headers)  # on MongoDB WikiTables these headers aren't formatted as the WikiTurlSnap ones
                        yield {'_id_numeric': doc['_id_numeric'], 'content': doc['content'], 'numeric_columns': doc['numeric_columns'], 'headers': headers}
            case _:
                for _id_numeric in self.mapping_id.keys():
                    if ignore_firsts and _id_numeric < ignore_firsts:
                        continue
                    yield self.get_table_by_numeric_id(_id_numeric)

    def copy(self):
        return SimpleDataLakeHelper(self.datalake_location, self.datalake_name, self.size, self.mapping_id_path, self.numeric_columns_path)

    def close(self):
        match self.datalake_location:
            case 'mongodb':
                self._mongoclient.close()
            case _:
                pass



