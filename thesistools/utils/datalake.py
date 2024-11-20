from abc import ABC, abstractmethod
import pickle

import polars as pl
import pymongo
import pymongo.collection


class DataLakeHandler(ABC):
    @abstractmethod
    def get_table_by_id(self, _id:str):
        pass

    @abstractmethod
    def get_table_by_numeric_id(self, numeric_id:int):
        pass

    @abstractmethod
    def scan_tables(self, _from:int=0, _to:int=-1):
        pass

    @abstractmethod
    def count_tables(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def clone(self):
        pass


class DataLakeHandlerFactory:
    def create_handler(datalake_location:str, *args):
        match datalake_location:
            case 'mongodb':
                return MongoDBDataLakeHandler(datalake_location, *args)
            case _:
                return LocalFileDataLakeHandler(datalake_location, *args)

class MongoDBDataLakeHandler(DataLakeHandler):
    def __init__(self, datalake_location, datalake_name, datasets:list[str]):
        self.datalake_location = datalake_location
        self.datalake_name = datalake_name
        self.dataset_names = datasets
        self._mongoclient = pymongo.MongoClient(directConnection=True)
        self._collections = [eval(f'mongoclient.{d}', {'mongoclient': self._mongoclient}) for d in self.dataset_names]

    def get_table_by_id(self, _id:str):
        for collection in self._collections:
            if (document := collection.find_one({'_id': _id})) != None:
                return document
            
    def get_table_by_numeric_id(self, numeric_id):
        for collection in self._collections:
            if (document := collection.find_one({'_id_numeric': numeric_id})) != None:
                return document

    def count_tables(self):
        return sum(c.count_documents({}, hint='_id_') for c in self._collections)
    
    def scan_tables(self, _from = 0, _to = -1):
        _to = -1 if _to == -1 else self.count_tables()
        query = {'$and': [
                    {'_id_numeric': { '$gt': _from}},
                    {'_id_numeric': {'$lt': _to}}]
                }
        
        for collection in self._collections:
            for doc in collection.find(query):
                yield self.doc_to_table(doc)
    
    def doc_to_table(self, doc):
        headers = doc['headers'] if 'headers' in doc else doc['content'][0] if len(doc['content']) > 0 else None
        return {
            '_id_numeric': doc['_id_numeric'], 
            'content': doc['content'], 
            'numeric_columns': doc['numeric_columns'], 
            'headers': headers
        }

    def clone(self):
        return DataLakeHandlerFactory.create_handler(self.datalake_location, self.datalake_name, self.dataset_names)

    def close(self):
        return self._mongoclient.close()


class LocalFileDataLakeHandler(DataLakeHandler):
    def __init__(self, datalake_location, datalake_name, *args):
        self.datalake_location = datalake_location
        self.datalake_name = datalake_name
        self.mapping_id_path = args[0]
        self.numeric_columns_path = args[1]
        with open(self.mapping_id_path, 'rb') as fr:
            self.mapping_id = pickle.load(fr)
        with open(self.numeric_columns_path, 'rb') as fr:
            self.numeric_columns = pickle.load(fr)

    def get_table_by_id(self, _id):
        raise NotImplementedError()

    def get_table_by_numeric_id(self, _id_numeric):
        content = pl.read_csv(f'{self.datalake_location}/{self.mapping_id[_id_numeric]}.csv', has_header=False, infer_schema_length=0, encoding='latin1').rows()
        numeric_columns = self.numeric_columns[_id_numeric]
        headers = content[0]
        return {'_id_numeric': _id_numeric, 'content': content, 'headers': headers, 'numeric_columns': numeric_columns}

    def count_tables(self):
        return len(self.mapping_id)
        
    def scan_tables(self, _from = 0, _to = -1):
        for _id_numeric in range(_from, _to+1):
            yield self.get_table_by_numeric_id(_id_numeric)

    def close(self):
        pass

    def clone(self):
        return DataLakeHandlerFactory.create_handler(self.datalake_location, self.datalake_name, self.mapping_id_path, self.numeric_columns_path)



