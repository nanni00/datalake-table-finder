import pickle
from abc import ABC, abstractmethod

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
    def config(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def clone(self):
        pass


class DataLakeHandlerFactory:
    def create_handler(datalake_location:str, *args) -> DataLakeHandler:
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
                return self.doc_to_table(document)
            
    def get_table_by_numeric_id(self, numeric_id, return_original_doc=False):
        for collection in self._collections:
            if (document := collection.find_one({'_id_numeric': numeric_id})) != None:
                return document if return_original_doc else self.doc_to_table(document)

    def count_tables(self):
        return sum(c.count_documents({}, hint='_id_') for c in self._collections)
    
    def scan_tables(self, _from = 0, _to = -1):
        _to = _to if _to != -1 else self.count_tables()
        query = {'$and': [
                    {'_id_numeric': { '$gte': _from}},
                    {'_id_numeric': {'$lte': _to}}]
                }
        
        for collection in self._collections:
            for doc in collection.find(query):
                yield self.doc_to_table(doc)
    
    def doc_to_table(self, doc):
        headers = doc['headers'] if 'headers' in doc else doc['content'][:doc['num_header_rows']]
        content = doc['content'] if 'num_header_rows' not in doc else doc['content'][doc['num_header_rows']:]

        return {
            '_id_numeric': doc['_id_numeric'], 
            'content': content,
            'valid_columns': doc['valid_columns'], 
            'headers': headers
        }
    
    def config(self):
        return self.datalake_location, self.datalake_name, self.dataset_names

    def clone(self):
        return DataLakeHandlerFactory.create_handler(self.datalake_location, self.datalake_name, self.dataset_names)

    def close(self):
        return self._mongoclient.close()


class LocalFileDataLakeHandler(DataLakeHandler):
    def __init__(self, datalake_location, datalake_name, *args):
        self.datalake_location = datalake_location
        self.datalake_name = datalake_name
        self.mapping_id_path = f"{self.datalake_location}/mapping_id.pickle"
        self.valid_columns_path = f"{self.datalake_location}/valid_columns.pickle"
        with open(self.mapping_id_path, 'rb') as fr:
            self.mapping_id = pickle.load(fr)
        with open(self.valid_columns_path, 'rb') as fr:
            self.valid_columns = pickle.load(fr)

    def get_table_by_id(self, _id):
        raise NotImplementedError()

    def get_table_by_numeric_id(self, _id_numeric):
        content = pl.read_csv(f'{self.datalake_location}/tables/{self.mapping_id[_id_numeric]}.csv', has_header=False, infer_schema_length=0, encoding='latin1').rows()
        valid_columns = self.valid_columns[_id_numeric]
        headers = content[0]
        return {'_id_numeric': _id_numeric, 'content': content, 'headers': headers, 'valid_columns': valid_columns}

    def count_tables(self):
        return len(self.mapping_id)
        
    def scan_tables(self, _from = 0, _to = -1):
        for _id_numeric in range(_from, _to+1):
            yield self.get_table_by_numeric_id(_id_numeric)

    def config(self):
        return self.datalake_location, self.datalake_name

    def close(self):
        pass

    def clone(self):
        return DataLakeHandlerFactory.create_handler(self.datalake_location, self.datalake_name)



