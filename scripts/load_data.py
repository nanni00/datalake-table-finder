import os
import json
import pickle
import multiprocessing as mp

import polars as pl
from tqdm import tqdm
from bidict import bidict
from pymongo import MongoClient, InsertOne, UpdateOne

from dltftools.utils.tables import naive_detect_valid_columns
from dltftools.utils.settings import DefaultPath as dp
from dltftools.utils.datalake import DataLakeHandlerFactory
    

def create_numeric_index_on_mongodb(*table_collections):
    """
    Create a numeric index on the MongoDB collections passed as input, in case the tables were already stored on it
    :param table_collections: The MongoDB collections that store the tables
    """
    batch_update = []
    batch_size = 1000
    _id_numeric = 0

    for collection in table_collections:
        print(f'Scanning documents from {collection.database.name}.{collection.name}...')
        for doc in tqdm(collection.find({}, projection={"_id": 1}), total=collection.count_documents({})):
            batch_update.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"_id_numeric": _id_numeric}}))
            _id_numeric += 1
            if len(batch_update) == batch_size:
                collection.bulk_write(batch_update, ordered=False)
                batch_update = []
        collection.bulk_write(batch_update, ordered=False)
    print('Completed.')



def insert_tables(batch_tables, table_collection):
    errors = 0
    try: 
        table_collection.bulk_write(batch_tables, ordered=False)
    except OverflowError:
        for table in batch_tables:
            try: table_collection.insert_one(table)
            except:
                errors += 1
    finally:
        batch_tables = []
    return errors




def _columns_detection_worker(t: tuple[str, list[list]]):
    return (t[0], naive_detect_valid_columns(t[1]))
    

def mongodb_detect_valid_columns(task, num_cpu, *datalake_args):
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
                    _columns_detection_worker, 
                    ((t['_id_numeric'], t['content']) for t in collection.find({}, projection={"_id_numeric": 1, "content": 1})), chunksize=100), 
                    total=collsize):
                    batch_update.append(UpdateOne({"_id_numeric": res[0]}, {"$set": {"valid_columns": res[1]}}))
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
            

def gittables2mongo(path_tables, table_collection, milestone, n):
    """
    :param path_tables: path of the directory containing the PARQUET version of the tables
    :param table_collection: MongoDB collection to store the tables
    :param milestone: table interval to use for tracking the progress
    :param n: maximum number of tables to load into the MongoDB collection
    """
    milestone = 10000
    batch_tables = []
    counter = 0
    errors = 0

    for subdir in os.listdir(path_tables):
        print(f'Working on {subdir}...')
        for table_id in tqdm(os.listdir(os.path.join(path_tables, subdir)), leave=False):
            try:
                table_df = pl.read_parquet(os.path.join(path_tables, subdir, table_id))
            except Exception:
                errors += 1
                continue
            
            table_obj = dict()
            table_obj["_id"] = f"{subdir}.{table_id}".replace('_csv', '').replace('_licensed', '').replace('.parquet', '')
            table_obj["_id_numeric"] = counter
            table_obj["content"] = table_df.rows()
            table_obj["headers"] = list(table_df.columns)
            table_obj["num_header_rows"] = 0
            table_obj["columns"] = len(table_obj["content"][0])
            table_obj['rows'] = len(table_obj["content"])

            counter += 1
            batch_tables.append(InsertOne(table_obj))

            if len(batch_tables) == milestone:
                errors += insert_tables(batch_tables, table_collection)
                batch_tables = []
            if counter >= n:
                break
        if counter >= n:
            break

    if batch_tables:
        errors += insert_tables(batch_tables, table_collection)
    print(f"Total tables that have not been loaded due to errors: {errors}")


def wiki2mongo(path_tables, table_collection, milestone, n):
    """
    :param path_tables: path of the JSONL file containing the tables
    :param table_collection: MongoDB collection to store the tables
    :param milestone: table interval to use for tracking the progress
    :param n: maximum number of tables to load into the MongoDB collection
    """
    tables = list()  # empty list to store the parsed tables
    counter = 0  # counter of the parsed tables
    with open(path_tables, "r") as input_file:  # open the JSONL file containing the tables
        for i, line in enumerate(input_file):  # for every table contained in the JSONL file
            if counter % milestone == 0 and len(tables) > 0:  # track the progress
                print(counter, end='\r')
                table_collection.insert_many(tables)
                tables = []
            raw_table = json.loads(line)  # load the table in its raw form

            if i >= n:
                break
            
            raw_table_content = raw_table["tableData"]  # load the table content in its raw form
            table = dict()  # empty dictionary to store the parsed table

            table["_id"] = raw_table["_id"]  # unique identifier of the table
            table["_id_numeric"] = counter
            
            table["content"] = [[cell["text"] for cell in row] for row in raw_table_content]  # table content (only cell text)
            
            caption = raw_table["tableCaption"] if "tableCaption" in raw_table else ""
            table["context"] = raw_table["pgTitle"] + " | " + raw_table["sectionTitle"] + " | " +  caption  # table context
            
            # basic format
            # table["headers"] = raw_table['tableHeaders']

            # for WikiTables format, which has a different "tableHeaders" format
            table["headers"] = [o['text'] for o in raw_table["tableHeaders"][0]]
            table['rows'] = raw_table['numDataRows']
            table['columns'] = raw_table['numCols']
            tables.append(table)  # append the parsed table to the list
            counter += 1  # increment the counter
    table_collection.insert_many(tables)
    print("Table parsing completed: " + str(counter) + " tables read and stored in the database.")


def santos2local(path_tables, mapping_id_path, valid_columns_path, n):
    """
    :param path_tables: path of the directory containing the CSV tables
    :param mapping_id_path: path of the PICKLE file that will store 
                            the mapping between the table original and integer ID
    :param valid_columns_path: path of the PICKLE file that will store 
                               the mapping between the table integer ID and its valid columns
                               as a list of integers, where 1 means valid and 0 not valid
    :param milestone: table interval to use for tracking the progress
    :param n: maximum number of tables to load into the MongoDB collection
    """
    ntables = len(os.listdir(path_tables))
    ignored_tables = 0
    nrows = 0
    ncols = 0
    counter = 0

    mapping_id = bidict()
    numeric_columns = dict()
    
    print(f'Scanning tables from {path_tables} and creating IDs mapping and checking for valid columns...')
    for table_file in tqdm(os.listdir(path_tables), total=ntables):
        try:
            mapping_id[counter] = table_file.removesuffix('.csv')
            table = pl.read_csv(f'{path_tables}/{table_file}', infer_schema_length=0, encoding='latin8', has_header=False).rows()
            nrows += len(table)
            ncols += len(table[0])
            numeric_columns[counter] = naive_detect_valid_columns(table)
            counter += 1
            if counter >= n:
                break
        except:
            ignored_tables += 1

    print(f'#tables={ntables}, #effectives={counter}, #ignored={ignored_tables}, #rows={nrows}, #cols={ncols}')

    print('Saving mapping...')
    with open(mapping_id_path, 'wb') as fw:
        pickle.dump(mapping_id, fw)

    print('Saving numeric columns...')
    with open(valid_columns_path, 'wb') as fw:
        pickle.dump(numeric_columns, fw)

    print('Completed.')


def main_wiki():
    # CHECK THE HEADER FORMAT!!!!!
    path_tables = f'{os.environ["HOME"]}/datasets_datalakes/WikiTables/tables.json'
    
    client = MongoClient()  # connect to MongoDB
    db = client.datasets  # define the database to use
    table_collection = db.wikitables  # define the collection in the database to store the tables
    milestone = 10000  # table interval to use for tracking the progress
    n = 100
    wiki2mongo(path_tables, table_collection, milestone, n)
    mongodb_detect_valid_columns('set', 8, 'mongodb', 'wikitables', [table_collection.full_name])
    

def main_gittables():
    # path_tables = f'{os.environ["HOME"]}/datasets_datalakes/GitTables'
    
    client = MongoClient()  # connect to MongoDB
    db = client.datasets  # define the database to use
    table_collection = db.gittables  # define the collection in the database to store the tables
    milestone = 10000  # table interval to use for tracking the progress
    n = 100
    # gittables2mongo(path_tables, table_collection, milestone, n)
    print("Detecting valid columns of the loaded tables...")
    mongodb_detect_valid_columns('set', 1, 'mongodb', 'gittables', [table_collection.full_name])
    

def main_santos_large():
    path_tables =           f'{os.environ["HOME"]}/datasets_datalakes/SantosLarge/tables'
    mapping_id_path =       f'{os.environ["HOME"]}/datasets_datalakes/SantosLarge/mapping_id.pickle'
    valid_columns_path =    f'{os.environ["HOME"]}/datasets_datalakes/SantosLarge/valid_columns.pickle'
    n = 100
    santos2local(path_tables, mapping_id_path, valid_columns_path, n)


def main_santos_small():
    path_tables =           f'{os.environ["HOME"]}/datasets_datalakes/SantosSmall/datalake'
    mapping_id_path =       f'{os.environ["HOME"]}/datasets_datalakes/SantosSmall/mapping_id.pickle'
    valid_columns_path =    f'{os.environ["HOME"]}/datasets_datalakes/SantosSmall/valid_columns.pickle'
    n = 1000
    santos2local(path_tables, mapping_id_path, valid_columns_path, n)
    

if __name__ == "__main__":
    # main_wiki()
    # main_gittables()
    main_santos_small()
    # main_santos_large()
