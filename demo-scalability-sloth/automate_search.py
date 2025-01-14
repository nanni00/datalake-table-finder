import os
import re
import pickle

import pymongo
import polars as pl

from dltf.utils import tables
from dltf.utils.misc import clean_string, largest_overlap_sloth
from dltf.testers.josie.josie import JOSIETester
from dltf.utils.datalake import MongoDBDataLakeHandler
from dltf.utils.loghandler import logging_setup, info


def prepare_query(qdoc, mode='bag', blacklist=set()):    
    global tokens_bidict

    # Extract a bag of tokens from the document's content
    query_sets = [
        [
            doc['_id_numeric'],
            tables.table_to_tokens(
                table=doc['content'], 
                valid_columns=doc['valid_columns'], 
                mode=mode, 
                blacklist=blacklist,
                string_translators=token_translators
            )
        ]
        for doc in [qdoc]
    ]
    
    # Map each token in the sets with its correspondent token ID for JOSIE
    query_sets = [
        [
            query_id, 
            sorted([
                tokens_bidict.inverse[clean_string(token, 'lowercase', 'whitespace')]
                for token in query_set 
                if clean_string(token, 'lowercase', 'whitespace') in tokens_bidict.inverse
            ])
        ]
        for query_id, query_set in query_sets
    ]

    # Transform the list <ID, tokens[]> into a dictionary <ID: tokens[]>
    query_sets = dict(query_sets)
    return query_sets


def get_result_ids(s):
    return list(map(int, re.findall(r'\d+', s)[::2]))

def get_result_overlaps(s):
    return list(map(int, re.findall(r'\d+', s)[1::2]))


# Connect to MongoDB
mongoclient = pymongo.MongoClient()
collection = mongoclient.sloth.latest_snapshot_tables

# Query and general parameters
data_path                   = f'{os.path.dirname(__file__)}/data'
k                           = 5
blacklist                   = set()
regex_replace_pattern       = re.compile('')

# Set up the DataLake handler
datalake_name               = 'demo'
datalake_location           = 'mongodb'
datasets                    = ['sloth.latest_snapshot_tables']
dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

# JOSIE (global search tool) parameters
mode                        = 'bag'
blacklist                   = set()
token_translators           = ['whitespace', 'lowercase']
force_sampling_cost         = False # force JOSIE to do cost sampling before querying
token_table_on_memory       = False # build the token table used by JOSIE directly on disk
tokens_bidict_file          = f'{data_path}/josie-tokens-bidict.pickle'
results_file                = f'{data_path}/results/tmp.csv'
logfile                     = f'{data_path}/.log'

# set up the logging to trace results
logging_setup(logfile=logfile, on_stdout=True)

# connection info for the JOSIE inverted index
db_config = {
    'drivername': 'postgresql',
    'database'  : 'DEMODB',
    'port'      :  5442,
    'host'      : 'localhost',
    'username'  : 'demo',
    'password'  : 'demo',
}

# Instatiate JOSIE
josie = JOSIETester(
    mode=mode,
    blacklist=blacklist,
    datalake_handler=dlh,
    token_translators=token_translators,
    dbstatfile=None,
    tokens_bidict_file=tokens_bidict_file,
    josie_db_connection_info=db_config,
    spark_config=None
)

# Load the bidictionary between the JOSIE tokens IDs and the correspondent original string
info('Start loading bidict...')
with open(tokens_bidict_file, 'rb') as fr:
    tokens_bidict = pickle.load(fr)



# Define what we want to search and what not
search_tokens = {' city', 'cancer', 'patho', 'environ', 'space', 'population'}
filter_tokens = {'champ', 'disc', 'race', 'olymp', 'result', 'minist', 'member', 'list', 'york', 'kansas', 'toronto', 'junction', 'minnesota'}
start_from = 0


info(f' Start searching for tokens {search_tokens}, filtering tokens {filter_tokens} '.center(200, '#'))

for i, doc in enumerate(collection.find({})):
    if not tables.is_valid_table(doc['content'][doc['num_header_rows']:], doc['valid_columns']):
        continue

    if any(token in str(doc['context']).lower() for token in search_tokens) and not any(token in str(doc['context']).lower() for token in filter_tokens):
        josie.query(results_file, k, prepare_query(doc, mode, blacklist))
        josie_results = [[q, list(zip(get_result_ids(r), get_result_overlaps(r)))] for q, r in pl.read_csv(f'{results_file}.raw').select('query_id', 'results').rows()][0][1]
        if len(josie_results) == 0:
            continue
        
        sloth_results = [
            [rid, collection.find_one({'_id_numeric': rid})]
            for rid, _ in josie_results
        ]

        sloth_results = sorted([
                [
                    rid,
                    largest_overlap_sloth(doc['content'][doc['num_header_rows']:], rtable['content'][rtable['num_header_rows']:],
                                          doc['valid_columns'], rtable['valid_columns'],
                                          verbose=False, blacklist=set())[0]
                ]
                for rid, rtable in sloth_results
            ], key=lambda x: x[1], reverse=True
        )

        if any(jrid != srid and srid >= 0 for (jrid, _), (srid, _) in zip(josie_results, sloth_results)):        
            info(f'{doc["_id"]} - {doc["_id_numeric"]} - {doc["context"]}')

        