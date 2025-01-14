import os
import pickle

from dltf.testers.josie.josie import JOSIETester

from dltf.utils.tables import table_to_tokens
from dltf.utils.datalake import MongoDBDataLakeHandler
from dltf.utils.misc import clean_string
from dltf.utils.loghandler import logging_setup, info



data_path                   = f'{os.path.dirname(__file__)}/data'
k                           = 50

# Set up the DataLake handler
datalake_name               = 'demo'
datalake_location           = 'mongodb'
datasets                    = ['sloth.latest_snapshot_tables']
dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

# create data folder if it doesn't exist
if not os.path.exists(data_path):
    os.mkdir(data_path)

# JOSIE (global search tool) parameters
mode                        = 'bag'
blacklist                   = set()
token_translators           = ['whitespace', 'lowercase']
force_sampling_cost         = False # force JOSIE to do cost sampling before querying
token_table_on_memory       = False # build the token table used by JOSIE directly on disk
results_directory           = f'{data_path}'
tokens_bidict_file          = f'{data_path}/josie-tokens-bidict.pickle'
results_file                = f'{data_path}/results/football.csv'

# connection info for the JOSIE inverted index
db_config = {
    'drivername': 'postgresql',
    'database'  : 'DEMODB',
    'port'      :  5442,
    'host'      : 'localhost',
    'username'  : 'demo',
    'password'  : 'demo',
}

# Logging utility
logging_setup()

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

# Define the query IDs
# Maybe these tables aren't indexed due to the num_header_rows != !
# since we consider only those tables with num_header_rows == 1...
query_ids = [
    38
]

# Get the query documents from MongoDB
# The documents have five fields: _id, _id_numeric, headers, content and valid_columns
query_docs = [dlh.get_table_by_numeric_id(qid) for qid in query_ids]

# Load the bidictionary between the JOSIE tokens IDs and the correspondent original string
info('Loading tokens bidictionary...')
with open(tokens_bidict_file, 'rb') as fr:
    tokens_bidict = pickle.load(fr)

# For each document, extract a set of tokens from its content
query_sets = [
    [
        doc['_id_numeric'],
        table_to_tokens(
            table=doc['content'], 
            valid_columns=doc['valid_columns'], 
            mode=mode, 
            blacklist=blacklist,
            string_translators=token_translators)
        ]
    for doc in query_docs
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

# for q, qs in query_sets.items():
#     print(q, len(qs))

# Search results for the query sets
josie.query(
    results_file=results_file, 
    k=k, 
    queries=query_sets, 
    force_sampling_cost=force_sampling_cost, 
    results_directory=results_directory, 
    token_table_on_memory=token_table_on_memory,
    verbose=True
)

