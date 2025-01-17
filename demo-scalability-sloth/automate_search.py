import os
import re
import pickle

import pymongo
import polars as pl

from dltf.utils import tables
from dltf.utils.misc import clean_string, largest_overlap_sloth
from dltf.gsa.josie.josie import JOSIEGS
from dltf.utils.datalake import MongoDBDataLakeHandler
from dltf.utils.loghandler import error, logging_setup, info


def prepare_query(qdoc):    
    global tokens_bidict, string_translators, string_patterns, blacklist, mode

    # Extract a bag of tokens from the document's content
    query_sets = [
        [
            doc['_id_numeric'],
            tables.table_to_tokens(
                table=doc['content'], 
                valid_columns=doc['valid_columns'], 
                mode=mode, 
                blacklist=blacklist,
                string_translators=string_translators,
                string_patterns=string_patterns
            )
        ]
        for doc in [qdoc]
    ]
    
    # Map each token in the sets with its correspondent token ID for JOSIE
    query_sets = [
        [
            query_id, 
            sorted([
                tokens_bidict.inverse[clean_string(token, string_translators, string_patterns)]
                for token in query_set 
                if clean_string(token, string_translators, string_patterns) in tokens_bidict.inverse
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
blacklist                   = set(['–', '—', '-', '●', '&nbsp', '&nbsp;', '&nbsp; &nbsp;', 'yes', 'no', 'n/a', 'none', '{{y}}', '{{n}}', '{{yes}}', '{{no}}', '{{n/a}}'] + list(map(str, range(1000))))

string_translators          = ['whitespace', 'lowercase']
string_patterns             = []

# Set up the DataLake handler
datalake_name               = 'demo'
datalake_location           = 'mongodb'
datasets                    = ['sloth.latest_snapshot_tables']
dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

# JOSIE (global search tool) parameters
mode                        = 'bag'
force_sampling_cost         = False # force JOSIE to do cost sampling before querying
token_table_on_memory       = False # build the token table used by JOSIE directly on disk
tokens_bidict_file          = f'{data_path}/josie-tokens-bidict.pickle'
results_file                = f'{data_path}/tmp/results.csv'

# SLOTH parameters
min_w                       = 3
min_h                       = 10

# Log CSV file, related to parameters
logfile                     = f'{data_path}/k{k}-w{min_w}-h{min_h}.csv'


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
josie = JOSIEGS(
    mode=mode,
    blacklist=blacklist,
    datalake_handler=dlh,
    string_translators=string_translators,
    string_patterns=string_patterns,
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
search_tokens = set(['cit']) # set(['city', 'state', 'country', 'env', 'path', 'cance', 'health', 'space', 'air', 'gdp', 'rich', 'poor', 'clima', 'weath'])
'List of cities in the Democratic Republic of the Congo', 'Towns and cities'

filter_tokens = {'election', 'building', 'cup', 'fifa', 'team', 'finals', 'club', 'achievements', 'tour', 'open', 
                 'career', 'disc', 'dates',
                 'f. c.', 'open', 'winner', 'champ', 'disc', 'olymp', 
                 'result', 'minist', 'member',
                 'york', 'kansas', 'toronto', 'junction', 'minnesota', 'season', 'f.', 'c.', 
                 'player', 'performance', 'united state', 'visa', 'city'
                 }
start_from = 0


info(f' Start search '.center(100, '#'))
info(f' {min_w=}, {min_h=} '.center(100, '#'))
if search_tokens:
    info(f'{search_tokens=}')

for i, qdoc in enumerate(collection.find({})):
    if i < start_from:
        continue

    if search_tokens and not any(token in str(qdoc['context']).lower() for token in search_tokens):
        continue

    if filter_tokens and any(token in str(qdoc['context']).lower() for token in filter_tokens):
        continue

    if not tables.is_valid_table(qdoc['content'][qdoc['num_header_rows']:], qdoc['valid_columns']):
        continue

    try:
        josie.query(results_file, k, prepare_query(qdoc))
        josie_results = [[q, list(zip(get_result_ids(r), get_result_overlaps(r)))] for q, r in pl.read_csv(f'{results_file}.raw').select('query_id', 'results').rows() if r]
    except Exception as exc:
        error(exc) 
        continue

    if len(josie_results) == 0:
        continue

    josie_results = josie_results[0][1]
    if all(ov <= min_w * min_h for _, ov in josie_results):
        continue
    
    sloth_results = [
        [rid, collection.find_one({'_id_numeric': rid})]
        for rid, _ in josie_results
    ]

    sloth_results = sorted([
            [
                rid,
                largest_overlap_sloth(
                    r_tab=qdoc['content'][qdoc['num_header_rows']:], 
                    s_tab=rdoc['content'][rdoc['num_header_rows']:],
                    r_valid_cols=qdoc['valid_columns'], 
                    s_valid_cols=rdoc['valid_columns'],
                    blacklist=set(),
                    verbose=False,
                    min_w=min_w, 
                    min_h=min_h
                )[0]
            ]
            for rid, rdoc in sloth_results
        ], key=lambda x: x[1], reverse=True
    )

    # I want cases where at least N results actually have an overlap with the query table
    if sum(ov > 0 for _, ov in sloth_results) <= 1:
        continue

    if any(jrid != srid and srid >= 0 for (jrid, _), (srid, _) in zip(josie_results, sloth_results)):        
        info(f'{qdoc["_id"]};{qdoc["_id_numeric"]};{str(sloth_results)};{str(qdoc["context"]).replace(";", " ")}')

        