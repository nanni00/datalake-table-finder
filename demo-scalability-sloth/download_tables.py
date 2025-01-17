import os
import json
from random import sample
from itertools import chain

import pymongo
import polars as pl


josie_results_file  = f'{os.path.dirname(__file__)}/data/results/results2demo.csv'
json_datalake_file  = f'{os.path.dirname(__file__)}/data/demo_datalake_tables.json'
json_queries_file   = f'{os.path.dirname(__file__)}/data/demo_query_tables.json'
csv_queries         = f'{os.path.dirname(__file__)}/data/queries'

df = pl.read_csv(josie_results_file)

print('Getting tables IDs')

# Get the query IDs
query_ids = list(chain(*df.select('query_id').rows()))

# Load the result IDs
table_ids = list(chain(*map(eval, chain(*df.select('result_ids').rows()))))

# Add some random table
random_table_ids = sample(range(int(2e6)), int(1e4))

ids = set(table_ids).union(random_table_ids)

print(f'Total tables: {len(ids)}')

# Connect to MongoDB
mongoclient = pymongo.MongoClient()
collection = mongoclient.sloth.latest_snapshot_tables
all_wiki_collection = mongoclient.sloth.wikipedia_tables



# Download each table from MongoDB
print('Downloading query tables from MongoDB and saving them to CSV')
query_tables = []
for i in query_ids:
    table_obj = collection.find_one({'_id_numeric': i})
    query_tables.append(table_obj)
    table = table_obj['content'][table_obj['num_header_rows']:]
    pl.DataFrame(table, orient='row').write_csv(f'{csv_queries}/{i}.csv', include_header=False)

tables_json = []

# For each query table sample some of its older versions
print('Downloading multiple version tables from MongoDB')
for i in query_ids:
    for table_obj in all_wiki_collection.aggregate([{'$match': {'page': 214440}}, {'$sample': {'size': 10}}]):
        tables_json.append(table_obj)
        
print('Downloading random and result tables from MongoDB')
for i in ids:
    tables_json.append(collection.find_one({'_id_numeric': i}))

# Save all the tables into a unique JSON file
print('Saving tables to JSON')
with open(json_queries_file, 'w') as fr:
    json.dump(query_tables, fr)
    
with open(json_datalake_file, 'w') as fr:
    json.dump(tables_json, fr)

print('Done')
