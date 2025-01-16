import os

import json
from random import sample
from itertools import chain

import polars as pl
import pymongo

josie_results_file = f'{os.path.dirname(__file__)}/data/results/results2demo.csv'
json_datalake_file = f'{os.path.dirname(__file__)}/data/demo_datalake_tables.json'

df = pl.read_csv(josie_results_file)

print('Getting tables IDs')

# Get the query IDs
query_ids = list(chain(*df.select('query_id').rows()))

# Load the result IDs
table_ids = list(chain(*map(eval, chain(*df.select('result_ids').rows()))))

# Add some random table
random_table_ids = sample(range(int(2e6)), 10000)

ids = set(query_ids).union(table_ids, random_table_ids)

print(f'Total tables: {len(ids)}')

# Connect to MongoDB
mongoclient = pymongo.MongoClient()
collection = mongoclient.sloth.latest_snapshot_tables

# Download each table from MongoDB
print('Downloading tables from MongoDB')
tables_json = []
for i in ids:
    tables_json.append(collection.find_one({'_id_numeric': i}))

# Save all the tables into a unique JSON file
print('Saving tables to JSON')
with open(json_datalake_file, 'w') as fr:
    json.dump(tables_json, fr)

print('Done')
