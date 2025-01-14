import os

from dltf.testers.josie.josie import JOSIETester
from dltf.utils.datalake import MongoDBDataLakeHandler
from dltf.utils.loghandler import logging_setup

# Set up the DataLake handler
datalake_name               = 'demo'
datalake_location           = 'mongodb'
datasets                    = ['sloth.latest_snapshot_tables']
dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

data_path                   = f'{os.path.curdir}/data'
query_file                  = f'{data_path}/query.json'
k                           = 20

# create data folder if it doesn't exist
if not os.path.exists(data_path):
    os.mkdir(data_path)

# JOSIE (global search tool) parameters
mode                        = 'bag'
blacklist                   = set()
token_translators           = ['whitespace', 'lowercase']
force_sampling_cost         = True # force JOSIE to do cost sampling before querying
token_table_on_memory       = False # build the token table used by JOSIE directly on disk
results_directory           = f'{data_path}'
dbstatfile                  = f'{data_path}/josie-db-stat.csv'
tokens_bidict_file          = f'{data_path}/josie-tokens-bidict.pickle'
results_file                = f'{data_path}/josie-results-raw.csv'

# connection info for the JOSIE inverted index
db_config = {
    'drivername': 'postgresql',
    'database'  : 'DEMODB',
    'port'      :  5442,
    'host'      : 'localhost',
    'username'  : 'demo',
    'password'  : 'demo',
}

# spark configuration used during index creation
spark_config = {
    "spark.app.name"                : "JOSIE Data Preparation",
    "spark.master"                  : "local[64]",
    "spark.executor.memory"         : "100g",
    "spark.driver.memory"           : "20g",
    "spark.local.dir"               : f"{os.path.curdir}/tmp",
    "spark.driver.maxResultSize"    : "12g",
    "spark.jars.packages"           : "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
    "spark.driver.extraClassPath"   : f"{os.environ['HOME']}/.ivy2/jars/org.postgresql_postgresql-42.7.3.jar"
}

logging_setup()

# Instatiate JOSIE
josie = JOSIETester(
    mode=mode,
    blacklist=blacklist,
    datalake_handler=dlh,
    token_translators=token_translators,
    dbstatfile=dbstatfile,
    tokens_bidict_file=tokens_bidict_file,
    josie_db_connection_info=db_config,
    spark_config=spark_config
)

# Create the index for JOSIE
josie.data_preparation()
dlh.close()

