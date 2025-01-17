import os

from dltf.testers.josie.josie import JOSIEGS
from dltf.utils.datalake import MongoDBDataLakeHandler
from dltf.utils.loghandler import logging_setup

# Set up the DataLake handler
datalake_name               = 'demo'
datalake_location           = 'mongodb'
datasets                    = ['sloth.demo']
dlh                         = MongoDBDataLakeHandler(datalake_location, datalake_name, datasets)

data_path                   = f'{os.path.dirname(__file__)}/data'
tmp_path                    = f'{os.path.dirname(__file__)}/tmp'

# create data folder if it doesn't exist
if not os.path.exists(data_path):
    os.mkdir(data_path)

# create tmp folder if it doesn't exist
if not os.path.exists(tmp_path):
    os.mkdir(tmp_path)

# JOSIE (global search tool) parameters
mode                        = 'bag'
blacklist                   = set()
string_translators          = ['whitespace', 'lowercase']
string_patterns             = []
force_sampling_cost         = True # force JOSIE to do cost sampling before querying
token_table_on_memory       = False # build the token table used by JOSIE directly on disk
dbstatfile                  = f'{data_path}/db-stat.csv'
tokens_bidict_file          = f'{data_path}/tokens-bidict.pickle'

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
    "spark.master"                  : "local[8]",
    "spark.executor.memory"         : "100g",
    "spark.driver.memory"           : "20g",
    "spark.local.dir"               : tmp_path,
    "spark.driver.maxResultSize"    : "12g",
    "spark.jars.packages"           : "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0",
    "spark.driver.extraClassPath"   : f"{os.environ['HOME']}/.ivy2/jars/org.postgresql_postgresql-42.7.3.jar"
}

logging_setup()

# Instatiate JOSIE
josie = JOSIEGS(
    mode=mode,
    blacklist=blacklist,
    datalake_handler=dlh,
    string_translators=string_translators,
    string_patterns=string_patterns,
    dbstatfile=dbstatfile,
    tokens_bidict_file=tokens_bidict_file,
    josie_db_connection_info=db_config,
    spark_config=spark_config
)

# Create the index for JOSIE
josie.data_preparation()
dlh.close()

