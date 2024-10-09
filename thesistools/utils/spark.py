import pymongo
import polars as pl

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf, RDD


def get_spark_session(
                      datalake_location:str, datalake_name:str, datalake_size:str='standard', 
                      datalake_mapping_id:dict|None=None, datalake_numeric_columns:dict|None=None,
                      **spark_config) -> tuple[SparkSession, RDD]:
    
    conf = SparkConf().setAll(list(spark_config.items()))
    sc = SparkContext.getOrCreate(conf=conf)
    spark = SparkSession(sc)

    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("WARN")

    match datalake_location:
        # in the end we must have a RDD with tuples (table_id, table_content, table_numeric_columns)
        # with table_id as an integer,
        # table_content as a list of list (rows)
        # table_numeric_columns as a list of 0/1 values (0 => column i-th is not numeric, 1 otherwise)
        case 'mongodb':
                # if the datalake is stored on MongoDB, then through the connector we
                # can easily access the tables
                mongoclient = pymongo.MongoClient(directConnection=True)
                
                match datalake_name:
                    case 'wikiturlsnap':
                        databases, collections = ['optitab', 'sloth'], ['turl_training_set', 'latest_snapshot_tables']
                    case 'gittables':
                        if 'sloth' in mongoclient.list_database_names():
                            databases = ['sloth']
                        elif 'dataset' in mongoclient.list_database_names():
                            databases = ['datasets']
                        collections = ['gittables']
                    case 'wikitables':
                        databases, collections = ['datasets'], ['wikitables']
                        
                collections = [c + '_small' if datalake_size == 'small' else c for c in collections]
                db_collections = zip(databases, collections)

                init_rdd = spark.sparkContext.emptyRDD()

                for database, collection_name in db_collections:
                    init_rdd = init_rdd.union(
                        spark 
                        .read
                        .format("mongodb")
                        .option ("uri", "mongodb://127.0.0.1:27017/") 
                        .option("database", database) 
                        .option("collection", collection_name)                         
                        .load() 
                        .select('_id_numeric', 'content', 'numeric_columns') 
                        .rdd
                        .map(list)
                    )
        case _:
                # otherwise, the datalake is stored on disk as CSV files
                init_rdd = spark.sparkContext.parallelize([(id_num, id_name) for id_num, id_name in datalake_mapping_id.items()])

                init_rdd = (
                    init_rdd
                    .map(lambda tabid_tabf: (tabid_tabf[0], pl.read_csv(f'{datalake_location}/{tabid_tabf[1]}.csv', infer_schema_length=0, encoding='latin1').rows()))
                    .map(lambda tid_tab: (tid_tab[0], tid_tab[1], datalake_numeric_columns[tid_tab[0]]))
                )

    return spark, init_rdd
