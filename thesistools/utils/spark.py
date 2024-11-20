import polars as pl

from thesistools.utils.datalake import DataLakeHandler

from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf, RDD


def get_spark_session(dlh:DataLakeHandler, **spark_config) -> tuple[SparkSession, RDD]:
    conf = SparkConf().setAll(list(spark_config.items()))
    sc = SparkContext.getOrCreate(conf=conf)
    spark = SparkSession(sc)

    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("WARN")

    match dlh.datalake_location:
        # in the end we must have a RDD with tuples (table_id, table_content, table_valid_columns)
        # with table_id as an integer,
        # table_content as a list of list (rows)
        # table_valid_columns as a list of 0/1 values (0 => column i-th is not numeric, 1 otherwise)
        case 'mongodb':
                init_rdd = spark.sparkContext.emptyRDD()
                for database, collection in [d.split('.') for d in dlh.dataset_names]:
                    init_rdd = init_rdd.union(
                        spark
                        .read
                        .format("mongodb")
                        .option ("uri", "mongodb://127.0.0.1:27017/")
                        .option("database", database)
                        .option("collection", collection)
                        .load()
                        .select('_id_numeric', 'content', 'valid_columns')
                        .rdd
                        .map(list)
                    )
        case _:
                # otherwise, the datalake is stored on disk as CSV files
                init_rdd = spark.sparkContext.parallelize([(id_num, id_name) for id_num, id_name in dlh.mapping_id.items()])

                init_rdd = (
                    init_rdd
                    .map(lambda tabid_tabf: (tabid_tabf[0], pl.read_csv(f'{dlh.datalake_location}/{tabid_tabf[1]}.csv', infer_schema_length=0, encoding='latin1').rows()))
                    .map(lambda tid_tab: (tid_tab[0], tid_tab[1], dlh.valid_columns[tid_tab[0]]))
                )

    return spark, init_rdd
