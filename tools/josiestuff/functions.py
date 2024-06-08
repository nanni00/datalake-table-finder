import json
import os
import mmh3
import pymongo.collection

import pymongo
import binascii
import jsonlines
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

import pyspark
from pyspark.sql import SparkSession

from tools.utils.utils import _create_token_set, print_info
    

@print_info(msg_before="Extracting tables from JSONL file...")
def extract_tables_from_jsonl_to_mongodb(
        input_tables_jsonl_file,
        table_collection:pymongo.collection.Collection,
        milestone=1000,
        ntotal_tables=570171
        ):
    tables = list()  # empty list to store the parsed tables

    with jsonlines.open(input_tables_jsonl_file) as reader:
        for counter, raw_table in tqdm(enumerate(reader, start=1), total=ntotal_tables):
            if counter % milestone == 0:
                # print(counter, end='\r')
                pass
            raw_table_content = raw_table["tableData"]  # load the table content in its raw form
            table = dict()  # empty dictionary to store the parsed table
            table["_id"] = raw_table["_id"]  # unique identifier of the table
            table["headers"] = raw_table["tableHeaders"][0]
            table["context"] = raw_table["pgTitle"] + " | " + raw_table["sectionTitle"] + " | " + raw_table["tableCaption"]  # table context            
            table["content"] = [[cell["text"] for cell in row] for row in raw_table_content]  # table content (only cell text)                
            tables.append(table)  # append the parsed table to the list
        table_collection.insert_many(tables)



# filtering on the IDs from the SLOTH results file should be deprecated, since now we'll work
# on a silver standard per test
@print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.')
def create_index(
    mode:str,
    thresholds:dict[str:int],
    small,
    test_name):

    MIN_ROW =     thresholds['min_rows']
    MAX_ROW =     thresholds['max_rows']
    MIN_COLUMN =  thresholds['min_columns']
    MAX_COLUMN =  thresholds['max_columns']
    MIN_AREA =    thresholds['min_area']
    MAX_AREA =    thresholds['max_area']

    # PostgreSQL write parameters
    url = "jdbc:postgresql://localhost:5442/nanni"

    properties = {
        "user": "nanni",
        "password": "",
        "driver": "org.postgresql.Driver"
    }
    
    spark_jars_packages = [
        'org.postgresql:postgresql:42.7.3',
        'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0'
    ]

    # fine tune of executor/driver.memory?
    builder = SparkSession.Builder()
    spark = builder \
        .appName("Big Bang Testing with MongoDB") \
        .master("local[64]") \
        .config('spark.jars.packages', ','.join(spark_jars_packages)) \
        .config('spark.executor.memory', '100g') \
        .config('spark.driver.memory', '10g') \
        .config('spark.local.dir', '/data4/nanni/spark') \
        .getOrCreate()
    
    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("ERROR")


    optitab__turl_training_set_df = spark \
        .read \
        .format("mongodb") \
        .option ("uri", "mongodb://127.0.0.1:27017/") \
        .option("database", "optitab") \
        .option("collection", "turl_training_set" if not small else "turl_training_set_small") \
        .load() \
        .select('_id', '_id_numeric', 'content', 'numeric_columns') \
        .filter(f"""
                size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
                AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
                AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""") \

    sloth__latest_snapshot_tables_df = spark \
        .read \
        .format('mongodb')\
        .option("uri", "mongodb://127.0.0.1:27017/") \
        .option("database", "sloth") \
        .option("collection", "latest_snapshot_tables" if not small else "latest_snapshot_tables_small") \
        .load() \
        .select('_id', '_id_numeric', 'content', 'numeric_columns') \
        .filter(f"""
                size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
                AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
                AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""") \
        .rdd \
        .map(list) \
        .union(
            optitab__turl_training_set_df.rdd.map(list)    
        )
    
    optitab__turl_training_set_df.unpersist()   # free memory used by the dataframe (is this really useful?)

    def prepare_tuple(t):
        nonlocal mode
        # t = (_id, _id_numeric, content, numeric_columns)
        return [t[1], _create_token_set(t[2], mode, t[3])]    
    
    print('Start creating inverted index and integer sets...')
    token_sets = sloth__latest_snapshot_tables_df \
        .map(
            # from MongoDB directly
            # (_id, _id_numeric, content, numeric_columns) -> (_id_numeric, [token1, token2, token3, ...])
            lambda t: prepare_tuple(t)
            ) \
            .flatMap(
                # (set_id, [tok1, tok2, tok3, ...]) -> [(tok1, set_id), (tok2, set_id), ...]
                lambda t: \
                    [
                        (token, t[0]) 
                        for token in t[1]
                    ]
            )

    posting_lists_sorted = token_sets \
        .groupByKey(
            # where the key is supposed to be the token itself in pairs (token, set_id)
        ) \
            .map(
                # (token, [set_idK, set_idJ, set_idM, ...]) -> (token, [set_id1, set_id2, set_id3, ..., set_idZ]) 
                lambda t: (t[0], sorted(list(t[1])))
            ) \
                .map(
                    # (token, set_ids) -> (token, set_ids, set_ids_hash)
                    lambda t: \
                        (
                            t[0], 
                            t[1], 
                            mmh3.hash_bytes(np.array(t[1]))
                        ) # is ok?
                ) \
                    .sortBy(
                        # t: (token, setIDs, hash)
                        lambda t: (len(t[1]), t[2], t[1])
                    ) \
                        .zipWithIndex() \
                            .map(
                                # t: ((rawToken, sids, hash), tokenIndex) -> (token_id, (raw_token, set_ids, set_ids_hash))
                                lambda t: (t[1], t[0])
                            ) \
                                .persist(pyspark.StorageLevel.MEMORY_ONLY)

    # create the duplicate groups
    duplicate_group_ids = posting_lists_sorted \
        .map(
            # t: (tokenIndex, (rawToken, sids, hash)) -> (token_index, (sids, hash))
            lambda t: (t[0] + 1, (t[1][1], t[1][2]))
        ) \
            .join(posting_lists_sorted) \
                .map(
                    # if the lower and upper posting lists are different, then the upper posting list
                    # belong to a new group, and the upper token index is the new group's
                    # starting index
                    # (tokenIndexUpper, ((sidsLower, hashLower), (_, sidsUpper, hashUpper)))
                    lambda t: 
                        -1 if t[1][0][1] == t[1][1][2] and t[1][0][0] == t[1][1][1] else t[0]
                ) \
                    .filter(
                        # add the first group's starting index, which is 0, and then
                        # create the group IDs
                        lambda i: i > 0
                    ) \
                        .union(spark.sparkContext.parallelize([0, posting_lists_sorted.count()])) \
                            .sortBy(lambda i: i) \
                                .zipWithIndex() \
                                    .map(
                                        # returns a mapping from group ID to the
                                        # starting index of the group
                                        # (startingIndex, GroupID) -> (GroupID, startingIndex)
                                        lambda t: (t[1], t[0])
                                    )    
    
    # generating all token indexes of each group
    token_group_ids = duplicate_group_ids \
        .join( # (GroupIDLower, startingIndexLower) JOIN (GroupIDUpper, startingIndexUpper) 
            duplicate_group_ids \
                .map(
                    # (GroupID, startingIndexUpper) -> (GroupID, startingIndexUpper)
                    lambda t: (t[0] - 1, t[1])
                )
            )
    
    token_group_ids = token_group_ids \
                .flatMap(
                    # GroupID, (startingIndexLower, startingIndexUpper) -> (tokenIndex, groupID)
                    lambda t: [(i, t[0]) for i in range(t[1][0], t[1][1])]
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)

    # join posting lists with their duplicate group IDs
    posting_lists_with_group_ids = posting_lists_sorted \
        .join(token_group_ids) \
            .map(
                # (tokenIndex, ((rawToken, sids, _), gid)) -> (token_index, (group_id, raw_token, sids))
                lambda t: (t[0], (t[1][1], t[1][0][0], t[1][0][1]))
            )
    
    # STAGE 2: CREATE INTEGER SETS
    # Create sets and replace text tokens with token index
    integer_sets = posting_lists_with_group_ids \
        .flatMap(
            # (tokenIndex, (_, _, sids))
            lambda t: [(sid, t[0]) for sid in t[1][2]]        
        ) \
            .groupByKey() \
                .map(
                    # (sid, tokenIndexes)
                    lambda t: (
                        t[0], 
                        sorted(t[1])
                    )
                )
    
    # STAGE 3: CREATE THE FINAL POSTING LISTS
    # Create new posting lists and join the previous inverted
    # lists to obtain the final posting lists with all the information
    posting_lists = integer_sets \
        .flatMap(
            # (sid, tokens)
            lambda t:
                [
                    (token, (t[0], len(t[1]), pos))
                    for pos, token in enumerate(t[1])
                ]
        ) \
            .groupByKey() \
                .map(
                    # (token, sets)
                    lambda t: (
                        t[0], 
                        sorted(t[1], 
                            key=lambda s: s[0]
                            )
                    )
                ) \
                    .join(posting_lists_with_group_ids) \
                        .map(
                            # (token, (sets, (gid, rawToken, _))) -> (token, rawToken, gid, sets)
                            lambda t: (t[0], t[1][1][1], t[1][1][0], t[1][0])
                        )
    
    # STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS
    # to load directly into the database

    def _set_format_psql(t):
        sid, indices = t
        return (sid, len(indices), len(indices), indices)
    
    def _postinglist_format_psql(t):
        token, raw_token, gid, sets = t
        byteatoken = binascii.hexlify(bytes(str(raw_token), 'utf-8'))
        set_ids =  [s[0] for s in sets]
        set_sizes = [s[1] for s in sets]
        set_pos = [s[2] for s in sets]

        return (token, len(sets), gid, 1, raw_token, byteatoken, set_ids, set_sizes, set_pos)

    integer_sets.map(
        lambda t: _set_format_psql(t)
    ).toDF(schema=['id', 'size', 'num_non_singular_token', 'tokens']).write.jdbc(url, test_name + '_sets', 'overwrite', properties)

    posting_lists.map(
        lambda t: _postinglist_format_psql(t)
    ).toDF(schema=[
        'token', 'frequency', 'duplicate_group_id', 'duplicate_group_count', 
        'str_token', 'raw_token', 'set_ids', 'set_sizes', 'match_positions'
        ]
    ).write.jdbc(url, test_name + '_inverted_lists', 'overwrite', properties)



def get_tables_statistics_from_mongodb(
        wikitables_coll:pymongo.collection.Collection, 
        table_stat_file):
    data = []
    for document in tqdm(wikitables_coll.find(), total=wikitables_coll.estimated_document_count()):
        _id, content = document.get('_id'), document.get('content')
        rows = len(content)
        columns = len(content[0])
        size = rows * columns
        distincts = len({token for row in content for token in row})
        nan = len([token for row in content for token in row if not token or pd.isna(token)])
        nan_p = nan / size
        
        data.append([_id, rows, columns, size, distincts, nan, nan_p])
    stat = pl.DataFrame(data, schema=['tab_id', 'rows', 'cols', 'size', 'distincts', 'nan', '%nan'])
    stat.write_csv(table_stat_file)


def sample_queries(
    output_query_json,
    nsamples,
    small:bool,
    tables_thresholds:dict[str, int]
    ):

    MIN_ROW =     tables_thresholds['min_rows']
    MAX_ROW =     tables_thresholds['max_rows']
    MIN_COLUMN =  tables_thresholds['min_columns']
    MAX_COLUMN =  tables_thresholds['max_columns']
    MIN_AREA =    tables_thresholds['min_area']
    MAX_AREA =    tables_thresholds['max_area']
    
    mongoclient = pymongo.MongoClient()
    if not small:
        turlcoll = mongoclient.optitab.turl_training_set
        snapcoll = mongoclient.sloth.latest_snapshot_tables
    else:
        turlcoll = mongoclient.optitab.turl_training_set_small
        snapcoll = mongoclient.sloth.latest_snapshot_tables_small

    turl_samples = turlcoll.aggregate(
        [ {"$sample": {"size": nsamples // 2} } ]
    )

    snap_samples = snapcoll.aggregate(
        [ {"$sample": {"size": nsamples // 2} } ]
    )

    samples = [
        {'_id': t['_id'], '_id_numeric': t['_id_numeric'], 'content': t['content'], 'numeric_columns': t['numeric_columns']} 
        for t in list(turl_samples) + list(snap_samples)
        if MIN_ROW <= len(t['content']) <= MAX_ROW \
        and MIN_COLUMN <= len(t['content'][0]) <= MAX_COLUMN \
        and MIN_AREA <= len(t['content']) * len(t['content'][0]) <= MAX_AREA
    ]

    print(f'Sampled {len(samples)} tables')
    with open(output_query_json, 'w') as wf:
        json.dump(
            samples,
            wf,
            indent=1
        )
    return len(samples)


def get_query_ids_from_query_file(query_file):
    with open(query_file) as fr:
        return [t['_id_numeric'] for t in json.load(fr)]
    

@print_info(msg_before='Starting JOSIE tests...', msg_after='Completed.')
def josie_test(josie_dbname, test_name, results_directory, k):
    GOPATH = os.environ['GOPATH']
    josie_cmd_dir = f'{GOPATH}/src/github.com/ekzhu/josie/cmd'
    os.chdir(josie_cmd_dir)

    os.system(f'go run {josie_cmd_dir}/sample_costs/main.go \
                --pg-database={josie_dbname} \
                --test_tag={test_name} \
                --pg-table-queries={test_name}_queries')

    os.system(f'go run {josie_cmd_dir}/topk/main.go \
                --pg-database={josie_dbname} \
                --test_tag={test_name} \
                --output={results_directory} \
                    --k={k}')