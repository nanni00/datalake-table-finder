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
from collections import defaultdict

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

from tools.utils.utils import print_info
    

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


_TOKEN_TAG_SEPARATOR = '@#'


def _create_token_set(table, mode, numeric_columns):
    """ Create the token set for the given table 
    :param table: a list of list (row-view) of the table content 
    :param mode: how to create the token set, with "set" or "bag" semantic
    :param numeric_columns: a flag vector, where if the ith element is 1, this means that the 
                            ith column is numeric and its elements are skipped while creating the token set
    """
    if mode == 'set':
        return list({str(token).replace('|', ' ') for row in table for icol, token in enumerate(row) 
                     if not pd.isna(token) and token and numeric_columns[icol] == 0})
    elif mode == 'bag':
        counter = defaultdict(int) # is that better? More space but no sort operation            
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        return [_create_token_tag(str(token).replace('|', ' ')) for row in table for icol, token in enumerate(row)
                if not pd.isna(token) and token and numeric_columns[icol] == 0]
    else:
        raise Exception('Unknown mode: ' + str(mode))



# filtering on the IDs from the SLOTH results file should be deprecated, since now we'll work
# on a silver standard per test
@print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.')
def create_index(
    mode:str,
    original_sloth_results_file,
    output_integer_set_file, 
    output_inverted_list_file,
    thresholds:dict[str:int],
    tables_limit,
    small):

    MIN_ROW =     thresholds['min_rows']
    MAX_ROW =     thresholds['max_rows']
    MIN_COLUMN =  thresholds['min_columns']
    MAX_COLUMN =  thresholds['max_columns']
    MIN_AREA =    thresholds['min_area']
    MAX_AREA =    thresholds['max_area']
    
    all_sloth_results = pl.scan_csv(original_sloth_results_file)
    sloth_tables_ids = set( \
        pl.concat( \
            [all_sloth_results.select('r_id').collect().to_series(), 
            all_sloth_results.select('s_id').collect().to_series()]
            ) \
                .to_list()
        )
    
    @F.udf(returnType=BooleanType())
    def check_is_in_sloth_results(col1):
        return col1 in sloth_tables_ids
  
    # fine tune of executor/driver.memory?
    spark = SparkSession \
        .builder \
        .appName("Big Bang Testing with MongoDB") \
        .master("local[64]") \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0') \
        .config('spark.executor.memory', '100g') \
        .config('spark.driver.memory', '10g') \
        .config('spark.local.dir', '/data4/nanni/spark') \
        .getOrCreate()
    
    # adjusting logging level to error, avoiding warnings
    spark.sparkContext.setLogLevel("ERROR")

    # carico le tabelle usate nei risultati di SLOTH 
    # e che sono la base per fare poi la query
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

    # carico il database MongoDB dove c'è lo snapshot principale e inizio il processing effettivo
    sloth__latest_snapshot_tables_df = spark \
        .read \
        .format('mongodb')\
        .option("uri", "mongodb://127.0.0.1:27017/") \
        .option("database", "sloth") \
        .option("collection", "latest_snapshot_tables" if not small else "latest_snapshot_tables_small") \
        .load() \
        .limit(tables_limit) \
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
                                lambda t: (t[1], (t[0][0], t[0][1], t[0][2]))
                            ) \
                                .persist(pyspark.StorageLevel.MEMORY_ONLY)

    def equal_arrays(a1, a2):
        return len(a1) == len(a2) and all(x1 == x2 for (x1, x2) in zip(a1, a2))

    # create the duplicate groups
    duplicate_group_ids = posting_lists_sorted \
        .map(
            # t: (tokenIndex, (rawToken, sids, hash))
            lambda t: (t[0] + 1, (t[1][1], t[1][2]))
        ) \
            .join(posting_lists_sorted) \
                .map(
                    # if the lower and upper posting lists are different, then the upper posting list
                    # belong to a new group, and the upper token index is the new group's
                    # starting index
                    # (tokenIndexUpper, ((sidsLower, hashLower), (_, sidsUpper, hashUpper)))
                    lambda t: 
                        -1 if t[1][0][1] == t[1][1][2] and equal_arrays(t[1][0][0], t[1][1][1]) else t[0]
                ) \
                    .filter(
                        # add the first group's starting index, which is 0, and then
                        # create the group IDs
                        lambda i: i > 0
                    ) \
                        .union(
                            spark.sparkContext.parallelize([0])
                        ) \
                            .sortBy(
                                lambda i: i
                            ) \
                                .zipWithIndex() \
                                    .map(
                                        # returns a mapping from group ID to the
                                        # starting index of the group
                                        # (startingIndex, GroupID)
                                        lambda t: (t[1], t[0])
                                    )

    # generating all token indexes of each group
    token_group_ids = duplicate_group_ids \
        .join(
            duplicate_group_ids \
                .map(
                    # (GroupID, startingIndexUpper)
                    lambda t: (t[0] - 1, t[1])
                )
            ) \
                .flatMap(
                    # GroupID, (startingIndexLower, startingIndexUpper)
                    lambda t: map( 
                            lambda token_index: (token_index, t[0]), range(t[1][0], t[1][1])
                        )
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)

    # join posting lists with their duplicate group IDs
    posting_lists_with_group_ids = posting_lists_sorted \
        .join(
            token_group_ids
        ) \
            .map(
                # (tokenIndex, ((rawToken, sids, _), gid))
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
    # print('STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS')
    # to load directly into the database

    def sets_format_string(t):
        sid, indices = t
        return "{}|{}|{}|{{{}}}".format(sid, len(indices), len(indices), ','.join(map(str, indices)))

    def postlist_format_string(t):
        token, raw_token, gid, sets = t
        freq = len(sets)
        set_ids = ','.join([str(s[0]) for s in sets])
        set_sizes = ','.join([str(s[1]) for s in sets])
        set_pos = ','.join([str(s[2]) for s in sets])

        return "{}|{}|{}|{}|{{{}}}|{{{}}}|{{{}}}|{}" \
            .format(
                token, freq, gid, 1, set_ids, set_sizes, set_pos, binascii.hexlify(bytes(str(raw_token), 'utf-8'))
            )
    
    integer_sets.map(
        lambda t: sets_format_string(t)
    ).saveAsTextFile(output_integer_set_file)

    posting_lists.map(
        lambda t: postlist_format_string(t)
    ).saveAsTextFile(output_inverted_list_file)


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