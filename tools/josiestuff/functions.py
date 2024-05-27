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
    

_TOKEN_TAG_SEPARATOR = '@#'



def _create_token_set(data, mode):
    if mode == 'set':
        return list({str(token).replace('|', ' ') for column in data for token in column if not pd.isna(token) and token})
    elif mode == 'bag':
        counter = defaultdict(int) # is that better? More space but no sort operation            
        def _create_token_tag(token):
            counter[token] += 1
            return f'{token}{_TOKEN_TAG_SEPARATOR}{counter[token]}'
        return [_create_token_tag(str(token).replace('|', ' ')) for column in data for token in column if not pd.isna(token) and token]
    else:
        raise Exception('Unknown mode: ' + str(mode))



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



@print_info(msg_before='Creating raw tokens...', msg_after='Completed.')
def create_raw_tokens(
    input_set_file, 
    output_raw_tokens_file, 
    single_txt=False, 
    spark_context=None):

    if not spark_context:
        conf = pyspark.SparkConf() \
            .setAppName('CreateIndex') \
                # .set('spark.executor.memory', '100g') \
                # .set('spark.driver.memory', '5g')
        spark_context = pyspark.SparkContext(conf=conf)
    
    skip_tokens = set()

    sets = spark_context \
        .textFile(input_set_file) \
            .map(
                lambda line: line.split('|')
            ) \
                .map(
                    lambda line: (
                        int(line[0]),
                        [token for token in line[1:] if token not in skip_tokens]
                    )
                )

    if single_txt:
        sets = sets \
            .flatMap(
                lambda sid_tokens: \
                    [
                        (token, sid_tokens[0]) 
                        for token in sid_tokens[1]
                    ]
                ) \
                    .map(
                        lambda token_sid: f'{token_sid[0]} {token_sid[1]}\n'
                    ).collect()
        with open(output_raw_tokens_file, 'w') as f:
            f.writelines(sets)
    else:
        sets \
            .flatMap(
                lambda sid_tokens: 
                    [
                        (token, sid_tokens[0]) 
                        for token in sid_tokens[1]
                    ]
                ) \
                    .map(
                        lambda token_sid: f'{token_sid[0]} {token_sid[1]}'
                    ).saveAsTextFile(output_raw_tokens_file)



@print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.')
def create_index(
    mode:str,
    original_sloth_results_file,
    output_sampled_sloth_results_file,
    output_id_for_queries_file,
    output_tables_id_file,
    output_integer_set_file, 
    output_inverted_list_file,
    thresholds:dict[str:int],
    tables_limit
    ):

    MIN_ROW =     0       if 'min_rows'       not in thresholds else thresholds['min_rows']
    MAX_ROW =     999999  if 'max_rows'       not in thresholds else thresholds['max_rows']
    MIN_COLUMN =  0       if 'min_columns'    not in thresholds else thresholds['min_columns']
    MAX_COLUMN =  999999  if 'max_columns'    not in thresholds else thresholds['max_columns']
    MIN_AREA =    0       if 'min_area'       not in thresholds else thresholds['min_area']
    MAX_AREA =    999999  if 'max_area'       not in thresholds else thresholds['max_area']
    
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
        .appName("mongodbtest1") \
        .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:10.3.0') \
        .config('spark.executor.memory', '100g') \
        .config('spark.driver.memory', '12g') \
        .config('spark.local.dir', '/data4/nanni/spark') \
        .getOrCreate()

    # print('#sampled rows from sloth.latest_snapshot_tables:')
    # print(sets.count())

    # carico le tabelle usate nei risultati di SLOTH e che sono la base per fare poi la query
    wikitables_df = spark \
        .read \
        .format("mongodb") \
        .option ("uri", "mongodb://127.0.0.1:27017/") \
        .option("database", "optitab") \
        .option("collection", "turl_training_set") \
        .load() \
        .select('_id', 'content') \
        .filter(f"""
                size(content) BETWEEN {MIN_ROW} AND {MAX_ROW} 
                AND size(content[0]) BETWEEN {MIN_COLUMN} AND {MAX_COLUMN} 
                AND size(content) * size(content[0]) BETWEEN {MIN_AREA} AND {MAX_AREA}""") \
        .filter(check_is_in_sloth_results('_id'))

    # ua c'è un po' di roba salvata per fare le query in seguito
    print('Saving the filtered original table IDs as CSV...')
    wikitables_df.select('_id').toPandas().to_csv(output_id_for_queries_file, index=False)    

    print('Saving the SLOTH results which have both r_id and s_id in the filtered table IDs...')
    with open(output_id_for_queries_file) as fr:
        filtered_ids = set(map(str.strip, fr.readlines()[1:]))
        sub_res = all_sloth_results \
            .filter( \
                (pl.col('r_id').is_in(filtered_ids)) & \
                    (pl.col('s_id').is_in(filtered_ids)) \
                ) \
                    .collect() \
                        .sort(by='overlap_area')
        sub_res.write_csv(output_sampled_sloth_results_file)
        filtered_ids = sub_res = None

    # carico il database MongoDB dove c'è lo snapshot principale e inizio il processing effettivo
    sets = spark \
        .read \
        .format('mongodb')\
        .option("uri", "mongodb://127.0.0.1:27017/") \
        .option("database", "sloth") \
        .option("collection", "latest_snapshot_tables") \
        .load() \
        .limit(tables_limit) \
        .select('_id', 'content') \
        .rdd \
        .map(list)


    print('Merging RDDs...')
    sets = sets \
        .union(
            wikitables_df.rdd.map(list)    
        ).zipWithUniqueId()
        #.zipWithIndex()
    
    wikitables_df.unpersist()   # free memory used by the dataframe (is this really useful?)
    # print(f'Total RDD size: {sets.count()}')

    print('Saving mapping between JOSIE and wikitables (SLOTH) IDs...')
    sets.map(lambda t: f"{t[1]},{t[0][0]}").saveAsTextFile(output_tables_id_file)

    def prepare_tuple(t, mode):
        return [t[1], _create_token_set(t[0][1], mode)]    
    
    print('Start creating inverted index and integer sets...')
    token_sets = sets \
        .map(
            # from MongoDB directly
            # (_id, content) -> (_id, [token1, token2, token3, ...])
            lambda t: prepare_tuple(t, mode)
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
    sloth_results_csv_file,
    josie_sloth_id_file,
    table_statistics_file,
    output_query_json
    ):

    sloth_res =     pl.scan_csv(sloth_results_csv_file)
    table_stat =    pl.read_csv(table_statistics_file)
    josloth_ids =   pl.read_csv(
        josie_sloth_id_file, 
        has_header=False,
        dtypes={'column_1': pl.datatypes.Int64, 'column_2': pl.datatypes.String}
        ) \
        .rename({'column_1': 'josie_id', 'column_2': 'sloth_id'})

    n_sample_per_interval = 10
    jsim_intervals = [0, 0.3, 0.7, 1]
    # overlap_intervals = [50, 100, 5000]
    nan_threshold = 0.1

    sloth_res_samples = defaultdict(list)
    ids = {}

    def lookup_for_josie_int_id(sloth_str_id:str):
        return josloth_ids.row(by_predicate=pl.col('sloth_id') == sloth_str_id)[0]

    table_stat = table_stat.rename({'tab_id': 'r_id'})
    for jsim_min, jsim_max in zip(jsim_intervals[:-1], jsim_intervals[1:]):
        sample = sloth_res.filter((jsim_min < pl.col('jsim')) & (pl.col('jsim') < jsim_max)).collect().sample(n_sample_per_interval)
        sample = sample.join(table_stat, on='r_id', how='left', suffix='_r')
        
        table_stat = table_stat.rename({'r_id': 's_id'})
        sample = sample.join(table_stat, on='s_id', how='left', suffix='_s')
        table_stat = table_stat.rename({'s_id': 'r_id'})

        sample = sample.rename({'rows': 'rows_r', 'cols': 'cols_r', 'size': 'tabsize_r', 'nan': 'nan_r', 'distincts': 'distincts_r', '%nan': '%nan_r'})
        sample = sample.filter((pl.col('%nan_r') < nan_threshold) & (pl.col('%nan_s') < nan_threshold))
        sample = sample.sort(by='overlap_area')
        sloth_res_samples[(jsim_min, jsim_max)] = sample.to_numpy().tolist()
        ids[(jsim_min, jsim_max)] = list(map(lambda sloth_id: (sloth_id, lookup_for_josie_int_id(sloth_id)), \
                                             sample.select(['r_id', 's_id']).to_numpy().flatten().tolist()))

    with open(output_query_json, 'w') as writer:            
        json.dump(
            [
                {
                    "jsim": (jsim_min, jsim_max),
                    "sloth_res": sloth_res_samples[(jsim_min, jsim_max)],
                    "ids": ids[(jsim_min, jsim_max)]
                } 
                for jsim_min, jsim_max in zip(jsim_intervals[:-1], jsim_intervals[1:])
            ],
            writer, 
            indent=3
            )
        
    totsamples = [i for _, v in ids.items() for i in v]
    print(f"Sampled {len(totsamples)} IDs from {len(jsim_intervals) - 1} intervals of Jaccard similarity.")
    return totsamples


def get_query_ids_from_query_file(josie_sloth_id_file, query_file, convert_query_ids):
    josloth_ids =   pl.read_csv(
        josie_sloth_id_file, 
        has_header=False,
        dtypes={'column_1': pl.datatypes.Int64, 'column_2': pl.datatypes.String}
        ) \
        .rename({'column_1': 'josie_id', 'column_2': 'sloth_id'})
    
    def lookup_for_josie_int_id(sloth_str_id:str):
        return josloth_ids.row(by_predicate=pl.col('sloth_id') == sloth_str_id)[0]
    
    with open(query_file) as fr:
        query_stuff = json.load(fr)

    if not convert_query_ids:
        sampled_ids = [_id[1] for interval in query_stuff for _id in interval['ids']]
    else:
        sampled_ids = [lookup_for_josie_int_id(_id[0]) for interval in query_stuff for _id in interval['ids']]
    return sampled_ids




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