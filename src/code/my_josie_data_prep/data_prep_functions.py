import mmh3
import spacy
import pyspark
import jsonlines
import pandas as pd
from pandas.api.types import is_numeric_dtype

from code.utils.settings import DefaultPath
from code.utils.utils import print_info, rebuild_table, my_tokenizer



def _infer_column_type(column: list, check_column_threshold:int=3, nlp=None|spacy.Language) -> str:
    NUMERICAL_NER_TAGS = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}
    parsed_values = nlp.pipe(column)
    ner_tags = {token.ent_type_ for cell in parsed_values for token in cell}
    ner_tags = (ner_tags.pop() for _ in range(check_column_threshold))
    rv = sum(1 if tag in NUMERICAL_NER_TAGS else -1 for tag in ner_tags)
    return 'real' if rv > 0 else 'text'
    

def _create_set_with_inferring_column_dtype(df: pd.DataFrame, check_column_threshold:int=3, nlp=None|spacy.Language):
    tokens_set = set()
    for i in range(len(df.columns)):
        try:
            if is_numeric_dtype(df.iloc[:, i]) or _infer_column_type(df.iloc[:, i].to_list()) == 'real':
                continue
        except: # may give error with strange tables
            continue
        for token in df.iloc[:, i].unique():
            tokens_set.add(token)

    return tokens_set


def _create_set_with_my_tokenizer(df: pd.DataFrame):
    tokens_set = set()
    for i in range(len(df.columns)):
        if is_numeric_dtype(df.iloc[:, i]):
            continue
        for token in df.iloc[:, i].unique():
            for t in my_tokenizer(token, remove_numbers=True):
                tokens_set.add(t.lower())

    return tokens_set


@print_info(msg_before='Extracting intitial sets...', msg_after='Completed.', time=True)
def extract_starting_sets_from_tables(tables_file, final_set_file, ntables_to_load_as_set=10, with_:str='mytok', **kwargs):
    if with_ not in {'mytok', 'infer'}:
        raise AttributeError(f"Parameter with_ must be a value in {{'mytok', 'infer'}}")
    if with_ == 'infer':
        nlp = spacy.load('en_core_web_sm')

    with jsonlines.open(tables_file) as table_reader:
        with open(final_set_file, 'w') as set_writer:
            for i, json_table in enumerate(table_reader):
                if i >= ntables_to_load_as_set:
                    break                
                table = rebuild_table(json_table).convert_dtypes()
                if with_ == 'mytok':
                    table_set = _create_set_with_my_tokenizer(table)
                else:
                    table_set = _create_set_with_inferring_column_dtype(table, nlp=nlp, **kwargs)
                set_writer.write(
                    str(i) + ',' + ','.join(table_set) + '\n'
                )
                # print(i, len(table_set))


@print_info(msg_before='Creating raw tokens...', msg_after='Completed.')
def create_raw_tokens(input_set_file, output_raw_tokens_file, single_txt=True, spark_context=None):
    if not spark_context:
        conf = pyspark.SparkConf() \
            .setAppName('CreateIndex') \
                # .set('spark.executor.memory', '100g') \
                # .set('spark.driver.memory', '5g')
        spark_context = pyspark.SparkContext(conf=conf)
    
    skip_tokens = set()

    sets = spark_context.textFile(input_set_file) \
        .map(
            lambda line: line.split(',')
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







@print_info(msg_before='Creating integer sets and inverted index...', msg_after='Completed.', time=True)
def create_index(input_set_file, output_integer_set_file, output_inverted_list_file, spark_context=None):
    if not spark_context:
        conf = pyspark.SparkConf() \
            .setAppName('CreateIndex')
                # .set('spark.executor.memory', '100g') \
                # .set('spark.driver.memory', '5g')

        spark_context = pyspark.SparkContext(conf=conf)

    skip_tokens = set()

    # STAGE 1: BUILD TOKEN TABLE
    # print('STAGE 1: BUILD TOKEN TABLE')
    # Load sets and filter out removed token
    sets = spark_context \
        .textFile(input_set_file) \
            .map(
                lambda line: line.split(',')
            ) \
                .map(
                    lambda line: (
                        int(line[0]),
                        [token for token in line[1:] if token not in skip_tokens]
                    )
                )

    token_sets = sets.flatMap(
            lambda sid_tokens: \
                [
                    (token, sid_tokens[0]) 
                    for token in sid_tokens[1]
                ]
        )

    def compare(a:tuple[int,list[int]], b:tuple[int,list[int]]):
        la, lb = len(a[1]), len(b[1])

        if la != lb: 
            return -1 if la < lb else 0 if la == lb else 1
        elif not a or not b:
            return 0
        elif a[0] != b[0]:
            return -1 if a[0] < b[0] else 0 if a[0] == b[0] else 1 
        else:
            for (x, y) in zip(a[1], b[1]):
                if x != y:
                    return -1 if x < y else 0 if x == y else 1
            return 0


    posting_lists_sorted = token_sets \
        .groupByKey() \
            .map(
            lambda token_sids: (token_sids[0], sorted(list(token_sids[1])))
            ) \
                .map(
                    lambda token_sids: (token_sids[0], token_sids[1], mmh3.hash_bytes(bytes(token_sids[1]))) # is ok?
                ) \
                    .sortBy(
                        lambda tok_sids_hash: (len(tok_sids_hash[1]), tok_sids_hash[2], tok_sids_hash[1])
                    ) \
                        .zipWithIndex() \
                            .map(
                                # t: (rawToken, sids, hash), tokenIndex
                                lambda t: (t[1], (t[0][0], t[0][1], t[0][2]))
                            ) \
                                .persist(pyspark.StorageLevel.MEMORY_ONLY)

    def equal_arrays(a1, a2):
        return len(a1) == len(a2) and all(x1 == x2 for (x1, x2) in zip(a1, a2))

    duplicate_group_ids = posting_lists_sorted \
        .map(
            # tokenIndex, (rawToken, sids, hash)
            lambda t: (t[0] + 1, (t[1][1], t[1][2]))
        ) \
            .join(posting_lists_sorted) \
                .map(
                    # (tokenIndexUpper, ((sidsLower, hashLower), (_, sidsUpper, hashUpper)))
                    lambda t: 
                        -1 if equal_arrays(t[1][0][0], t[1][1][1]) and t[1][0][1] == t[1][1][2] else t[0]
                ) \
                    .filter(
                        lambda i: i > 0
                    ) \
                        .union(
                            spark_context.parallelize([0])
                        ) \
                            .sortBy(
                                lambda i: i
                            ) \
                                .zipWithIndex() \
                                    .map(
                                        # (startingIndex, GroupID)
                                        lambda t: (t[1], t[0])
                                    )

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
                            lambda token_index: (token_index, t[0]),
                            range(t[1][0], t[1][1])
                        )
                ).persist(pyspark.StorageLevel.MEMORY_ONLY)

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
    # print('STAGE 2: CREATE INTEGER SETS')

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

    # print('STAGE 3: CREATE THE FINAL POSTING LISTS')

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
                            # (token, (sets, (gid, rawToken, _)))
                            lambda t: (t[0], t[1][1][1], t[1][1][0], t[1][0])
                        )

    # STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS

    # print('STAGE 4: SAVE INTEGER SETS AND FINAL POSTING LISTS')
    def sets_format_string(t):
        sid, indices = t
        return "{}|{}|{}|{{{}}}\n".format(sid, len(indices), len(indices), ','.join(map(str, indices)))

    integer_sets = integer_sets.map(
        # (sid, indices)
        lambda t: sets_format_string(t) #f"{t[0]}, {','.join(map(str, t[1]))}\n"
    ).collect() #.saveAsTextFile(output_integer_set_file)


    def postlist_format_string(t):
        token, raw_token, gid, sets = t
        freq = len(sets)
        set_ids = ','.join([str(s[0]) for s in sets])
        set_sizes = ','.join([str(s[1]) for s in sets])
        set_pos = ','.join([str(s[2]) for s in sets])

        return "{}|{}|{}|{}|{{{}}}|{{{}}}|{{{}}}|{}\n" \
            .format(
                token, freq, gid, 1, set_ids, set_sizes, set_pos, bytes(raw_token.encode('utf-8'))
            )

    posting_lists = posting_lists.map(
        # token, rawToken, gid, sets
        # lambda t: f"{t[0]} {t[1]} {t[2]} {' '.join(map(str, t[3]))}\n"
        lambda t: postlist_format_string(t)
    ).collect() # .saveAsTextFile(output_inverted_list_file)


    # since the data now is really small, a single file is ok (no Spark partitioning)

    with open(output_integer_set_file, 'w') as f:
        f.writelines(integer_sets) 


    with open(output_inverted_list_file, 'w') as f:
        f.writelines(posting_lists) 


if __name__ == '__main__':
    ROOT_TEST_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/josie-test/'
    input_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set'
    output_integer_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set-2'
    output_inverted_list_file = ROOT_TEST_DIR + 'sloth_tables_n10.inverted-list'
