import os
from pprint import pprint
import sys
import pyspark
import mmh3
import pyspark.storagelevel
from code.utils.settings import DefaultPath


ROOT_TEST_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/josie-test/'

input_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set'
output_integer_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set-2'
output_inverted_list_file = ROOT_TEST_DIR + 'sloth_tables_n10.inverted-list'

### OCCHIO A TENERE QUESTA OPZIONE! 
if os.path.exists(output_integer_set_file):
    os.system(f'rm -rf {output_integer_set_file}')

### OCCHIO A TENERE QUESTA OPZIONE! 
if os.path.exists(output_inverted_list_file):
    os.system(f'rm -rf {output_inverted_list_file}')

skip_tokens = set()


conf = pyspark.SparkConf() \
    .setAppName('CreateIndex') \
        .set('spark.executor.memory', '100g') \
        .set('spark.driver.memory', '5g')

spark_context = pyspark.SparkContext(conf=conf)


# STAGE 1: BUILD TOKEN TABLE

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

def test_sort(x, y):
    print(x, '---', y)
    return 0
def test_t(t):
    print(len(t), t)
    return t

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
print()
print('Secondo pezzo')
print()
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
 
print()
print('Terzo pezzo')
print()

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

print()
print('Quarto pezzo')
print()

posting_lists_with_group_ids = posting_lists_sorted \
    .join(
        token_group_ids
    ) \
        .map(
            # (tokenIndex, ((rawToken, sids, _), gid))
            lambda t: (t[0], (t[1][1], t[1][0][0], t[1][0][1]))
        )

print()
print('Quinto pezzo')
print()

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
def test_gr(t):
    print(t)
    return t


print()
print('Sesto pezzo')
print()

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

integer_sets = integer_sets.map(
    # (sid, indices)
    lambda t: f"{t[0]} {' '.join(map(str, t[1]))}\n"
).collect() #.saveAsTextFile(output_integer_set_file)


posting_lists = posting_lists.map(
    # token, rawToken, gid, sets
    lambda t: f"{t[0]} {t[1]} {t[2]} {' '.join(map(str, t[3]))}\n"
).collect() # .saveAsTextFile(output_inverted_list_file)


# since the data now is really small, a single file is ok (no Spark partitioning)

with open(output_integer_set_file, 'w') as f:
    f.writelines(integer_sets) 


with open(output_inverted_list_file, 'w') as f:
    f.writelines(posting_lists) 
