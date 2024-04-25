import os
import sys
import pyspark

from code.utils.settings import DefaultPath


conf = pyspark.SparkConf() \
    .setAppName('CreateIndex') \
        .set('spark.executor.memory', '100g') \
        .set('spark.driver.memory', '5g')

spark_context = pyspark.SparkContext(conf=conf)

ROOT_TEST_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/josie-test'

input_set_file = ROOT_TEST_DIR + 'sloth_tables_n10.set'
output_raw_tokens_file = ROOT_TEST_DIR + 'sloth_tables_n10.raw-token'

### OCCHIO A TENERE QUESTA OPZIONE! 
if os.path.exists(output_raw_tokens_file):
    os.system(f'rm -rf {output_raw_tokens_file}')

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

sets.flatMap(
    lambda sid_tokens: [(token, sid_tokens[0]) for token in sid_tokens[1]]
).map(
    lambda token_sid: f'{token_sid[0]} {token_sid[1]}'
).saveAsTextFile(output_raw_tokens_file)

