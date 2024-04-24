from typing import Dict
import polars
import jsonlines
import pandas as pd
from pandas.api.types import is_numeric_dtype

import spacy
from spacy.lang.en import English

from code.utils.settings import DefaultPath
from code.utils.utils import rebuild_table, my_tokenizer

nlp = spacy.load('en_core_web_sm')

NUMERICAL_NER_TAGS = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'CARDINAL'}


def infer_column_type(column: list) -> str:
    parsed_values = nlp.pipe(column)
    ner_tags = (token.ent_type_ for cell in parsed_values for token in cell)

    rv = sum(1 if tag in NUMERICAL_NER_TAGS else -1 for tag in ner_tags)
    return 'real' if rv > 0 else 'text'
    

def create_set(df: pd.DataFrame):
    df = df.convert_dtypes()
    tokens_set = set()
    for i in range(len(df.columns)):
        try:
            if is_numeric_dtype(df.iloc[:, i]) or infer_column_type(df.iloc[:, i].to_list()) == 'real':
                continue
        except: # may give error with strange tables
            continue
        for token in df.iloc[:, i].unique():
            tokens_set.add(token)

    return tokens_set


def create_set_with_my_tokenizer(df: pd.DataFrame):
    df = df.convert_dtypes()
    tokens_set = set()
    for i in range(len(df.columns)):
        if is_numeric_dtype(df.iloc[:, i]):
            continue
        for token in df.iloc[:, i].unique():
            for t in my_tokenizer(token, remove_numbers=True):
                tokens_set.add(t.lower())

    return tokens_set



TABLES_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/'
TABLES_FILE = TABLES_DIR + 'sloth-tables-r5-c2-a50.jsonl'

ntables_to_load_as_set = 10

final_set_file = TABLES_DIR + f'sloth_tables_n{ntables_to_load_as_set}.set'

with jsonlines.open(TABLES_FILE) as table_reader:
    with open(final_set_file, 'w') as set_writer:
        print('Start reading tables...')
        for i, json_table in enumerate(table_reader):
            if i >= ntables_to_load_as_set:
                break
            # table_set = create_set(rebuild_table(json_table))
            table_set = create_set_with_my_tokenizer(rebuild_table(json_table))
            set_writer.write(
                str(i) + ',' + ','.join(table_set) + '\n'
            )
            print(i, len(table_set))
print('Completed.')


