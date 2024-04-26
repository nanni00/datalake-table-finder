import jsonlines
import pandas as pd
from pandas.api.types import is_numeric_dtype

import spacy

from code.utils.settings import DefaultPath
from code.utils.utils import rebuild_table, my_tokenizer



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



if __name__ == '__main__':
    TABLES_DIR = DefaultPath.data_path.wikitables + 'threshold_r5-c2-a50/'
    TABLES_FILE = TABLES_DIR + 'sloth-tables-r5-c2-a50.jsonl'

    ntables_to_load_as_set = 10

    final_set_file = TABLES_DIR + f'sloth_tables_n{ntables_to_load_as_set}.set'
