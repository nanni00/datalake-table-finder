import os
import json
import random
from string import whitespace, digits, punctuation, ascii_uppercase, ascii_lowercase
from functools import reduce


whitespace_translator =     str.maketrans(whitespace, ' ' * len(whitespace))
digits_translator =         str.maketrans(digits, ' ' * len(digits))
punctuation_translator =    str.maketrans(punctuation, ' ' * len(punctuation))
lowercase_translator =      str.maketrans(ascii_uppercase, ascii_lowercase)

def containment(X:set, Y:set):
    return len(X.intersection(Y)) / min(len(X), len(Y)) if min(len(X), len(Y)) > 0 else 0

def jaccard(X:set, Y:set):
    return len(X.intersection(Y)) / len(X.union(Y)) if len(X.union(Y)) > 0 else 0

metrics = {
    'jaccard': jaccard,
    'containment': containment
}


def create_str_token(token, *translators):
    return reduce(lambda to, tr: str(to).translate(tr), translators, str(token)).strip()


def tokenize_column(column, *translators):
    """Extract distinct tokens from the column, apply on them the translators and remove empty strings """
    column = [create_str_token(token, *translators) for token in set(column)]
    return [token for token in column if set(token)]


def column_to_text(column, *translators):
    return ' '.join(tokenize_column(column, *translators))


def are_joinable_columns(c1: list, c2:list, t:float, metric:str):
    return int(metrics[metric](set(c1), set(c2)) >= t)


def table_rows_to_columns(table, num_headers, num_cols, bad_columns:list=[]):
    # Restructure the table as the list of its columns, ignoring the headers
    return [[row[i] for row in table[num_headers:]] for i in range(num_cols) if bad_columns[i] == 0]
