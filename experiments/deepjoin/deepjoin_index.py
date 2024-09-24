import multiprocessing as mp
from time import time

import faiss
import numpy as np
from datasets import Dataset


from tools.utils.datalake import SimpleDataLakeHelper
from tools.utils.misc import is_valid_table, whitespace_translator, lowercase_translator, punctuation_translator
from tools.utils.table_embedder import DeepJoinTableEmbedder

# model = SentenceTransformer.load('models/mpnet-base-all-nli-triplet/final')

def main():
    tabemb = DeepJoinTableEmbedder('models/mpnet-base-all-nli-triplet/final')

    dlh = SimpleDataLakeHelper('mongodb', 'wikitables', 'standard')

    table_obj = dlh.get_table_by_numeric_id(0)

    table = table_obj['content']
    bad_columns = table_obj['numeric_columns']

    print(tabemb.get_dimension())

    print()
    print(len(table[0]), bad_columns)
    print()
    print(tabemb.embed_columns(table, bad_columns, [], whitespace_translator, lowercase_translator, punctuation_translator).shape)


if __name__ == '__main__':
    main()

