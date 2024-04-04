import itertools
import re
import string
from typing import List
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import compress_fasttext
import fasttext

from nltk.corpus import stopwords

# tr = str.maketrans('', '', string.punctuation.replace('-', '')) # to keep minus sign
stopwords_set = set(stopwords.words('english'))


def my_tokenizer(s: str, remove_numbers=False):
    #if type(s) is not str:
    #    return str(s)
    s = str(s)
    if not remove_numbers:
        return [            
            x for x in re.findall(r'\b([a-z]+|\d{1}|\d{2}|\d{3}|\d{4})\b', s.lower()) 
            if x not in stopwords_set
        ]
    else:
        return [
            x for x in re.findall(r'[a-z]+', s.lower())
            if x not in stopwords_set
        ]


def np_cosine_similarity(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))


def drop_columns_with_only_nan(df: pd.DataFrame, threshold:float=0.8):
    to_drop = []
    for col in df.columns:
        if df[col].notna().shape[0] < df[col].shape[0] * threshold:
            to_drop.append(col)
    
    return df.drop(to_drop, axis=1)




class TableEncoder:
    available_models = {
        'cc.en.300.compressed.bin': '/home/giovanni/unimore/TESI/src/models/fastText/cc.en.300.compressed.bin'
    }

    def __init__(self, 
                 model: str|compress_fasttext.compress.CompressedFastTextKeyedVectors|fasttext.FastText._FastText|None='cc.en.300.compressed.bin',
                 model_path: str|None=None
                 ):

        if model:
            if type(model) is str:
                if model in TableEncoder.available_models:
                    self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(TableEncoder.available_models[model])
                else:
                    raise KeyError(f'Unknown model code: {model}')
            else:
                self.model = model
        elif model_path:
            self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)
        else:
            raise Exception('You must pass a specification for the model')
        
        self.model_size = self.model.vector_size
            
    def embedding_cell(self, cell):
        return \
            np.mean( 
                [self.model.get_vector(token) \
                 for token in cell], # compute cell embedding
                axis=0
                ) \
                    if type(cell) in [str, list] and len(cell) > 0 \
                        else np.zeros(self.model_size)

    def embedding_column(self, column: list):
        return \
            np.mean(                
                np.array(
                    [self.embedding_cell(cell) for cell in column if type(cell) in [str, list] and len(cell) > 0]
                ),
                axis=0
            )
    
    def embedding_row(self, row: list):
        return \
            np.mean(
                np.array(
                    [self.embedding_cell(cell) for cell in row if type(cell) in [str, list] and len(cell) > 0],
                ),
                axis=0
            )

    def create_row_embeddings(self, df: pd.DataFrame, add_label=False, remove_numbers=False):
        labels_embedding = list(map(my_tokenizer, df.columns)) if add_label else None
        embeddings = []
        for _, row in df.iterrows():
            if add_label:
                e = self.embedding_row(row.apply(my_tokenizer, args=(remove_numbers, )).to_list() + labels_embedding)
            else:
                e = self.embedding_row(row.apply(my_tokenizer, args=(remove_numbers, )).to_list())

            if type(e) is np.float64: # case the embedding is nan
                embeddings.append(np.zeros(self.model_size))
            else:
                embeddings.append(e)
        return embeddings 

    def create_column_embeddings(self, df: pd.DataFrame, add_label=False, remove_numbers=False):
        embeddings = []
        for i, column in enumerate(df.columns):
            if add_label:
                e = self.embedding_column(df.iloc[:, i].apply(my_tokenizer, args=(remove_numbers, )).to_list() \
                                          + [my_tokenizer(column, remove_numbers)]) # why not simply my_tokenizer(column, remove_numbers)?
            else:
                e = self.embedding_column(df.iloc[:, i].apply(my_tokenizer, args=(remove_numbers, )).to_list())

            if type(e) is np.float64:
                embeddings.append(np.zeros(self.model_size))
            else:
                embeddings.append(e)
        
        return embeddings

    def create_embeddings(self, df: pd.DataFrame, 
                          on='columns', 
                          add_label:bool=False,
                          remove_numbers:bool=False):
        if on == 'rows':
            return self.create_row_embeddings(df, add_label, remove_numbers)
        elif on == 'columns':
            return self.create_column_embeddings(df, add_label, remove_numbers)
        else:
            raise AttributeError(f"Parameter 'on' accepts only values 'rows' or 'columns', passed {on}.")

    def full_embedding(self, 
                       df: pd.DataFrame, 
                       add_label=False, remove_numbers=False
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates and returns (in order) both row and column embeddings """
        # ok, the embeddings seem to be the same of only column/row version
        embedding_matrix = \
            np.array(
                [
                    [self.embedding_cell(my_tokenizer(cell, remove_numbers)) for cell in row]
                    for _, row in (pd.concat([df, pd.DataFrame([df.columns], columns=df.columns)]) if add_label else df).iterrows() 
                ]
            )
        
        column_embeddings = np.mean(embedding_matrix, axis=0)
        embedding_matrix = embedding_matrix[:-1, :, :] if add_label else embedding_matrix

        if add_label:
            labels_embedding = \
                np.repeat(
                    np.expand_dims(
                        np.array(
                            list(
                                map(self.embedding_cell, 
                                    map(lambda x: my_tokenizer(x, remove_numbers), df.columns)
                                    )
                                )
                            ),
                        0), 
                    df.shape[0], axis=0
                )
        
            embedding_matrix = np.concatenate((embedding_matrix, labels_embedding), axis=1)
        row_embeddings = np.mean(embedding_matrix, axis=1)
        return row_embeddings, column_embeddings



def compare_embeddings(df1: pd.DataFrame, df2: pd.DataFrame,
                          tabenc: TableEncoder,
                          on:str='columns',
                          sort_by_cosine_similarity:bool=True,
                          add_label:bool|List[bool]=False,
                          remove_numbers:bool|List[bool]=False,
                          delta:bool=False
                          ) -> pd.DataFrame:
    """
    Compare column/row embeddings of two datasets. Each embedding of df1 is compared with 
    each embedding of df2 (quadratic complexity) 
    """

    if on not in {'columns', 'rows'}:
        raise AttributeError(f"'on' parameter accepts only 'rows' or 'columns': {on} passed.")
    
    add_label = [add_label] if type(add_label) is bool else add_label
    remove_numbers = [remove_numbers] if type(remove_numbers) is bool else remove_numbers

    embeddings1, embeddings2 = [], []
    columns = []

    for al, rn in itertools.product(add_label, remove_numbers):
        tag = f"cosine similarity{'-wlabel' if al else ''}{'-nonum' if rn else ''}"
        columns.append(tag)
        
        embeddings1.append(tabenc.create_embeddings(df1, on, al, rn)),
        embeddings2.append(tabenc.create_embeddings(df2, on, al, rn))

    comparisons = pd.DataFrame(columns=['DF1', 'DF2'] + columns)

    for i in range(len(embeddings1[0])):
        for j in range(len(embeddings2[0])):
            cosim = []
            for k in range(len(columns)):
                cosim.append(
                    np_cosine_similarity(embeddings1[k][i], embeddings2[k][j])
                )
            
            idx_i, idx_j = (df1.columns[i], df2.columns[j]) if on == 'columns' else (i, j)
            comparisons.loc[len(comparisons)] = [idx_i, idx_j] + cosim
    
    if delta:
        new_columns = ['DF1', 'DF2', 'cosine similarity']
        columns.remove('cosine similarity')
        i = 1
        for c in columns:
            comparisons[f'delta#{i}'] = (comparisons['cosine similarity'] - comparisons[c]).apply(lambda x: format(x, ".3f"))
            new_columns.extend([c, f'delta#{i}'])
            i += 1
        comparisons = comparisons[new_columns]

    comparisons = comparisons.convert_dtypes().rename({'DF1': 'C1' if on == 'columns' else 'R1', 
                                                       'DF2': 'C2' if on == 'columns' else 'R2'}, axis=1)
    return comparisons if not sort_by_cosine_similarity \
        else comparisons.sort_values(by=comparisons.columns[2], ascending=False, ignore_index=True)


def show_most_similar_rows(compared_rows: pd.DataFrame, 
                           df1: pd.DataFrame, df2: pd.DataFrame, 
                           n=5, sorted=True):
    if not sorted:
        compared_rows = compared_rows.sort_values(by=compared_rows.columns[2], ascending=False, ignore_index=True)

    for i in range(n):
        r1, r2, cosim = compared_rows.loc[i]
        r1, r2 = int(r1), int(r2)
        s = df1.loc[r1]
        p = df2.loc[r2]
        print(f"#{i}: {cosim}\n\t{r1}: {' '.join([str(c) for c in s])}\n\t{r2}: {' '.join([str(c) for c in p])}")
        print()


