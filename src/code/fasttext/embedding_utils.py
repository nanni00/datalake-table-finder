import re
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


import compress_fasttext
import fasttext


def my_tokenizer(s: str):
    if type(s) is not str:
        return str(s)
    return [re.sub('[^a-z]+', '', x) for x in s.lower().split()]

def np_cosine_similarity(a1, a2):
    return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))




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
                ) if type(cell) in [str, list] else np.zeros(self.model_size)

    def embedding_row(self, row):
        return \
            np.mean(
                [self.embedding_cell(cell) for cell in row],
                axis=0
            )

    def embedding_column(self, column):
        return \
            np.mean(
                [
                    self.embedding_cell(cell)
                    for cell in column
                ],
                axis=0
            )

    def create_row_embeddings(self, df: pd.DataFrame):
        return \
            (     
                self.embedding_row(row.apply(my_tokenizer)) for _, row in df.iterrows()
            )
    
    def create_column_embeddings(self, df: pd.DataFrame):
        return \
            (
                self.embedding_column(df[column].apply(my_tokenizer))                 
                for column in df.columns
            )



def compare_embeddings_of(df1: pd.DataFrame, df2: pd.DataFrame,
                          tabenc: TableEncoder,
                          on:str='columns',
                          sort_by_cosine_similarity=True
                          ) -> pd.DataFrame:
    """
    Compare column/row embeddings of two datasets. Each embedding of df1 is compared with 
    each embedding of df2 (quadratic complexity) 
    """

    if on not in {'columns', 'rows'}:
        raise AttributeError(f"'on' parameter accepts only 'rows' or 'columns': {on} passed.")
    
    if on == 'columns':
        comparisons = pd.DataFrame(columns=['C1', 'C2', 'cosine similarity'])
        emb1, emb2 = list(tabenc.create_column_embeddings(df1)), list(tabenc.create_column_embeddings(df2))
        for i, column1 in enumerate(df1.columns):
            for j, column2 in enumerate(df2.columns):

                emb_1_i, emb_2_j = emb1[i], emb2[j]
                cosim = np_cosine_similarity(emb_1_i, emb_2_j)
                comparisons.loc[len(comparisons)] = [column1, column2, cosim]
    else:
        comparisons = pd.DataFrame(columns=['R1', 'R2', 'cosine similarity'])
        emb1, emb2 = list(tabenc.create_row_embeddings(df1)), list(tabenc.create_row_embeddings(df2))
        for i, emb_s in enumerate(emb1):
            for j, emb_p in enumerate(emb2):
                cosim = np_cosine_similarity(emb_s, emb_p)
                comparisons.loc[len(comparisons)] = [i, j, cosim]

    comparisons = comparisons.convert_dtypes()
    return comparisons if not sort_by_cosine_similarity \
        else comparisons.sort_values(by='cosine similarity', ascending=False, ignore_index=True)


def show_most_similar_rows(compared_rows: pd.DataFrame, 
                           df1: pd.DataFrame, df2: pd.DataFrame, 
                           n=5, sorted=True):
    if not sorted:
        compared_rows = compared_rows.sort_values(by='cosine similarity', ascending=False, ignore_index=True)

    for i in range(n):
        r1, r2, cosim = compared_rows.loc[i]
        r1, r2 = int(r1), int(r2)
        s = df1.loc[r1]
        p = df2.loc[r2]
        print(f"#{i}: {cosim}\n\t{r1}: {' '.join([str(c) for c in s])}\n\t{r2}: {' '.join([str(c) for c in p])}")
        print()

