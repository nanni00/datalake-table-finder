import re
import string
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import compress_fasttext
import fasttext

from nltk.corpus import stopwords

# tr = str.maketrans('', '', string.punctuation.replace('-', '')) # to keep minus sign
stopwords_set = set(stopwords.words('english'))


def my_tokenizer(s: str, keepnumbers=True):
    #if type(s) is not str:
    #    return str(s)
    s = str(s)
    if keepnumbers:
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

    def create_row_embeddings(self, df: pd.DataFrame, add_label=False, keepnumbers=True):
        labels_embedding = list(map(my_tokenizer, df.columns)) if add_label else None
        embeddings = []
        for _, row in df.iterrows():
            if add_label:
                e = self.embedding_row(row.apply(my_tokenizer, args=(keepnumbers, )).to_list() + labels_embedding)
            else:
                e = self.embedding_row(row.apply(my_tokenizer, args=(keepnumbers, )).to_list())

            if type(e) is np.float64: # case the embedding is nan
                embeddings.append(np.zeros(self.model_size))
            else:
                embeddings.append(e)
        return embeddings 
        #return \
        #    (     
        #        self.embedding_row(row.apply(my_tokenizer, args=(keepnumbers, )).to_list()) if not add_label \
        #            else self.embedding_row(row.apply(my_tokenizer, args=(keepnumbers, )).to_list() + labels_embedding) 
        #        
        #         for _, row in df.iterrows()
        #    )
    
    def create_column_embeddings(self, df: pd.DataFrame, add_label=False, keepnumbers=True):
        embeddings = []
        for column in df.columns:
            if add_label:
                e = self.embedding_column(df[column].apply(my_tokenizer, args=(keepnumbers, )).to_list() + [my_tokenizer(df[column].name, keepnumbers)])
            else:
                e = self.embedding_column(df[column].apply(my_tokenizer, args=(keepnumbers, )).to_list())

            if type(e) is np.float64:
                embeddings.append(np.zeros(self.model_size))
            else:
                embeddings.append(e)
        
        return embeddings
    
    def full_embedding(self, df: pd.DataFrame, add_label=False, keepnumbers=True):
        # ok, the embeddings seem to be the same of only column/row version
        embedding_matrix = \
            np.array(
                [
                    [self.embedding_cell(my_tokenizer(cell, keepnumbers)) for cell in row]
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
                            list(map(self.embedding_cell, 
                                     map(lambda x: my_tokenizer(x, keepnumbers), df.columns)
                                    )
                                )
                            ),
                        0), 
                    df.shape[0], axis=0
                ) \

            embedding_matrix = np.concatenate((embedding_matrix, labels_embedding), axis=1)
        
        row_embeddings = np.mean(embedding_matrix, axis=1)

        return row_embeddings, column_embeddings
        



def compare_embeddings_of(df1: pd.DataFrame, df2: pd.DataFrame,
                          tabenc: TableEncoder,
                          on:str='columns',
                          sort_by_cosine_similarity=True,
                          add_label=False,
                          keepnumbers=True
                          ) -> pd.DataFrame:
    """
    Compare column/row embeddings of two datasets. Each embedding of df1 is compared with 
    each embedding of df2 (quadratic complexity) 
    """

    if on not in {'columns', 'rows'}:
        raise AttributeError(f"'on' parameter accepts only 'rows' or 'columns': {on} passed.")
    
    if on == 'columns':
        comparisons = pd.DataFrame(columns=['C1', 'C2', 'cosine similarity'])
        emb1 = tabenc.create_column_embeddings(df1, add_label, keepnumbers)
        emb2 = tabenc.create_column_embeddings(df2, add_label, keepnumbers)
        for i, column1 in enumerate(df1.columns):
            for j, column2 in enumerate(df2.columns):

                emb_1_i, emb_2_j = emb1[i], emb2[j]
                cosim = np_cosine_similarity(emb_1_i, emb_2_j)
                comparisons.loc[len(comparisons)] = [column1, column2, cosim]
    else:
        comparisons = pd.DataFrame(columns=['R1', 'R2', 'cosine similarity'])
        emb1 = tabenc.create_row_embeddings(df1, add_label, keepnumbers)
        emb2 = tabenc.create_row_embeddings(df2, add_label, keepnumbers)
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
        compared_rows = compared_rows.sort_values(by=compared_rows.columns[2], ascending=False, ignore_index=True)

    for i in range(n):
        r1, r2, cosim = compared_rows.loc[i]
        r1, r2 = int(r1), int(r2)
        s = df1.loc[r1]
        p = df2.loc[r2]
        print(f"#{i}: {cosim}\n\t{r1}: {' '.join([str(c) for c in s])}\n\t{r2}: {' '.join([str(c) for c in p])}")
        print()


