import re
import numpy as np
import pandas as pd


import compress_fasttext
import fasttext

def my_tokenizer(s: str):
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
                [self.model.get_vector(token) if type(token) is str else np.zeros(self.model_size) \
                 for token in cell], # compute cell embedding
                axis=0
                ) if type(cell) is str else np.zeros(self.model_size)

    def embedding_row(self, row):
        return \
            np.mean(
                [self.embedding_cell(cell) for cell in row],
                axis=0
            )

    def create_row_embeddings(self, df: pd.DataFrame):
        return \
            (     
                self.embedding_row(row) for _, row in df.iterrows()
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

    def create_column_embeddings(self, df: pd.DataFrame):
        return \
            (
                self.embedding_column(df[column].apply(my_tokenizer))                 
                for column in df.columns
            )
