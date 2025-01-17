import numpy as np

import fasttext
import compress_fasttext

from dltf.utils.tables import table_rows_to_columns, column_to_text



class TableEmbedder:
    def __init__(self, model_path) -> None:
        self.model_path = model_path

    def get_dimension(self) -> int:
        pass

    def embed_columns(self, table, valid_columns, *args):
        pass

    def embed_rows(self, table, valid_columns, *args):
        raise NotImplementedError()


def table_embedder_factory(table_embedder:str, model_path) -> TableEmbedder:
    match table_embedder:
        case 'ft'|'ftdist':
            return FastTextTableEmbedder(model_path)
        case 'cft'|'cftdist':
            return FastTextTableEmbedder(model_path, compressed=True)
        case _:
            raise ValueError('Unknown mode for table embedder with naive query mode: ' + table_embedder)


class FastTextTableEmbedder(TableEmbedder):
    def __init__(self, model_path, compressed=False):
        super().__init__(model_path)
        self.compressed = compressed
        if not compressed:
            self.model = fasttext.load_model(model_path)
        else:
            self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)

    def embed_columns(self, table, valid_columns, *args):
        """ 
        Return columns embeddings as a np.array of shape (#table_columns, #model_vector_size)

        For fastText.get_sentence_vector() see the github repo at src/fasttext.cc, lines 490-521
        it takes a sentence, splits it on blank spaces and for each word computes and normalises its 
        embedding; then gets the average of the normalised embeddings
        """

        blacklist, string_translators, string_patterns = args[0], args[1:]
        max_token_per_column = 512

        table = [column_to_text(column, string_translators, string_patterns, blacklist, max_seq_len=max_token_per_column) 
                 for column in table_rows_to_columns(table, 0, len(table[0]), valid_columns)]

        return np.array([self.model.get_sentence_vector(column) for column in table if column])


    def get_dimension(self):
        return self.model.get_dimension() if not self.compressed else self.model.vector_size
    