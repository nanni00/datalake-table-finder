import random
import fasttext
import compress_fasttext
import numpy as np

from tools.sloth.utils import parse_table
from tools.utils.misc import prepare_token


class TableEmbedder:
    def __init__(self, model_path) -> None:
        self.mode_path = model_path


    def get_dimension(self) -> int:
        pass

    def embedding_table(self, table, numeric_columns, *args):
        pass



class CompressFastTextTableEmbedder(TableEmbedder):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
        self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(model_path)

    def get_dimension(self) -> int:
        return self.model.vector_size

    def embedding_table(self, table, numeric_columns, *args):
        blacklist = args
        max_token_per_column = 10 ** 4
        
        # with the any(column) we avoid that empty columns are embedded (btw they should have already been filtered previously)
        table = [column for i, column in enumerate(parse_table(table, len(table[0]), 0)) if numeric_columns[i] == 0 and any(column)]
        table = [column if len(column) <= max_token_per_column else random.sample(column, max_token_per_column) for column in table]

        return np.array([
            self.model.get_sentence_vector(
                ' '.join([prepare_token(token) for token in column if token not in blacklist])
                ) for column in table
            ]
        )


class FastTextTableEmbedder(TableEmbedder):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.model = fasttext.load_model(model_path)

    def embedding_table(self, table, numeric_columns, *args):
        """ 
        Return columns embeddings as a np.array of shape (#table_columns, #model_vector_size)

        For fastText.get_sentence_vector() see the github repo at src/fasttext.cc, lines 490-521
        it takes a sentence, splits it on blank spaces and for each word computes and normalises its 
        embedding; then gets the average of the normalised embeddings
        """

        blacklist = args
        max_token_per_column = 10 ** 4

        # with the any(column) we avoid that empty columns are embedded
        table = [column for i, column in enumerate(parse_table(table, len(table[0]), 0)) if numeric_columns[i] == 0 and any(column)]
        table = [column if len(column) <= max_token_per_column else random.sample(column, max_token_per_column) for column in table]
        table = [{prepare_token(token) for token in column if token not in blacklist} for column in table]

        return np.array([self.model.get_sentence_vector(' '.join(column)) for column in table])

    def get_dimension(self):
        return self.model.get_dimension()


class TaBERTTableEmbedder(TableEmbedder):
    def __init__(self, model_path) -> None:
        from tools.table_bert.table_bert import TableBertModel
        from tools.table_bert.table import Column, Table
        self.model = TableBertModel.from_pretrained(model_name_or_path=model_path)

    def get_dimension(self):
        return self.model.output_size

    def _prepare_table(self, table, numeric_columns):
        return Table(
            id=None,
            header=[Column(f'Column_{i}', type='text') for i, _ in enumerate(numeric_columns) if numeric_columns[i] == 0],
            data=[[prepare_token(cell) for i, cell in enumerate(row) if numeric_columns[i] == 0] for row in table]
        ).tokenize(self.model.tokenizer)

    def embedding_table(self, table, numeric_columns, context=""):
        _, column_embedding, _ = self.model.encode(
            contexts=[self.model.tokenizer.tokenize(context)],
            tables=[self._prepare_table(table, numeric_columns)])
        return column_embedding.detach().numpy()[0]

    def embedding_table_batch(self, tables, n_numeric_columns):
        tables = [self._prepare_table(table, numeric_columns) for (table, numeric_columns) in zip(tables, n_numeric_columns)]

        _, column_embeddings, _ = self.model.encode(
            contexts=[self.model.tokenizer.tokenize("") for _ in range(len(tables))],
            tables=tables)

        column_embeddings = column_embeddings.detach().numpy()
        # print(len(tables), len(n_numeric_columns), column_embeddings.shape)
        return [column_embeddings[i, :len(table.header), :] for i, table in enumerate(tables)]

