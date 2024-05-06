import polars as pl
import pandas as pd


class Table:
    def __init__(self) -> None:
        self.columns = []
        self.headers = []
        self.shape = ()

    def from_pandas(self, df:pd.DataFrame):
        self.columns = []
        self.headers = []
        for i in range(len(df.columns)):
            self.headers.append(df.columns[i])
            self.columns.append(df.iloc[:, i].tolist())
        self.shape = df.shape

    def from_polars(self, df:pl.DataFrame):
        self.columns = []
        self.headers = []
        for i in df.schema: # unique values only here
            self.headers.append(i)
            self.columns.append(df.select(pl.col(i))[i].to_list())
        self.shape = df.shape

    def get_tuples(self):
        return [
            [column[i] for column in self.columns]
            for i in range(self.shape[0])
        ]