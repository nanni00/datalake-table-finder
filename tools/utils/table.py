import polars as pl
import pandas as pd


class Table:
    def __init__(self) -> None:
        self.columns = []
        self.headers = []
        self.shape = ()

    def get_tuples(self):
        return [
            [column[i] for column in self.columns]
            for i in range(self.shape[0])
        ]
    
    
def from_pandas(df:pd.DataFrame) -> Table:
    table = Table()
    table.columns = []
    table.headers = []
    for i in range(len(df.columns)):
        table.headers.append(df.columns[i])
        table.columns.append(df.iloc[:, i].tolist())
    table.shape = df.shape


def from_polars(self, df:pl.DataFrame) -> Table:
    table = Table()
    table.columns = []
    table.headers = []
    for i in df.schema: # unique values only here
        table.headers.append(i)
        table.columns.append(df.select(pl.col(i))[i].to_list())
    table.shape = df.shape


