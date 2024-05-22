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
    tab = Table()
    tab.columns = []
    tab.headers = []
    for i in range(len(df.columns)):
        tab.headers.append(df.columns[i])
        tab.columns.append(df.iloc[:, i].tolist())
    tab.shape = df.shape
    return tab


def from_polars(self, df:pl.DataFrame) -> Table:
    tab = Table()
    tab.columns = []
    tab.headers = []
    for i in df.schema: # unique values only here
        tab.headers.append(i)
        tab.columns.append(df.select(pl.col(i))[i].to_list())
    tab.shape = df.shape
    return tab


