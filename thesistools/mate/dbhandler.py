import os.path
from typing import List
from itertools import chain

# import vertica_python
import pandas as pd
from sqlalchemy.orm import Session, Bundle
from sqlalchemy import create_engine, MetaData, select, func

from thesistools.mate.base import *

class DBHandler:
    """Bloom filter using murmur3 hash function.

    Parameters
    ----------
    main_table_name : str
        Name of the main inverted index table in the database.
    """
    def __init__(self, mate_cache_path:str, main_table_name: str = 'main_tokenized', **connection_info):
        self.main_table_name = main_table_name
        self.mate_cache_path = mate_cache_path
        self.engine = create_engine(pool_size=10, max_overflow=20, **connection_info)
        metadata = MetaData(self.engine)
        metadata.reflect()
        self.table = metadata.tables[self.main_table_name]


    def get_concatinated_posting_list(self,
                                      dataset_name: str,
                                      query_column_name: str,
                                      value_list: pd.Series,
                                      top_k: int = -1,
                                      database_request: bool = True) -> List[str]:
        """Fetches posting lists for top-k values.

        Parameters
        ----------
        dataset_name : str
            Name of the query dataset.

        query_column_name: str
            Name of the query column.

        value_list : pd.Series
            Values to fetch posting lists for.

        top_k : int
            Number of posting lists to fetch. -1 to fetch all.

        database_request : bool
            If true, posting lists are fetched from the database. Otherwise cached files are used (if existing).

        Returns
        -------
        List[str]
            Posting lists for values.
        """
        if os.path.isfile(f"{self.mate_cache_path}/{dataset_name}_{query_column_name}_concatenated_posting_list.txt")\
                and top_k == -1 and not database_request:
            pl = []
            with open(f"{self.mate_cache_path}/{dataset_name}_{ query_column_name}_concatenated_posting_list.txt", "r") as f:
                for line in f:
                    pl += [line.strip()]
            return pl
        else:
            distinct_clean_values = value_list.unique().tolist()
            if top_k != -1:
                stmt = (
                    # select(func.concat(self.table.c.tableid, "_", self.table.c.rowid, ";", self.table.c.colid, "_", self.table.c.tokenized, "$", self.table.c.superkey).distinct())
                    select(self.table.c.tableid, self.table.c.rowid, self.table.c.colid, self.table.c.tokenized, self.table.c.superkey)
                    .where(
                        func.regexp_replace(
                            func.regexp_replace(self.table.c.tokenized, '\W+', ' '), ' +', ' '
                        ).in_(distinct_clean_values))
                    .limit(top_k)
                )
            else:
                stmt = (
                    # select(func.concat(self.table.c.tableid, "_", self.table.c.rowid, ";", self.table.c.colid, "_", self.table.c.tokenized, "$", self.table.c.superkey).distinct()) \
                    select(self.table.c.tableid, self.table.c.rowid, self.table.c.colid, self.table.c.tokenized, self.table.c.superkey)
                    .distinct()
                    .where(self.table.c.tokenized.in_(distinct_clean_values))
                )

            # with Session(self.engine) as session:
            #     # pl = session.execute(stmt)
            
            with self.engine.connect() as connection:
                pl = connection.execute(stmt)

            if top_k == -1 and not database_request:
                with open(f"{self.mate_cache_path}/{dataset_name}_{query_column_name}_concatenated_posting_list.txt", "w") as f:
                    for s in pl:
                        f.write(str(s) + "\n")
            return pl

    def get_pl_by_table_and_rows(self, joint_list: List[str]) -> List[List[str]]:
        """Fetches posting lists a set of table_row_ids.

        Parameters
        ----------
        joint_list : List[str]
            List of table_row_ids.

        Returns
        -------
        List[List[str]]
            Posting lists for given table_row_ids.
        """
        distinct_clean_values = set(joint_list)
        tables, rows = zip(*map(lambda s: s.split('_'), joint_list))

        query = (
            select(func.concat(self.table.c.tableid, '_', self.table.c.rowid), self.table.c.colid, self.table.c.tokenized)
            .where(self.table.c.tableid.in_(tables))
            .where(self.table.c.rowid.in_(rows))
            .where(func.concat(self.table.c.tableid, '_', self.table.c.rowid).in_(distinct_clean_values))
        )

        with Session(self.engine) as session:
            return list(session.execute(query))
