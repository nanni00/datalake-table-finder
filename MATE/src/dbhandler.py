from itertools import chain
from typing import List
import pandas as pd
# import vertica_python
from base import *
import os.path

from sqlalchemy import create_engine, MetaData, select, func
from sqlalchemy.orm import Session


class DBHandler:
    """Bloom filter using murmur3 hash function.

    Parameters
    ----------
    main_table_name : str
        Name of the main inverted index table in the database.
    """
    def __init__(self, mate_cache_path:str, main_table_name: str = 'main_tokenized', **connection_info):
        # conn_info = {
        #     'host': 'SERVER_IP_ADDRESS',
        #      'port': 5433,
        #      'user': 'USERNAME',
        #      'password': 'PASSWORD',
        #      'database': 'DATABASE_NAME',
        #      'session_label': 'some_label',
        #      'read_timeout': 6000,
        #      'unicode_error': 'strict',
        # }
        # connection = vertica_python.connect(**conn_info)
        # self.cur = connection.cursor()
        self.engine = create_engine(**connection_info)
        self.main_table_name = main_table_name
        self.mate_cache_path = mate_cache_path

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
            # print('clean values:', distinct_clean_values)
            metadata = MetaData(self.engine)
            metadata.reflect()
            table = metadata.tables[self.main_table_name]
            
            if top_k != -1:
                stmt = select(func.concat(table.c.tableid, "_", table.c.rowid, ";", table.c.colid, "_", table.c.tokenized, "$", table.c.superkey).distinct()) \
                    .where(
                        func.regexp_replace(
                            func.regexp_replace(table.c.tokenized, '\W+', ' '), ' +', ' '
                        ).in_(distinct_clean_values)
                    ).limit(top_k)
            else:
                stmt = select(func.concat(table.c.tableid, "_", table.c.rowid, ";", table.c.colid, "_", table.c.tokenized, "$", table.c.superkey).distinct()) \
                    .where(table.c.tokenized.in_(distinct_clean_values))

            with Session(self.engine) as session:
                pl = list(chain(*session.execute(stmt)))

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
        distinct_clean_values = list(set(joint_list))
        # joint_distinct_values = '\',\''.join(distinct_clean_values)
        
        #tables = '\',\''.join(list(set([x.split('_')[0] for x in joint_list])))
        tables = list(set([x.split('_')[0] for x in joint_list]))

        # rows = '\',\''.join(list(set([x.split('_')[1] for x in joint_list])))
        rows = list(set([x.split('_')[1] for x in joint_list]))
        metadata = MetaData(self.engine)
        metadata.reflect()
        table = metadata.tables[self.main_table_name]

        query = select(func.concat(table.c.tableid, '_', table.c.rowid), table.c.colid, table.c.tokenized) \
            .where(table.c.tableid.in_(tables)) \
            .where(table.c.rowid.in_(rows)) \
            .where(func.concat(table.c.tableid, '_', table.c.rowid).in_(distinct_clean_values))
            
        # query = f'SELECT concat(concat(tableid, \'_\'), rowid), colid, tokenized ' \
        #         f'FROM {self.main_table_name} ' \
        #         f'WHERE tableid IN (\'{tables}\') ' \
        #         f'AND rowid IN(\'{rows}\') ' \
        #         f'AND concat(concat(tableid, \'_\'), rowid) IN (\'{joint_distinct_values}\');'
        # self.cur.execute(query)
        # pl = self.cur.fetchall()
        # return pl

        with Session(self.engine) as session:
            return list(session.execute(query))
