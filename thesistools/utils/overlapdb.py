from typing import Any

from sqlalchemy import (
    create_engine, 
    select, insert,
    MetaData, Table, Column, Integer
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from sqlalchemy import exc


class OverlapDB:
    """ Used only for testing, in order to avoid computing each time the SLOTH overlap """
    def __init__(self, table_name='results_table', url:Any|None=None, engine:Any|None=None, connection_info:dict|None=None):
        self.table_name = table_name
        if engine and url:
            self.url = url
            self.engine = engine
        else:
            self.url = URL.create(**connection_info)
            self.engine = create_engine(self.url)
        
        self.inner_engine = not (engine and url)

        self.results_table_name = table_name
        self.metadata = MetaData(self.engine)
        self.metadata.reflect()
        
    def create_table(self):
        Table(
            self.results_table_name, self.metadata,
            Column('r_id', Integer, primary_key=True),
            Column('s_id', Integer, primary_key=True),
            Column('overlap', Integer),
            keep_existing=True
        )
        self.metadata.create_all(self.engine)
        
    def add_overlaps(self, values:list[list[int,int,int]]):
        """
        Inserts a list of computed overlap, where each entry is a list of three elements:
        (r_id, s_id, overlap), assuming r_id < s_id
        """
        values = [{'r_id': r_id, 's_id': s_id, 'overlap': o} for r_id, s_id, o in values]
        table = self.metadata.tables[self.results_table_name]
        with Session(self.engine) as session:
            try:
                session.execute(insert(table).values(values))
                session.commit()
            except exc.IntegrityError:
                session.rollback()
                for value in values:
                    try:
                        session.execute(insert(table).values([value]))
                        session.commit()
                    except exc.IntegrityError:
                        session.rollback()
                        continue

    def lookup(self, r_id, s_id) -> int|None:
        """ 
        Returns the stored overlap for the pair (r_id, s_id), assuming that r_id < s_id 
        """
        table = self.metadata.tables[self.results_table_name]
        stmt = select(table.c.overlap).where(table.c.r_id == r_id, table.c.s_id == s_id)

        with Session(self.engine) as session:
            return session.execute(stmt).scalar()

    def clear(self):
        table = self.metadata.tables[self.results_table_name]
        table.drop(self.engine)

    def close(self):
        if self.inner_engine:
            self.engine.dispose()
        

if __name__ == '__main__':
    db_connection_info = {
        'drivername':   'postgresql',
        'database':     'JOSIEDB',
        'username':     'nanni',
        'password':     '',
        'port':         5442,
        'host':         '127.0.0.1',
    }
    table_name = 'test__full_overlap'

    print('Creating OverlapDB object...')
    resultsdb = OverlapDB(table_name, connection_info=db_connection_info)

    print('Clear already existent data...')
    resultsdb.clear()

    print('Creating overlap table...')
    resultsdb.create_table()
    
    values = [
        [0, 1, 22],
        [0, 2, 16],
        [1, 2, 34]
    ]

    print('Inserting new overlaps into the db...')
    resultsdb.add_overlaps(values)

    print('Closing connection...')
    resultsdb.close()

    print('Reopening the connection...')
    url = URL.create(**db_connection_info)
    engine = create_engine(url)
    resultsdb = OverlapDB(table_name, url, engine)
    
    print('Inserting new overlaps into the db (with duplicates!)...')
    values = [
        [1, 2, 36],
        [3, 4, 98]
    ]
    resultsdb.add_overlaps(values)

    print('Looking for some table pairs...')
    keys = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]
    for key in keys:
        print(f'{key} --> {resultsdb.lookup(*key)}')

    print('Closing connection...')
    resultsdb.close()
    engine.dispose()