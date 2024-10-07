from sqlalchemy.orm import Session
from sqlalchemy.engine import URL, Engine
from sqlalchemy import (
    create_engine, MetaData, text,
    select, insert,
    Table, Column,
    Integer, LargeBinary, ARRAY)

from thesistools.utils.logging_handler import info


class JOSIEDBHandler:
    def __init__(self, tables_prefix, url:URL|None=None, engine:Engine|None=None, **connection_info) -> None:
        self.tables_prefix = tables_prefix

        self._SET_TABLE_NAME =               f'{self.tables_prefix}_sets'
        self._INVERTED_LISTS_TABLE_NAME =    f'{self.tables_prefix}_inverted_lists'
        self._SET_INDEX_NAME =               f'{self.tables_prefix}_sets_id_idx'
        self._INVERTED_LISTS_INDEX_NAME =    f'{self.tables_prefix}_inverted_lists_token_idx'
        self._QUERY_TABLE_NAME =             f'{self.tables_prefix}_queries'

        self._READ_LIST_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_list_cost_samples'
        self._READ_SET_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_set_cost_samples'

        if url and engine:
            self.url = url
            self.engine = engine
        else:
            self.url = URL.create(**connection_info)
            self.engine = create_engine(self.url)

        self.metadata = MetaData(self.engine)
        self.metadata.reflect()

    def drop_tables(self):
        info('Dropping JOSIE database tables...')
        for table_name in [
            self._INVERTED_LISTS_TABLE_NAME,
            self._SET_TABLE_NAME,
            self._QUERY_TABLE_NAME,
            self._READ_LIST_COST_SAMPLES_TABLE_NAME,
            self._READ_SET_COST_SAMPLES_TABLE_NAME]:
            try:
                table_obj = self.metadata.tables[table_name]
                table_obj.drop(self.engine)
            except KeyError:
                continue

    def create_tables(self):
        info('Creating JOSIE database tables...')
        Table(
            self._INVERTED_LISTS_TABLE_NAME,
            self.metadata,
            Column('token',                 Integer,        primary_key=True),
            Column('frequency',             Integer),
            Column('duplicate_group_id',    Integer),
            Column('duplicate_group_count', Integer),
            Column('raw_token',             LargeBinary),
            Column('set_ids',               ARRAY(Integer)),
            Column('set_sizes',             ARRAY(Integer)),
            Column('match_positions',       ARRAY(Integer)),
            keep_existing=True)

        Table(
            self._SET_TABLE_NAME,
            self.metadata,
            Column('id',                    Integer,        primary_key=True),
            Column('size',                  Integer),
            Column('num_non_singular_token',Integer),
            Column('tokens',                ARRAY(Integer)),
            keep_existing=True)

        Table(
            self._QUERY_TABLE_NAME,
            self.metadata,
            Column('id',                    Integer,        primary_key=True),
            Column('tokens',                ARRAY(Integer)),
            keep_existing=True)

        self.metadata.create_all(self.engine)

    def clear_query_table(self):
        query_table = self.metadata.tables[f"{self._QUERY_TABLE_NAME}"]
        with Session(self.engine) as session:
            ndel = session.query(query_table).delete()
            session.commit()
            
    def insert_data_into_query_table(self, table_ids:list[int]=None, table_id:int=None, tokens_ids:list[int]=None):
        with Session(self.engine) as session:
            set_table = self.metadata.tables[self._SET_TABLE_NAME]
            query_table = self.metadata.tables[self._QUERY_TABLE_NAME]
            if table_ids:
                session.execute(
                    insert(query_table)
                    .from_select(
                        select(set_table.c.id, set_table.c.tokens)
                        .where(set_table.c.id.in_(table_ids))
                    )
                )
            elif table_id and tokens_ids:
                session.execute(insert(query_table).values(table_id, tokens_ids))
          
    def get_statistics(self):
        q = f"""
            SELECT 
                i.relname 
                "table_name",
                indexrelname "index_name",
                pg_size_pretty(pg_total_relation_size(relid)) as "total_size",
                pg_size_pretty(pg_relation_size(relid)) as "table_size",
                pg_size_pretty(pg_relation_size(indexrelid)) "index_size",
                reltuples::bigint "estimated_table_row_count"
            FROM pg_stat_all_indexes i JOIN pg_class c ON i.relid = c.oid 
            WHERE i.relname LIKE '{self.tables_prefix}%'
            """
        with Session(self.engine) as session:
            return list(session.execute(text(q)))
    
    def cost_tables_exist(self):
        q = f"""
            SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_name   = '{self._READ_LIST_COST_SAMPLES_TABLE_NAME}'
               OR table_name = '{self._READ_SET_COST_SAMPLES_TABLE_NAME}'
            );
        """
        with Session(self.engine) as session:
            return list(session.execute(text(q)))
