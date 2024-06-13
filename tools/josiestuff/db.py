import os
import psycopg
import psycopg.sql

from tools.utils.utils import print_info



class JosieDB:
    def __init__(self, dbname, table_prefix) -> None:
        self.dbname = dbname
        self.tprefix = table_prefix

        self._dbconn = None
        self._dbcur = None

        self._SET_TABLE_NAME =               f'{self.tprefix}_sets'
        self._INVERTED_LISTS_TABLE_NAME =    f'{self.tprefix}_inverted_lists'
        self._SET_INDEX_NAME =               f'{self.tprefix}_sets_id_idx'
        self._INVERTED_LISTS_INDEX_NAME =    f'{self.tprefix}_inverted_lists_token_idx'
        self._QUERY_TABLE_NAME =             f'{self.tprefix}_queries'

        self._READ_LIST_COST_SAMPLES_TABLE_NAME = f'{self.tprefix}_read_list_cost_samples'
        self._READ_SET_COST_SAMPLES_TABLE_NAME = f'{self.tprefix}_read_set_cost_samples'

    @print_info(msg_before='Opening connection to the PostgreSQL database...')
    def open(self):
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}")
        self._dbcur = None

    @print_info(msg_before='Closing connection...')
    def close(self):
        self._dbconn.close()

    def _commit_dec(f):
        def inner(self, *args, **kwargs):
            self._dbcur = self._dbconn.cursor(row_factory=psycopg.rows.dict_row)
            res = f(self, *args, **kwargs)
            self._dbcur.close()
            self._dbconn.commit()
            return res
        return inner

    @_commit_dec
    @print_info(msg_before='Dropping tables...')
    def drop_tables(self, all=False):    
        self._dbcur.execute(
            f"""
            DROP TABLE IF EXISTS {self._INVERTED_LISTS_TABLE_NAME};
            DROP TABLE IF EXISTS {self._SET_TABLE_NAME};
            DROP TABLE IF EXISTS {self._QUERY_TABLE_NAME};
            """        
        )

        if all:
            self._dbcur.execute(
                f"""
                DROP TABLE IF EXISTS {self._READ_LIST_COST_SAMPLES_TABLE_NAME};
                DROP TABLE IF EXISTS {self._READ_SET_COST_SAMPLES_TABLE_NAME};
                """
            )

    @_commit_dec
    @print_info(msg_before='Creating database tables...')
    def create_tables(self):
        self._dbcur.execute(
            f"""              
            CREATE TABLE {self._INVERTED_LISTS_TABLE_NAME} (
                token integer NOT NULL,
                frequency integer NOT NULL,
                duplicate_group_id integer NOT NULL,
                duplicate_group_count integer NOT NULL,
                raw_token bytea NOT NULL,
                set_ids integer[] NOT NULL,
                set_sizes integer[] NOT NULL,
                match_positions integer[] NOT NULL
            );        
            """
        )

        self._dbcur.execute(
            f"""          
                CREATE TABLE {self._SET_TABLE_NAME} (
                    id integer NOT NULL,
                    size integer NOT NULL,
                    num_non_singular_token integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )

        self._dbcur.execute(
            f"""
                CREATE TABLE {self._QUERY_TABLE_NAME} (
                    id integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )

    @_commit_dec
    @print_info(msg_before='Clearing query table...')
    def clear_query_table(self):
        self._dbcur.execute(
            f"""
                TRUNCATE {self._QUERY_TABLE_NAME}
            """
        )
            
    @_commit_dec
    @print_info(msg_before='Inserting queries...')
    def insert_data_into_query_table(self, table_ids:list[int]):
        # maybe is better to translate all in postgresql...
        self._dbcur.execute(
            f"""
            INSERT INTO {self._QUERY_TABLE_NAME} SELECT id, tokens FROM {self._SET_TABLE_NAME} WHERE id in {tuple(table_ids)};
            """
        )

    @_commit_dec
    @print_info(msg_before='Creating table sets index...')
    def create_sets_index(self):
        self._dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._SET_INDEX_NAME}; """
        )
        self._dbcur.execute(
            f"""CREATE INDEX {self._SET_INDEX_NAME} ON {self._SET_TABLE_NAME}(id);"""
        )

    @_commit_dec
    @print_info(msg_before='Creating inverted list index...')
    def create_inverted_list_index(self):
        self._dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._INVERTED_LISTS_INDEX_NAME}; """
        )

        self._dbcur.execute(
            f"""CREATE INDEX {self._INVERTED_LISTS_INDEX_NAME} ON {self._INVERTED_LISTS_TABLE_NAME}(token);"""
        )

    @_commit_dec
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
            WHERE i.relname LIKE '{self.tprefix}%'
            """
        
        return self._dbcur.execute(q).fetchall()
