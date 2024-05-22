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

    def open(self):
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}")
        self.dbcur = self._dbconn.cursor(row_factory=psycopg.rows.dict_row)

    def close(self):
        self.dbcur.close()
        self._dbconn.commit()

    @print_info(msg_before='Dropping tables...', msg_after='Completed.')
    def drop_tables(self):    
        self.dbcur.execute(
            f"""
            DROP TABLE IF EXISTS {self._INVERTED_LISTS_TABLE_NAME};
            """        
        )

        self.dbcur.execute(
            f"""
            DROP TABLE IF EXISTS {self._SET_TABLE_NAME};
            """        
        )

        self.dbcur.execute(
            f"""
            DROP TABLE IF EXISTS {self._QUERY_TABLE_NAME};
            """        
        )


    @print_info(msg_before='Creating database tables...', msg_after='Completed.')
    def create_tables(self):
        
        self.dbcur.execute(
            f"""              
            CREATE TABLE {self._INVERTED_LISTS_TABLE_NAME} (
                token integer NOT NULL,
                frequency integer NOT NULL,
                duplicate_group_id integer NOT NULL,
                duplicate_group_count integer NOT NULL,
                set_ids integer[] NOT NULL,
                set_sizes integer[] NOT NULL,
                match_positions integer[] NOT NULL,
                raw_token bytea NOT NULL
            );        
            """
        )

        self.dbcur.execute(
            f"""          
                CREATE TABLE {self._SET_TABLE_NAME} (
                    id integer NOT NULL,
                    size integer NOT NULL,
                    num_non_singular_token integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )

        self.dbcur.execute(
            f"""
                CREATE TABLE {self._QUERY_TABLE_NAME} (
                    id integer NOT NULL,
                    tokens integer[] NOT NULL
                );
            """
        )


    # @print_info(msg_before='Inserting sets...', msg_after='Completed.')
    def insert_data_into_sets_table(self, sets_file:str):
        self.dbcur.execute(
            f"""COPY {self._SET_TABLE_NAME} FROM '{sets_file}' (FORMAT CSV, DELIMITER('|'));"""
        )


    # @print_info(msg_before='Inserting inverted list...', msg_after='Completed.')
    def insert_data_into_inverted_list_table(self, inverted_list_file:str):
        self.dbcur.execute(
            f"""COPY {self._INVERTED_LISTS_TABLE_NAME} FROM '{inverted_list_file}' (FORMAT CSV, DELIMITER('|'));"""
        )

    @print_info(msg_before='Inserting queries...', msg_after='Completed.')
    def insert_data_into_query_table(self, table_ids:list[int]):
        # maybe is better to translate all in postgresql...
        self.dbcur.execute(
            f"""
            INSERT INTO {self._QUERY_TABLE_NAME} SELECT id, tokens FROM {self._SET_TABLE_NAME} WHERE id in {tuple(table_ids)};
            """
        )

    # @print_info(msg_before='Creating table sets index...', msg_after='Completed.', time=True)
    def create_sets_index(self):
        self.dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._SET_INDEX_NAME}; """
        )
        self.dbcur.execute(
            f"""CREATE INDEX {self._SET_INDEX_NAME} ON {self._SET_TABLE_NAME} USING btree (id);"""
        )


    # @print_info(msg_before='Creating inverted list index...', msg_after='Completed.', time=True)
    def create_inverted_list_index(self):
        self.dbcur.execute(
            f""" DROP INDEX IF EXISTS {self._INVERTED_LISTS_INDEX_NAME}; """
        )

        self.dbcur.execute(
            f"""CREATE INDEX {self._INVERTED_LISTS_INDEX_NAME} ON {self._INVERTED_LISTS_TABLE_NAME} USING btree (token);"""
        )


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
        
        return self.dbcur.execute(q).fetchall()
