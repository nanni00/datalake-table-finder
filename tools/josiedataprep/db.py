import psycopg
import psycopg.sql

from tools.utils.utils import print_info


_SET_TABLE_NAME = 'sets'
_INVERTED_LISTS_TABLE_NAME = 'inverted_lists'

_SET_INDEX_NAME = 'sets_id_idx'
_INVERTED_LISTS_INDEX_NAME = 'inverted_lists_token_idx'

_QUERY_TABLE_NAME = 'queries'



@print_info(msg_before='Dropping database...', msg_after='Completed.')
def drop_database(db:psycopg.Cursor, dbname:str):
    db.execute(f""" 
               SELECT 'DROP DATABASE {dbname}' 
               WHERE EXISTS (SELECT FROM pg_database WHERE datname = '{dbname}'); """)

@print_info(msg_before='Creating database...', msg_after='Completed.')
def create_db(db:psycopg.Cursor, dbname:str):
    db.execute(f""" 
               SELECT 'CREATE DATABASE {dbname}' 
               WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{dbname}'); """)


@print_info(msg_before='Dropping tables...', msg_after='Completed.')
def drop_tables(db:psycopg.Cursor):    
    db.execute(
        f"""
        DROP TABLE IF EXISTS {_INVERTED_LISTS_TABLE_NAME};
        """        
    )

    db.execute(
        f"""
        DROP TABLE IF EXISTS {_SET_TABLE_NAME};
        """        
    )

    db.execute(
        f"""
        DROP TABLE IF EXISTS {_QUERY_TABLE_NAME};
        """        
    )


@print_info(msg_before='Creating database tables...', msg_after='Completed.')
def create_tables(db:psycopg.Cursor):
    
    db.execute(
        f"""              
        CREATE TABLE {_INVERTED_LISTS_TABLE_NAME} (
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

    db.execute(
        f"""          
            CREATE TABLE {_SET_TABLE_NAME} (
                id integer NOT NULL,
                size integer NOT NULL,
                num_non_singular_token integer NOT NULL,
                tokens integer[] NOT NULL
            );
        """
    )

    db.execute(
        f"""
            CREATE TABLE {_QUERY_TABLE_NAME} (
                id integer NOT NULL,
                tokens integer[] NOT NULL
            );
        """
    )


# @print_info(msg_before='Inserting sets...', msg_after='Completed.', time=True)
def insert_data_into_sets_table(db:psycopg.Cursor, sets_file:str):
    db.execute(
        f"""COPY {_SET_TABLE_NAME} FROM '{sets_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


# @print_info(msg_before='Inserting inverted list...', msg_after='Completed.', time=True)
def insert_data_into_inverted_list_table(db:psycopg.Cursor, inverted_list_file:str):
    db.execute(
        f"""COPY {_INVERTED_LISTS_TABLE_NAME} FROM '{inverted_list_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


def insert_data_into_query_table(db:psycopg.Cursor, table_ids:list[int]):
    # maybe is better to translate all in postgresql...
    db.execute(
        f"""
        INSERT INTO {_QUERY_TABLE_NAME} SELECT id, tokens FROM {_SET_TABLE_NAME} WHERE id in {tuple(table_ids)};
        """
    )




@print_info(msg_before='Creating table sets index...', msg_after='Completed.', time=True)
def create_sets_index(db:psycopg.Cursor):
    db.execute(
        f""" DROP INDEX IF EXISTS {_SET_INDEX_NAME}; """
    )
    db.execute(
        f"""CREATE INDEX {_SET_INDEX_NAME} ON {_SET_TABLE_NAME} USING btree (id);"""
    )


@print_info(msg_before='Creating inverted list index...', msg_after='Completed.', time=True)
def create_inverted_list_index(db:psycopg.Cursor):
    db.execute(
        f""" DROP INDEX IF EXISTS {_INVERTED_LISTS_INDEX_NAME}; """
    )

    db.execute(
        f"""CREATE INDEX {_INVERTED_LISTS_INDEX_NAME} ON {_INVERTED_LISTS_TABLE_NAME} USING btree (token);"""
    )


def get_statistics_from_(db:psycopg.Cursor, dbname:str):
    q = f"""
        SELECT 
            i.relname 
            "table_name",
            indexrelname "index_name",
            pg_size_pretty(pg_total_relation_size(relid)) as "total_size",
            pg_size_pretty(pg_indexes_size(relid)) as "total_size_all_indexes",
            pg_size_pretty(pg_relation_size(relid)) as "table_size",
            pg_size_pretty(pg_relation_size(indexrelid)) "index_size",
            reltuples::bigint "estimated_table_row_count"
        FROM pg_stat_all_indexes i JOIN pg_class c ON i.relid = c.oid 
        WHERE i.relname LIKE '{dbname}%'
        """
    
    return db.execute(q).fetchall()
