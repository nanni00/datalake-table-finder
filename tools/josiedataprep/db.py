import psycopg
import psycopg.sql

from tools.utils.utils import print_info


_SET_TNAME = 'SETS'
_INVERTED_LISTS_TABLE_NAME = 'inverted_lists'

_SET_IDXNAME = 'sets_id_idx'
_INVERTED_LISTS_INDEX_NAME = 'inverted_lists_token_idx'


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
def drop_tables(db:psycopg.Cursor, prefix:str=None):
    prefix = prefix + '_' if prefix else ''
    
    db.execute(
        f"""
        DROP TABLE IF EXISTS {prefix}{_INVERTED_LISTS_TABLE_NAME};
        """        
    )

    db.execute(
        f"""
        DROP TABLE IF EXISTS {prefix}{_SET_TNAME};
        """        
    )


@print_info(msg_before='Creating database tables...', msg_after='Completed.')
def create_tables(db:psycopg.Cursor, prefix:str=None):
    prefix = prefix + '_' if prefix else ''
    
    db.execute(
        f"""              
        CREATE TABLE {prefix}{_INVERTED_LISTS_TABLE_NAME} (
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
            CREATE TABLE {prefix}{_SET_TNAME} (
                id integer NOT NULL,
                size integer NOT NULL,
                num_non_singular_token integer NOT NULL,
                tokens integer[] NOT NULL
            );
        """
    )


# @print_info(msg_before='Inserting sets...', msg_after='Completed.', time=True)
def insert_data_into_sets_table(db:psycopg.Cursor, sets_file:str, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f"""COPY {prefix}{_SET_TNAME} FROM '{sets_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


# @print_info(msg_before='Inserting inverted list...', msg_after='Completed.', time=True)
def insert_data_into_inverted_list_table(db:psycopg.Cursor, inverted_list_file:str, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f"""COPY {prefix}{_INVERTED_LISTS_TABLE_NAME} FROM '{inverted_list_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


@print_info(msg_before='Creating table sets index...', msg_after='Completed.', time=True)
def create_sets_index(db:psycopg.Cursor, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f""" DROP INDEX IF EXISTS {prefix}{_SET_IDXNAME}; """
    )
    db.execute(
        f"""CREATE INDEX {prefix}{_SET_IDXNAME} ON {prefix}{_SET_TNAME} USING btree (id);"""
    )


@print_info(msg_before='Creating inverted list index...', msg_after='Completed.', time=True)
def create_inverted_list_index(db:psycopg.Cursor, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f""" DROP INDEX IF EXISTS {prefix}{_INVERTED_LISTS_INDEX_NAME}; """
    )

    db.execute(
        f"""CREATE INDEX {prefix}{_INVERTED_LISTS_INDEX_NAME} ON {prefix}{_INVERTED_LISTS_TABLE_NAME} USING btree (token);"""
    )


def get_statistics_from_(db:psycopg.Cursor, prefix:str=None):
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
        WHERE i.relname LIKE '{prefix}%'
        """
    
    return db.execute(q).fetchall()
