import psycopg
import psycopg.sql

from code.utils.utils import print_info


_SET_TNAME = 'SETS'
_INVERTED_LISTS_TNAME = 'inverted_lists'

_SET_IDXNAME = 'sets_id_idx'
_INVERTED_LISTS_IDXNAME = 'inverted_lists_token_idx'


@print_info(msg_before='Creating database tables...', msg_after='Completed.')
def create_tables(db:psycopg.Cursor, prefix:str=None):
    prefix = prefix + '_' if prefix else ''
    
    db.execute(
        f"""
        DROP TABLE IF EXISTS {prefix}{_INVERTED_LISTS_TNAME};
        """        
    )

    db.execute(
        f"""              
        CREATE TABLE {prefix}{_INVERTED_LISTS_TNAME} (
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
        DROP TABLE IF EXISTS {prefix}{_SET_TNAME};
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
        f"""COPY {prefix}{_INVERTED_LISTS_TNAME} FROM '{inverted_list_file}' (FORMAT CSV, DELIMITER('|'));"""
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
        f""" DROP INDEX IF EXISTS {prefix}{_INVERTED_LISTS_IDXNAME}; """
    )

    db.execute(
        f"""CREATE INDEX {prefix}{_INVERTED_LISTS_IDXNAME} ON {prefix}{_INVERTED_LISTS_TNAME} USING btree (token);"""
    )


