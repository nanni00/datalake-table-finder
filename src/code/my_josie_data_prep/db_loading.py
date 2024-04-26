import psycopg
import psycopg.sql

from code.utils.utils import print_info


@print_info(msg_before='Creating database tables...', msg_after='Completed.')
def create_tables(db:psycopg.Cursor, prefix:str=None):
    prefix = prefix + '_' if prefix else ''
    
    db.execute(
        f"""
        DROP TABLE IF EXISTS {prefix}inverted_list;
        """        
    )

    db.execute(
        f"""              
        CREATE TABLE {prefix}inverted_list (
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
        DROP TABLE IF EXISTS {prefix}sets;
        """        
    )

    db.execute(
        f"""          
            CREATE TABLE {prefix}sets (
                id integer NOT NULL,
                size integer NOT NULL,
                num_non_singular_token integer NOT NULL,
                tokens integer[] NOT NULL
            );
        """
    )


@print_info(msg_before='Inserting sets...', msg_after='Completed.', time=True)
def insert_data_into_sets_table(db:psycopg.Cursor, sets_file:str, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f"""COPY {prefix}sets FROM '{sets_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


@print_info(msg_before='Inserting inverted list...', msg_after='Completed.', time=True)
def insert_data_into_inverted_list_table(db:psycopg.Cursor, inverted_list_file:str, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f"""COPY {prefix}inverted_list FROM '{inverted_list_file}' (FORMAT CSV, DELIMITER('|'));"""
    )


@print_info(msg_before='Creating table sets index...', msg_after='Completed.', time=True)
def create_sets_index(db:psycopg.Cursor, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f""" DROP INDEX IF EXISTS {prefix}sets_id_idx; """
    )
    db.execute(
        f"""CREATE INDEX {prefix}sets_id_idx ON {prefix}sets USING btree (id);"""
    )


@print_info(msg_before='Creating inverted list index...', msg_after='Completed.', time=True)
def create_inverted_list_index(db:psycopg.Cursor, prefix:str=None):
    prefix = f'{prefix}_' if prefix else ''
    db.execute(
        f""" DROP INDEX IF EXISTS {prefix}inverted_lists_token_idx; """
    )

    db.execute(
        f"""CREATE INDEX {prefix}inverted_lists_token_idx ON {prefix}inverted_list USING btree (token);"""
    )


