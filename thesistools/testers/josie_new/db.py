import logging
from sqlalchemy.orm import Session
from sqlalchemy.engine import URL, Engine
from sqlalchemy import (
    create_engine, MetaData, 
    text, inspect, func,
    select, insert,
    Column,
    Integer, LargeBinary, ARRAY, update
)
from sqlalchemy.ext.declarative import declarative_base


from josie_io import RawTokenSet, ListEntry


Base = declarative_base()


# Table Classes
# TODO create indexes on InvertedList and Set
class InvertedList(Base):
    __tablename__ =         'inverted_lists'
    token =                 Column(Integer, primary_key=True)
    frequency =             Column(Integer)
    duplicate_group_id =    Column(Integer)
    duplicate_group_count = Column(Integer)
    raw_token =             Column(LargeBinary)
    set_ids =               Column(ARRAY(Integer))
    set_sizes =             Column(ARRAY(Integer))
    match_positions =       Column(ARRAY(Integer))


class Set(Base):
    __tablename__ =         'sets'
    id =                    Column(Integer, primary_key=True)
    size =                  Column(Integer)
    num_non_singular_token = Column(Integer)
    tokens =                Column(ARRAY(Integer))


class Query(Base):
    __tablename__ =         'queries'
    id =                    Column(Integer, primary_key=True)
    tokens =                Column(ARRAY(Integer))


class ReadListCostSamples(Base):
    __tablename__ =         'read_list_cost_samples'
    token =                 Column(Integer, primary_key=True)
    frequency =             Column(Integer)
    cost =                  Column(Integer)


class ReadSetCostSamples(Base):
    __tablename__ =         'read_set_cost_samples'
    id =                    Column(Integer, primary_key=True)
    size =                  Column(Integer)
    cost =                  Column(Integer)


class JOSIEDBHandler:
    def __init__(self, url: URL | None = None, engine: Engine | None = None, **connection_info) -> None:
        # self.tables_prefix = tables_prefix

        # self._SET_TABLE_NAME = f'{self.tables_prefix}_sets'
        # self._LISTS_TABLE_NAME = f'{self.tables_prefix}_inverted_lists'
        # self._SET_INDEX_NAME = f'{self.tables_prefix}_sets_id_idx'
        # self._INVERTED_LISTS_INDEX_NAME = f'{self.tables_prefix}_inverted_lists_token_idx'
        # self._QUERY_TABLE_NAME = f'{self.tables_prefix}_queries'
        # self._READ_LIST_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_list_cost_samples'
        # self._READ_SET_COST_SAMPLES_TABLE_NAME = f'{self.tables_prefix}_read_set_cost_samples'

        if url and engine:
            self.url = url
            self.engine = engine
        else:
            self.url = URL.create(**connection_info)
            self.engine = create_engine(self.url)

        self.metadata = MetaData(self.engine)
        self.metadata.reflect()

    def execute_in_session(self, q):
        with Session(self.engine) as session:
            return session.execute(q)

    def drop_tables(self):
        for table_class in [InvertedList, Set, Query, ReadListCostSamples, ReadSetCostSamples]:
            try:
                table_class.__table__.drop(self.engine, checkfirst=True)
            except Exception as e:
                logging.error(f"Failed to drop table {table_class.__tablename__}: {e}")
                continue

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def clear_query_table(self):
        with Session(self.engine) as session:
            session.query(Query).delete()
            session.commit()

    def add_queries_from_existent_tables(self, table_ids: list[int] = None):
        with Session(self.engine) as session:
            set_table = Set
            query_table = Query
            session.execute(
                insert(query_table)
                .from_select(
                    ['id', 'tokens'],
                    select(set_table.id, set_table.tokens).where(set_table.id.in_(table_ids))
                )
            )
            session.commit()

    def add_queries(self, table_ids: int = None, tokens_ids: list[int] = None):
        values = [{'id': table_id, 'tokens': tokens} for table_id, tokens in zip(table_ids, tokens_ids)]
        with Session(self.engine) as session:
            session.execute(insert(Query).values(values))
            session.commit()

    def exist_cost_tables(self):
        return inspect(self.engine).has_table(self._READ_LIST_COST_SAMPLES_TABLE_NAME) or inspect(self.engine).has_table(self._READ_SET_COST_SAMPLES_TABLE_NAME)

    def get_statistics(self):
        q = f"""
            SELECT 
                i.relname AS "table_name",
                indexrelname AS "index_name",
                pg_size_pretty(pg_total_relation_size(relid)) AS "total_size",
                pg_size_pretty(pg_relation_size(relid)) AS "table_size",
                pg_size_pretty(pg_relation_size(indexrelid)) AS "index_size",
                reltuples::bigint AS "estimated_table_row_count"
            FROM pg_stat_all_indexes i 
            JOIN pg_class c ON i.relid = c.oid 
        """
        with Session(self.engine) as session:
            return list(session.execute(text(q)))

    def cost_tables_exist(self):
        q = f"""
            SELECT EXISTS (
               SELECT FROM information_schema.tables 
               WHERE table_name = '{ReadListCostSamples.__tablename__}'
               OR table_name = '{ReadSetCostSamples.__tablename__}'
            );
        """
        return self.execute_in_session(text(q)).first()[0]

    def close(self):
        self.engine.dispose()

    def reset_cost_function_parameters(self, verbose: bool) -> None:
        global read_list_cost_slope, read_list_cost_intercept
        global read_set_cost_slope, read_set_cost_intercept

        with Session(self.engine) as session:
            q = f"""SELECT regr_slope(cost, frequency), regr_intercept(cost, frequency)
                FROM {ReadListCostSamples.__tablename__};"""
            slope, intercept = session.execute(text(q)).fetchone()

            if verbose:
                logging.info(f"Resetting read list cost slope {read_list_cost_slope:.4f} -> {slope:.4f}")
                logging.info(f"Resetting read list cost intercept {read_list_cost_intercept:.4f} -> {intercept:.4f}")

            read_list_cost_slope = slope
            read_list_cost_intercept = intercept

            q = f"""SELECT regr_slope(cost, size), regr_intercept(cost, size)
                    FROM {ReadSetCostSamples.__tablename__};"""
            slope, intercept = session.execute(text(q)).fetchone()

            if verbose:
                logging.info(f"Resetting read set cost slope {read_set_cost_slope:.4f} -> {slope:.4f}")
                logging.info(f"Resetting read set cost intercept {read_set_cost_intercept:.4f} -> {intercept:.4f}")

            read_set_cost_slope = slope
            read_set_cost_intercept = intercept

    def count_posting_lists(self) -> int:
        return self.execute_in_session(select(func.count()).select_from(InvertedList)).fetchone()[0]

    def max_duplicate_group_id(self):
        return self.execute_in_session(select(func.max(InvertedList.duplicate_group_id))).fetchone()[0]
    
    def posting_lists__memproc(self):
        return self.execute_in_session(
            select(InvertedList.raw_token, InvertedList.token,
                   InvertedList.frequency, InvertedList.duplicate_group_id)).fetchall()

    def posting_lists__diskproc(self, ignore_self: bool):
        if ignore_self:
            q = f"""
                SELECT token, frequency - 1 AS count, duplicate_group_id
                FROM {InvertedList.__tablename__}
                WHERE token = ANY(%s) AND frequency > 1
                ORDER BY token ASC;
            """
        else:
            q = f"""
                SELECT token, frequency - 1 AS count, duplicate_group_id
                FROM {InvertedList.__tablename__}
                WHERE token = ANY(%s)
                ORDER BY token ASC;
            """
        return self.execute_in_session(text(q)).fetchall()

    def get_set_tokens(self, set_id):
        return self.execute_in_session(select(Set.tokens).filter(Set.id == set_id)).fetchone()[0]
    
    # TODO check is this works correctly
    def get_set_tokens_by_prefix(self, set_id, end_pos):
        with Session(self.engine) as session:
            try:
                return session.execute(select(Set.tokens[:end_pos]).filter(Set.id == set_id)).fetchone()[0]
            except:
                return (
                    row
                    for i, row in enumerate(session.execute(select(Set.tokens).filter(Set.id == set_id)).all())
                    if i <= end_pos
                )
    
    # TODO check is this works correctly
    def get_set_tokens_by_suffix(self, set_id, start_pos):
        with Session(self.engine) as session:
            try:
                return session.execute(select(Set.tokens[start_pos:]).filter(Set.id == set_id)).fetchone()[0]
            except:
                return (
                        row
                        for i, row in enumerate(session.execute(select(Set.tokens).filter(Set.id == set_id)).all())
                        if i >= start_pos
                    )
        
    # TODO check is this works correctly
    def get_set_tokens_subset(self, set_id, start_pos, end_pos):
        with Session(self.engine) as session:
            try:
                return session.execute(select(Set.tokens[start_pos:end_pos]).filter(Set.id == set_id)).fetchone()[0]
            except:
                return (
                        row
                        for i, row in enumerate(session.execute(select(Set.tokens).filter(Set.id == set_id)).all())
                        if start_pos <= i <= end_pos
                    )
    
    def get_inverted_list(self, token:int):
        set_ids, sizes, match_positions = self.execute_in_session(
            select(InvertedList.set_ids, InvertedList.set_sizes, InvertedList.match_positions)
            .filter(InvertedList.token == token)).fetchone()
    
        entries = []
        for i in range(len(set_ids)):
            entry = ListEntry(
                set_id=set_ids[i],
                size=int(sizes[i]),
                match_position=int(match_positions[i])
            )
            entries.append(entry)
        return entries

    
    def get_query_sets(self):
        rows = self.execute_in_session(
            select(Query.id, 
                   select(func.array_agg(InvertedList.raw_token)).filter(Query.tokens.any_(InvertedList.token)),
                   Query.tokens
                   )
                )
        queries = []

        for row in rows:
            set_id, raw_tokens, tokens = row
            query = RawTokenSet(set_id, tokens, raw_tokens)
            queries.append(query)
        
        return queries
    
    def get_queries_agg_id(self):
        return self.execute_in_session(
            select(func.array_agg(Query.id))
        ).fetchone()[0]

    def insert_read_set_cost(self, set_id, size, cost):
        self.execute_in_session(
            insert(ReadSetCostSamples)
            .values(id=set_id, size=size, cost=cost)
        )

    def insert_read_list_cost(self, min_freq, max_freq, sample_size_per_step):
        self.execute_in_session(
            insert(ReadListCostSamples)
            .values(
                select(InvertedList.token, InvertedList.frequency)
                .where(min_freq <= InvertedList.frequency <= max_freq)
                .order_by(func.random())
                .limit(sample_size_per_step)
            )
        )

    def count_token_from_read_list_cost(self, min_freq, max_freq):
        return self.execute_in_session(
            select(func.count(ReadListCostSamples.token))
            .where(min_freq <= ReadListCostSamples.frequency <= max_freq)
        ).fetchone()[0]
    
    def get_array_agg_token_read_list_cost(self):
        return self.execute_in_session(
            select(func.array_agg(ReadListCostSamples.token))
        ).fetchone()[0]
    
    def update_read_list_cost(self, token, cost):
        self.execute_in_session(
            update(ReadListCostSamples)
            .values(cost=cost)
            .where(token=token)
        )

    def count_number_of_sets(self):
        self.execute_in_session(
            select(func.count(Set.id))
        )
