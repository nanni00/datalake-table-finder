import time

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.engine import URL, Engine
from sqlalchemy import (
    create_engine, MetaData, 
    text, inspect, func,
    select, insert, delete,
    Column,
    Integer, LargeBinary, ARRAY, update
)

from dltftools.utils.loghandler import info, error
from dltftools.testers.josie.josie_io import RawTokenSet, ListEntry


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
        if url and engine:
            self.url = url
            self.engine = engine
        else:
            self.url = URL.create(**connection_info)
            self.engine = create_engine(self.url)

        # SqlAlchemy==1.4
        # self.metadata = MetaData(self.engine)
        # self.metadata.reflect()

        # SqlAlchemy==2.0
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        
        # initial cost values 
        self.min_read_cost = 1000000.0
        self.read_set_cost_slope = 1253.19054300781
        self.read_set_cost_intercept = -9423326.99507381
        self.read_list_cost_slope = 1661.93366983753
        self.read_list_cost_intercept = 1007857.48225696

    def execute_in_session(self, q):
        with Session(self.engine) as session:
            results = session.execute(q)
            session.commit()
        return results

    def drop_tables(self):
        for table_class in [InvertedList, Set, Query, ReadListCostSamples, ReadSetCostSamples]:
            try:
                table_class.__table__.drop(self.engine, checkfirst=True)
            except Exception as e:
                error(f"Failed to drop table {table_class.__tablename__}: {e}")
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

    def are_costs_sampled(self):
        q = f""" 
            SELECT 
                (SELECT COUNT(*) FROM {ReadListCostSamples.__tablename__}),
                (SELECT COUNT(*) FROM {ReadSetCostSamples.__tablename__});"""
        return all(x != 0 for x in self.execute_in_session(text(q)).first())

    def close(self):
        self.engine.dispose()

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
                return session.execute(select(Set.tokens[1:end_pos]).filter(Set.id == set_id)).fetchone()[0]
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
                # print('#1 ::: ', session.execute(select(Set.tokens[start_pos:1e9]).filter(Set.id == set_id)).fetchone()[0])
                # print(select(Set.tokens[start_pos:1e9]).filter(Set.id == set_id).compile(bind=self.engine))
                return session.execute(select(Set.tokens[start_pos:1e9]).filter(Set.id == set_id)).fetchone()[0]
            except:
                # print('#2 ::: ', session.execute(select(Set.tokens).where(Set.id == set_id)).fetchone()[0])
                # print(select(Set.tokens[start_pos:]).filter(Set.id == set_id).compile(bind=self.engine))
                return [
                        row
                        for i, row in enumerate(session.execute(select(Set.tokens).where(Set.id == set_id)).fetchone()[0])
                        if i >= start_pos
                ]
        
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
    
    def get_inverted_list(self, token:int) -> list[ListEntry]:
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
        # TODO translate this into sqlalchemy code
        q = """
        SELECT id, (
			SELECT array_agg(raw_token)
			FROM inverted_lists
			WHERE token = any(tokens)
		), tokens FROM queries
        ORDER BY id
        """

        rows = self.execute_in_session(text(q))

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
        subq = (
            select(InvertedList.token, InvertedList.frequency)
            .where(min_freq <= InvertedList.frequency, InvertedList.frequency <= max_freq)
            .order_by(func.random())
            .limit(sample_size_per_step)
        )
        q = (
            insert(ReadListCostSamples)
            .from_select(['token', 'frequency'], subq)
        )
        self.execute_in_session(q)

    def count_token_from_read_list_cost(self, min_freq, max_freq):
        return self.execute_in_session(
            select(func.count(ReadListCostSamples.token))
            .where(min_freq <= ReadListCostSamples.frequency, ReadListCostSamples.frequency <= max_freq)
        ).fetchone()[0]
    
    def get_array_agg_token_read_list_cost(self):
        return self.execute_in_session(
            select(func.array_agg(ReadListCostSamples.token))
        ).fetchone()[0]
    
    def update_read_list_cost(self, token, cost):
        self.execute_in_session(
            update(ReadListCostSamples)
            .values(cost=cost)
            .where(ReadListCostSamples.token == token)
        )

    def count_number_of_sets(self) -> int:
        return int(self.execute_in_session(
            select(func.count(Set.id))
        ).fetchone()[0])

    def reset_cost_function_parameters(self, verbose: bool) -> None:
        with Session(self.engine) as session:
            q = f"""SELECT regr_slope(cost, frequency), regr_intercept(cost, frequency)
                FROM {ReadListCostSamples.__tablename__};"""
            slope, intercept = session.execute(text(q)).fetchone()

            if verbose:
                info(f"Resetting read list cost slope {self.read_list_cost_slope:.4f} -> {slope:.4f}")
                info(f"Resetting read list cost intercept {self.read_list_cost_intercept:.4f} -> {intercept:.4f}")

            self.read_list_cost_slope = slope
            self.read_list_cost_intercept = intercept

            q = f"""SELECT regr_slope(cost, size), regr_intercept(cost, size)
                    FROM {ReadSetCostSamples.__tablename__};"""
            slope, intercept = session.execute(text(q)).fetchone()

            if verbose:
                info(f"Resetting read set cost slope {self.read_set_cost_slope:.4f} -> {slope:.4f}")
                info(f"Resetting read set cost intercept {self.read_set_cost_intercept:.4f} -> {intercept:.4f}")

            self.read_set_cost_slope = slope
            self.read_set_cost_intercept = intercept

    def delete_cost_tables(self):
        self.execute_in_session(delete(ReadListCostSamples))
        self.execute_in_session(delete(ReadSetCostSamples))

    def sample_costs(self, 
                     min_length:int = 0, 
                     max_length:int = 20_000, 
                     step:int = 500, 
                     sample_size_per_step:int = 10):
        # the table should have been already created 
        # when the JOSIE has built the main indexes
        sample_set_ids = self.get_queries_agg_id()

        for set_id in sample_set_ids:
            start = time.time()
            s = self.get_set_tokens(set_id)
            duration = time.time() - start  # Duration in seconds
            self.insert_read_set_cost(set_id, len(s), int(duration * 1e9))

        for l in range(min_length, max_length, step):
            self.insert_read_list_cost(l, l+step, sample_size_per_step)

        sample_list_tokens = self.get_array_agg_token_read_list_cost()
        for token in sample_list_tokens:
            start = time.time()
            self.get_inverted_list(token)
            duration = time.time() - start
            self.update_read_list_cost(token, int(duration * 1e9))

    def read_list_cost(self, length: int) -> float:
        cost = self.read_list_cost_slope * float(length) + self.read_list_cost_intercept
        if cost < self.min_read_cost:
            cost = self.min_read_cost
        return cost / 1000000.0

    def read_set_cost(self, size: int) -> float:
        cost = self.read_set_cost_slope * float(size) + self.read_set_cost_intercept
        if cost < self.min_read_cost:
            cost = self.min_read_cost
        return cost / 1000000.0

    def read_set_cost_reduction(self, size: int, truncation: int) -> float:
        return self.read_set_cost(size) - self.read_set_cost(size - truncation)

