from sqlalchemy import Float, exc
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from sqlalchemy import (
    create_engine, 
    select, insert,
    MetaData, Column, Integer
)
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()


class Overlaps(Base):
    __tablename__ = 'overlaps'
    r_id = Column(Integer, primary_key=True, index=True)
    s_id = Column(Integer, primary_key=True, index=True)
    table_overlap = Column(Integer)
    set_q_size = Column(Integer)
    set_r_size = Column(Integer)
    set_overlap = Column(Integer)
    set_union_size = Column(Integer)
    bag_q_size = Column(Integer)
    bag_r_size = Column(Integer)
    bag_overlap = Column(Integer)
    sloth_time = Column(Float)
    set_time = Column(Float)
    bag_time = Column(Float)


class OverlapsDBHandler:
    """ Used only for testing, in order to avoid computing each time the SLOTH overlap """
    def __init__(self, connection_info:dict|None=None):
        self.url = URL.create(**connection_info)
        self.engine = create_engine(self.url)
        
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        
    def create_table(self):
        Base.metadata.create_all(self.engine)

    def add_overlaps(self, values:list[list[int,int,int,int,int]]):
        """
        # TODO these rollbacks are not really nice
        Inserts a list of computed overlap, where each entry is a list of three elements:
        (r_id, s_id, table_overlap, set_q_size, set_r_size, set_overlap, bag_q_size, bag_q_size, bag_overlap, sloth_time, set_time, bag_time), assuming r_id < s_id
        """
        values = [{
            'r_id': r_id, 's_id': s_id, 
            'table_overlap': to, 
            'set_q_size': sqs, 'set_r_size': srs, 'set_overlap': so, 'set_union_size': sus,
            'bag_q_size': bqs, 'bag_r_size': brs, 'bag_overlap': bo,
            'sloth_time': slt, 'set_time': st, 'bag_time': bt} 
            for r_id, s_id, to, sqs, srs, so, sus, bqs, brs, bo, slt, st, bt in values]
        
        with Session(self.engine) as session:
            try:
                session.execute(insert(Overlaps).values(values))
                session.commit()
            except exc.IntegrityError:
                session.rollback()
                for value in values:
                    try:
                        session.execute(insert(Overlaps).values([value]))
                        session.commit()
                    except exc.IntegrityError:
                        session.rollback()
                        continue

    def lookup(self, r_id, s_id):
        """ 
        Returns the stored overlaps and timings for the pair (r_id, s_id), assuming that r_id < s_id 
        """
        with Session(self.engine) as session:
            return session.execute(select(Overlaps.table_overlap, 
                                          Overlaps.set_q_size, Overlaps.set_r_size, Overlaps.set_overlap, Overlaps.set_union_size,
                                          Overlaps.bag_q_size, Overlaps.bag_r_size, Overlaps.bag_overlap,
                                          Overlaps.sloth_time, Overlaps.set_time, Overlaps.bag_time)
                                   .where(Overlaps.r_id == r_id, Overlaps.s_id == s_id)).fetchone()

    def drop_table(self):
        Overlaps.__table__.drop(self.engine, checkfirst=True)

    def close(self):
        self.engine.dispose()
