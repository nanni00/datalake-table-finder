"""
In this module there are utility classes for various stuff,
such as the abstract class for the testing pipeline
"""
from abc import ABC, abstractmethod
import bisect
import logging

# import orjson
import psycopg

from tools.utils.misc import print_info
from tools.utils.datalake import SimpleDataLakeHelper


class AlgorithmTester(ABC):
    def __init__(self, mode, dataset, size, num_cpu, blacklist, datalake_helper:SimpleDataLakeHelper) -> None:
        self.mode = mode
        self.dataset = dataset
        self.size = size
        self.num_cpu = num_cpu
        self.blacklist = blacklist
        self.datalake_helper = datalake_helper

    @abstractmethod
    def data_preparation(self) -> None:
        pass
    
    @abstractmethod
    def query(self, results_file, k, query_ids, *args) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass




class ResultDatabase:
    """ Used only for testing, in order to avoid computing each time the SLOTH overlap """
    def __init__(self, dbname, table_name='results_table'):
        self.dbname = dbname
        self.table_name = table_name
    
    def open(self):
        self._dbconn = psycopg.connect(f"port=5442 host=/tmp dbname={self.dbname}", row_factory=psycopg.rows.dict_row)

    def create_table(self):
        q = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{self.table_name}'
            );
        """
        exists = self._dbconn.execute(q).fetchone()['exists'] == True
        if not exists:
            self._dbconn.execute(
                f"""
                CREATE TABLE {self.table_name} (
                    r_id integer NOT NULL,
                    s_id integer NOT NULL,
                    sloth_overlap integer NOT NULL,
                    PRIMARY KEY (r_id, s_id)
                );

                CREATE INDEX {self.table_name}_r_id_index ON {self.table_name}(r_id, s_id);
                """
            )
        else:
            logging.info(f'Results table "{self.table_name}" already exists')
        self._dbconn.commit()

    def insert_results(self, values:list[list[int,int,int]]):
        """
        Inserts a list of computed overlap, where each entry is a list of three elements:
        (r_id, s_id, sloth_overlap), assuming r_id < s_id
        """
        self._dbconn \
            .cursor() \
                .executemany(f"INSERT INTO {self.table_name} VALUES(%s, %s, %s) ON CONFLICT (r_id, s_id) DO NOTHING RETURNING (r_id);", values)
        self._dbconn.commit()

    def lookup(self, r_id, s_id) -> int|None:
        """ Where the r_id < s_id """

        result = self._dbconn.execute(
            f"""
            SELECT sloth_overlap FROM {self.table_name}
            WHERE r_id = {r_id} AND s_id = {s_id}
            """
        )

        try:
            result = result.fetchone()
        except psycopg.ProgrammingError:
            raise Exception()

        return None if result == None else result['sloth_overlap']
    
    def clear(self):
        q = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{self.table_name}'
            );
        """
        exists = self._dbconn.execute(q).fetchone()['exists'] == True
        if exists:
            self._dbconn.execute(f"TRUNCATE {self.table_name} ;")
            self._dbconn.commit()

    def get_number_of_sloth_failures(self):
        return self._dbconn.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE sloth_overlap = -1").fetchone()

    def get_numer_of_records(self):
        return self._dbconn.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()

    def close(self):
        self._dbconn.close()




