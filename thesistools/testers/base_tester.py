from abc import ABC, abstractmethod

from thesistools.utils.datalake import DataLakeHandler


class AlgorithmTester(ABC):
    def __init__(self, mode, blacklist, datalake_helper:DataLakeHandler, token_translators) -> None:
        self.mode = mode
        self.blacklist = blacklist
        self.dlh = datalake_helper
        self.token_translators = token_translators

    @abstractmethod
    def data_preparation(self) -> None:
        pass
    
    @abstractmethod
    def query(self, results_file, k, query_ids, *args) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

