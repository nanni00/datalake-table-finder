from abc import ABC, abstractmethod

class AlgorithmTester(ABC):
    def __init__(self, mode, blacklist, datalake_helper, token_translators) -> None:
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

