from abc import ABC, abstractmethod

from dltf.utils.datalake import DataLakeHandler


class AbstractGlobalSearchAlgorithm(ABC):
    def __init__(self, mode, blacklist, dlh:DataLakeHandler, string_translators, string_patterns) -> None:
        self.mode = mode
        self.blacklist = blacklist
        self.dlh = dlh
        self.string_translators = string_translators
        self.string_patterns = string_patterns

    @abstractmethod
    def data_preparation(self) -> None:
        pass
    
    @abstractmethod
    def query(self, results_file, k, query_ids, *args) -> None:
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

