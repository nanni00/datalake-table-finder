from abc import ABC, abstractmethod
from typing import Dict, List

from dltf.utils.datalake import DataLakeHandler


class AbstractGlobalSearchAlgorithm(ABC):
    def __init__(self, mode, dlh:DataLakeHandler, string_blacklist, string_translators, string_patterns) -> None:
        self.mode               = mode
        self.string_blacklist   = string_blacklist
        self.dlh                = dlh
        self.string_translators = string_translators
        self.string_patterns    = string_patterns

    @abstractmethod
    def data_preparation(self) -> None:
        pass
    
    @abstractmethod
    def query(self, queries:List[int]|Dict[int,List], k:int, results_file:str=None, **kwargs):
        """
        :param queries: the queries. If the type is list, it's expected to be a
                        list of integers of tables already loaded in the datalake, 
                        otherwise it's a dictionary of pairs <ID, table (row-view)> 
                        where for each table all of its columns are valid.
        :param k: the number of results to return
        """
        pass

    @abstractmethod
    def clean(self) -> None:
        pass

