"""
Dataset class
"""
from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement __getitem__.")
