"""
Dataset class
"""
from abc import ABC, abstractmethod

import unittest

class Dataset(ABC):
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement __getitem__")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Subclasses should implement __len__")

class TestDataset(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()
