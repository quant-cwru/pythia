from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def to_torch(self):
        pass

