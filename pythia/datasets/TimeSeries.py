from abc import ABC, abstractmethod

class TimeSeries(ABC):

    @abstractmethod
    def norm(self): pass

    def __str__(self): return self.data.__str__()
