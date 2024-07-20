from abc import ABC, abstractmethod

class TimeSeries(ABC):

    @abstractmethod
    def norm(self): pass

