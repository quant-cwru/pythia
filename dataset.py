import warnings

class Dataset():
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement __getitem__.")