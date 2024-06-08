import pandas as pd
from torch.utils.data import Dataset
import torch
from platform.data.datasets.timeseries.StockHistorical import StockHistorical


class DataProcessor:
    """
    Important distinction between a dataset and a processor: only the latter contains features and labels.
    For now assume the input object is formatted exactly like data.datasets.timeseries.StockHistorical
    If data does not have the attribute data_n, then the data is unnormed.
    """
    def __init__(self, data : StockHistorical):
        self.data = data
    
    def set_features(self, features : list):
        """
        Takes in a list of columns identified by their name and creates the respective feature subset.
        """
        pass
    
    def set_labels(self, labels: list, shift=1):
        """
        Takes in a list of labels and creates the respective target subset.
        Since we are dealing with time-series data right now, the shift indicates how many timesteps ahead the labels are.
        """
        pass

    def drop(self, col: str):
        """
        Drops a column in the dataset. Should also drop the same column in its features and labels (if present).
        """
        pass

    def plot(self, sample_idx = None):
        """
        Plots the input dataframe if no sample index is provided. Else plots the associated feature-target on a timeseries scale.
        """
        pass

    def to_torch(self):
        """
        Returns a torch dataset (in tensors) from the corresponding features and labels.
        See https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        """

        #Custom torch class
        class CustomTorchDataset(Dataset):
            def __init__(self, dataframe):
                self.data = dataframe

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return torch.tensor(self.data.iloc[idx].values, dtype=torch.float)

        #Instantiates the torch class with the data and returns it
        torchDataset = CustomTorchDataset(self.data.data)
        return torchDataset

    def __getitem__(self, idx):
        return self.data[idx]
        
