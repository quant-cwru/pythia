import pandas as pd
import matplotlib.pyplot as plt
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
        if hasattr(self.data, "data_n"):
            self.features = self.data.data_n[features]
        else:
            self.features = self.data.data[features]
    
    def set_labels(self, labels: list, shift=1):
        """
        Takes in a list of labels and creates the respective target subset.
        Since we are dealing with time-series data right now, the shift indicates how many timesteps ahead the labels are.
        """
        if hasattr(self.data, "data_n"):
            data = self.data.data_n
        else:
            data = self.data.data

        self.labels = data[labels].shift(-shift)

        # Drop the last shift labels (nans)
        self.labels = self.labels[:-shift]

    def drop(self, col: str):
        """
        Drops a column in the dataset. Should also drop the same column in its features and labels (if present).
        """
        pass

    def plot(self, sample_idx = None):
        """
        Plots the input dataframe if no sample index is provided. Else plots the associated feature-target on a timeseries scale.
        """
        
        if hasattr(self.data, "data_n"):
            # Data is normalized, denormalize it
            data = self.data.data_n
            means = self.data.means
            stds = self.data.stds

            for c in data.columns:
                data[c] = data[c] * stds[c] + means[c]
        else:
            # Data is unnormalized
            data = self.data.data

        if sample_idx is None:
            # Plot the entire DataFrame
            data.plot(figsize=(10, 6))
            plt.title("Unnormalized Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.grid(True)
            plt.show()
        else:
            if sample_idx < 0 or sample_idx >= len(self.data):
                raise IndexError("Sample index out of range.")

            # Plot the feature-target pair for the specified sample index
            sample_data = data.iloc[sample_idx]
            sample_data.plot(figsize=(10, 6))
            plt.title(f"Unnormalized Data for Sample Index {sample_idx}")
            plt.xlabel("Feature")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()

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
        
