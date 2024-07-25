import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
from pythia.datasets.TimeSeries import TimeSeries

class DataProcessor:
    """
    This class handles feature selection, label creation, and data transformation for both normalized and raw time series data. 
    It provides methods for data manipulation, visualization, and conversion to PyTorch datasets.

    Attributes:
        dataset (TimeSeries): The input time series data.
        features (pd.DataFrame): Selected features for model input.
        labels (pd.DataFrame): Target variables for prediction.
    """
 
    def __init__(self, data : TimeSeries):
        """
        Initialize the DataProcessor with a TimeSeries dataset.

        Args:
            data (TimeSeries): Input dataset
        """
        self.dataset = data
        self.features = None
        self.labels = None
    
    def set_features(self, features : list):
        """
        Takes in a list of columns identified by their name and creates the respective feature subset.

        Args:
            features (list): List of column names to be used as features.
        """
        # Use normalized data if available, otherwise use raw data
        data = self.dataset.data_n if hasattr(self.dataset, "data_n") else self.dataset.data
        self.features = data[features]
    
    def set_labels(self, labels: list, shift=1):
        """
        Set the label columns and apply a time shift for prediction.

        Args:
            labels (list): List of column names to be used as labels.
            shift (int): Number of time steps to shift the labels. Defaults to 1.
        """
        # Use normalized data if available, otherwise use raw data
        data = self.dataset.data_n if hasattr(self.dataset, "data_n") else self.dataset.data

        # Shift the labels so that they are the target values, i.e. we use today's ohlc [features] to predict tomorrow's close [label]
        self.labels = data[labels].shift(-shift)
        
        # Remove NaN values and align the features with the labels
        self.features = self.features[:-shift]
        self.labels = self.labels[:-shift].dropna()
        self.features = self.features.loc[self.labels.index]

    def drop(self, col: str):
        """
        Drops a column in the dataset. Should also drop the same column in its features and labels (if present).

        Args:
            col (str): Name of the column to be dropped.
        """
        if hasattr(self.dataset, "data_n"):
            # If the data has been normalized drop the column in the normalized data
            self.dataset.data_n.drop(columns=[col], inplace=True)

        # Drop the column from raw data
        self.dataset.data.drop(columns=[col], inplace=True)

        # Drop from features and labels if present
        if self.features is not None and col in self.features.columns:
            self.features.drop(columns=[col], inplace=True)
        if self.labels is not None and col in self.labels.columns:
            self.labels.drop(columns=[col], inplace=True)

    def plot(self, feature_idx=None):
        """
        Plots the input dataframe if no sample index is provided. Else plots the associated feature-target on a timeseries scale.

        Args:
            feature_idx (int, optional): Index of the feature to plot against the target.
                                         If None, plots all data. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The created plot figure.

        Raises:
            IndexError: If the provided feature_idx is out of range.
        """
        # Denormalize data if necessary
        if hasattr(self.dataset, "data_n"):
            data = self.dataset.data_n.copy()
            means, stds = self.dataset.data.mean(), self.dataset.data.std()
            data = data * stds + means
        else:
            data = self.dataset.data

        fig, ax = plt.subplots(figsize=(10, 6))

        if feature_idx is None:
            # Plot the entire DataFrame
            data.plot(ax=ax)
            ax.set_title("Stock Prices")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
        else:
            # Plot the feature-target pair
            if feature_idx < 0 or feature_idx >= len(self.features.columns):
                raise IndexError("Feature index out of range.")
            
            feature_name = self.features.columns[feature_idx]

            ax.plot(self.features.index, self.features.iloc[:, feature_idx], 
                    label=feature_name, color='blue')

            target_name = self.labels.columns[0] 
            ax.plot(self.labels.index, self.labels.iloc[:, 0], 
                    label=f'Target ({target_name})', color='red', linestyle='--')

            ax.set_title(f"Feature-Target Timeseries: {feature_name} vs {target_name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.legend()

        ax.grid(True)
        return fig

    def to_torch(self):
        """
        Returns a torch dataset (in tensors) from the corresponding features and labels.
        
        Returns:
            CustomTorchDataset: A PyTorch Dataset containing the features and labels.
        """
        # Custom torch class
        class CustomTorchDataset(Dataset):
            def __init__(self, features, labels):
                self.features = torch.tensor(features.values, dtype=torch.float32)
                self.labels = torch.tensor(labels.values, dtype=torch.float32)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]

        return CustomTorchDataset(self.features, self.labels)

    def __getitem__(self, idx):
        """
        Get a sample from the processed data.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing features and labels for the given index.

        Raises:
            ValueError: If features and labels haven't been set.
        """
        if self.features is None or self.labels is None:
            raise ValueError("Features and labels must be set before indexing.")
        return {
            'features': self.features.iloc[idx],
            'labels': self.labels.iloc[idx]
        }
    
    def get_num_features(self):
        """
        Get the number of features in the processed data.

        Returns:
            int: The number of feature columns.
        """
        return self.features.shape[1] if self.features is not None else 0
    
    def __len__(self):
        """
        Get the number of samples in the processed data.

        Returns:
            int: The number of samples.

        Raises:
            ValueError: If features haven't been set.
        """
        if self.features is None:
            raise ValueError("Features must be set before getting length.")
        return len(self.features)
    
    def __str__(self) -> str:
        """
        Provide a string representation of the DataProcessor.

        Returns:
            str: A description of the DataProcessor including dataset dimensions.
        """
        return f"DataProcessor containing {len(self.dataset.data.columns)}x{len(self.dataset.data)} samples"
