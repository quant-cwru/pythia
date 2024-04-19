from openbb import obb
from torch.utils.data import Dataset
import numpy as np
import torch

class CurrencyHistoricalDataset(Dataset):

  def __init__(self, pair, start_date, end_date, seq_length, interval='1d', transform=None, provider='yfinance'):
    """
    Intializes a time-series dataset of exchange rates between a pair of currencies.
    :param pair: concatenated string of currencies to use (ex: 'EURUSD')
    :param start_date: the start date of the dataset
    :param end_date: the end date of the dataset
    :param seq_length: the sequence length to use for time series prediction
    :param interval: the interval between each data point (exchange rate)
    :param transform: see torch datasets documentation. Applied a transform to the data
    :param provider: the provider to fetch data from. Keep this yfinance for now
    """

    # pull data from openbb
    currency_df = obb.currency.price.historical(symbol=pair, start_date=start_date, end_date=end_date, interval=interval, provider=provider).to_df()

    # we are concerned with the open values of exchange rates
    open_values = currency_df.open.values
    self.X, self.y = [], []
    self.transform = transform

    # create input/label samples
    for i in range(len(open_values)-seq_length):
      self.X.append(open_values[i:i+seq_length])
      self.y.append(open_values[i+1:i+seq_length+1])

    # record means and standard deviations for unscaling
    self.X_mean = np.mean(self.X)
    self.X_std = np.std(self.X)
    self.y_mean = np.mean(self.y)
    self.y_std = np.std(self.y)

  def __len__(self):
    """
    :return: the length of/number of samples in the dataset
    """
    return len(self.X)

  def __getitem__(self, idx):
    """
    :return: the sample indexed at idx of the dataset
    """
    inputs = self.X[idx]
    targets = self.y[idx]
    sample = (inputs, targets)

    if self.transform:
      sample = self.transform(sample)

    return sample


class Rescale(object):
  def __call__(self, sample):
    inputs = sample[0]
    targets = sample[1]
    inputs_mean = np.mean(inputs)
    inputs_std = np.std(inputs, axis=0)
    targets_mean = np.mean(targets)
    targets_std = np.std(targets, axis=0)

    inputs = (inputs-inputs_mean) / inputs_std
    targets = (targets-targets_mean) / targets_std

    return inputs, targets

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
      inputs = sample[0]
      targets = sample[1]
      return torch.tensor(inputs), torch.tensor(targets)
