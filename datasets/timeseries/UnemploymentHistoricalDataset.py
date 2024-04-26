from openbb import obb
from torch.utils.data import Dataset
import numpy as np

class UnemploymentHistoricalDataset(Dataset):
  countries = ['colombia', 'new_zealand', 'united_kingdom', 'italy', 'luxembourg', 'euro_area19', 'sweden', 'oecd', 'south_africa', 'denmark', 'canada', 'switzerland', 'slovakia', 'hungary', 'portugal', 'spain', 'france', 'czech_republic', 'costa_rica', 'japan', 'slovenia', 'russia', 'austria', 'latvia', 'netherlands', 'israel', 'iceland', 'united_states', 'ireland', 'mexico', 'germany', 'greece', 'turkey', 'australia', 'poland', 'south_korea', 'chile', 'finland', 'european_union27_2020', 'norway', 'lithuania', 'euro_area20', 'estonia', 'belgium', 'brazil', 'indonesia', 'all']
  def __init__(self, country, start_date=None, end_date=None, sex='total', age='total', seasonal_adjustment=False, frequency='monthly', seq_length=50, transform=None):
    """
    Intializes a dataset of unemployment rates for a given country.
    :param country: the country to retrieve unemployment data from. See countries for a valid list of names.
    :param start_date: start date of dataset.
    :param end_date: end date of dataset.
    :param sex: sex to get unemployment for. ['total', 'male', 'female']
    :param age: age range to get umemployment data for. ['total', '15-24', '15-64', '25-54', '55-64']
    :param seasonal_adjustment: Whether to get seasonally adjusted unemployment. Defaults to False.
    :param frequency: frequency to get unemployment rates for. ['monthly', 'quarterly', 'annual']
    :param seq_length: the sequence length to use for time series prediction.
    :param transform: see torch datasets documentation. Applies a transform to the data.
    """

    # pull data from openbb
    df = df = obb.economy.unemployment(country=country, 
                                       start_date=start_date, 
                                       end_date=end_date, 
                                       sex=sex, age=age, 
                                       seasonal_adjustment=seasonal_adjustment,
                                       frequency=frequency,
                                       provider='oecd').to_df()
    values = df.value.values
    self.X, self.y = [], []
    self.transform = transform

    for i in range(len(values)-seq_length):
      self.X.append(values[i:i+seq_length])
      self.y.append(values[i+1:i+seq_length+1])

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
