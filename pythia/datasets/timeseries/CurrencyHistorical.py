from pythia.datasets.TimeSeries import TimeSeries 
from openbb import obb
from datetime import datetime

class CurrencyHistorical(TimeSeries):
    def __init__(self, pair, start_date, end_date=None, interval='1d', transform=None, provider='yfinance'):
        """
        Intializes a time-series dataset of exchange rates between a pair of currencies.
        :param pair: concatenated string of currencies to use (ex: 'EURUSD').
        :param start_date: the start date of the dataset.
        :param end_date: the end date of the dataset.
        :param seq_length: the sequence length to use for time series prediction.
        :param interval: the interval between each data point (exchange rate).
        :param transform: see torch datasets documentation. Applies a transform to the data.
        :param provider: the provider to fetch data from. Keep this yfinance for now.
        """
        if end_date is None:
            end_date = datetime.today().strftime("%Y-%m-%d")
        # pull data from openbb
        self.data = obb.currency.price.historical(symbol=pair, start_date=start_date, end_date=end_date, interval=interval, provider=provider).to_df()


    def norm(self):
        self.data_n = self.data.copy()
        for c in self.data_n.columns:
            mean = self.data_n[c].mean()
            std = self.data_n[c].std()
            self.data_n[c] = (self.data_n[c] - mean) / std



