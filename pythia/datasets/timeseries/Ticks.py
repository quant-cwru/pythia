import yfinance as yf
import pandas as pd
from pythia.datasets.TimeSeries import TimeSeries 

class Ticks(TimeSeries):
    def __init__(self, ticker, start_date=None, end_date=None, interval='1d'):
        self.data : pd.DataFrame = yf.download(tickers=ticker, interval=interval, start=start_date, end=end_date)
    
    def norm(self):
        self.data_n = self.data.copy()
        for c in self.data_n.columns:
            mean = self.data_n[c].mean()
            std = self.data_n[c].std()
            self.data_n[c] = (self.data_n[c] - mean) / std

