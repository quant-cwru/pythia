import yfinance as yf
import pandas as pd
import numpy as np
from package.data.datasets.Dataset import Dataset

class StockHistorical(Dataset):
    def __init__(self, tickers, start_date=None, end_date=None, interval='1d'):
        self.data : pd.DataFrame = yf.download(tickers=tickers, interval=interval, start=start_date, end=end_date)
    
    def norm(self):
        self.data_n = self.data.copy()
        for c in self.data_n.columns:
            mean = self.data_n[c].mean()
            std = self.data_n[c].std()
            self.data_n[c] = (self.data_n[c] - mean) / std



