import yfinance as yf
import pandas as pd
from package.data.datasets.Dataset import Dataset

class StockHistorical(Dataset):
    def __init__(self, tickers, start_date=None, end_date=None, interval='1d'):
        self.data = yf.download(tickers=tickers, interval=interval, start=start_date, end=end_date)

    def __call__(self):
        return self.data
    
    def to_torch(self):
        pass

