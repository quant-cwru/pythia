import yfinance as yf
import pandas as pd
from pythia.datasets.TimeSeries import TimeSeries 

class StockHistorical(TimeSeries):
    def __init__(self, ticker, start_date=None, end_date=None, interval='1d'):
        self.data : pd.DataFrame = yf.download(tickers=ticker, interval=interval, start=start_date, end=end_date)
        self.norm()

    def adjust(self):
        # Adjust Open, High, Low prices
        self.data['Open'] = self.data['Open'] * self.data['Adj Close'] / self.data['Close']
        self.data['High'] = self.data['High'] * self.data['Adj Close'] / self.data['Close']
        self.data['Low'] = self.data['Low'] * self.data['Adj Close'] / self.data['Close']

        # Save unadjusted Close price
        self.data['Unadj Close'] = self.data['Close']

        # Default to using Adj Close as Close
        self.data['Close'] = self.data['Adj Close']
        self.data.drop(columns=['Adj Close'], inplace=True)
    
    def norm(self):
        self.adjust()
        self.data_n = self.data.copy()
        for c in self.data_n.columns:
            mean = self.data_n[c].mean()
            std = self.data_n[c].std()
            self.data_n[c] = (self.data_n[c] - mean) / std

