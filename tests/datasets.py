import yfinance as yf
from pythia.datasets.timeseries.CurrencyHistorical import CurrencyHistorical
from pythia.datasets.timeseries.CommoditiesHistorical import CommoditiesHistorical

#ch = CurrencyHistorical("USDEUR", "1980-01-01", "2024-02-25")
#print(ch) 

# oil = yf.Ticker("CL=F")
oil = CommoditiesHistorical("CL=F")
print(oil)
