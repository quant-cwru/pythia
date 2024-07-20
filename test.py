from pythia.datasets.timeseries.StockHistorical import StockHistorical
from pythia.utils.DataProcessor import DataProcessor
#import matplotlib
#matplotlib.use('module://matplotlib-backend-kitty')

dataset = StockHistorical("AAPL")
dataset.norm()

dp = DataProcessor(dataset)
dp.plot().show()
