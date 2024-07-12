from pythia.data.datasets.timeseries.StockHistorical import StockHistorical
from pythia.utils.DataProcessor import DataProcessor
from pythia.utils.ModelEditor import ModelEditor
from pythia.utils.Trainer import Trainer

class MultiStockTrainer:
    def __init__(self, tickers, model_architecture, hyperparams, start_date=None, end_date=None):
        self.tickers = tickers
        self.model_architecture = model_architecture
        self.hyperparams = hyperparams
        self.start_date = start_date
        self.end_date = end_date
        self.trainers = {}

    def prepare_data(self):
        for ticker in self.tickers:
            stock_data = StockHistorical(ticker, self.start_date, self.end_date)
            data_processor = DataProcessor(stock_data)
            data_processor.set_features(['Open', 'High', 'Low', 'Volume'])
            data_processor.set_labels(['Close'], shift=1)
            self.trainers[ticker] = {'data_processor': data_processor}

    def create_models(self):
        for ticker, components in self.trainers.items():
            num_features = components['data_processor'].get_num_features()
            model_editor = ModelEditor()
            model_name = f"{ticker}_model"
            model_editor.add_model(model_name, self.model_architecture(num_features))
            model_editor.set_hyperparams(model_name, self.hyperparams)
            self.trainers[ticker]['model_editor'] = model_editor

    def train_models(self):
        for ticker, components in self.trainers.items():
            trainer = Trainer(f"{ticker}_model", components['data_processor'], components['model_editor'])
            trainer.setup_training()
            trainer.train()
            self.trainers[ticker]['trainer'] = trainer

    def run(self):
        self.prepare_data()
        self.create_models()
        self.train_models()

    def get_trainer(self, ticker):
        return self.trainers.get(ticker, {}).get('trainer')