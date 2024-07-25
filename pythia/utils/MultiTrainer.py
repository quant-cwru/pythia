from pythia.datasets.timeseries.StockHistorical import StockHistorical
from pythia.utils.DataProcessor import DataProcessor
from pythia.utils.ModelEditor import ModelEditor
from pythia.utils.Trainer import Trainer

class MultiStockTrainer:
    """
    This class manages the process of training multiple stock prediction models, one for each specified ticker symbol.

    Attributes:
        tickers (list): List of stock ticker symbols to train models for.
        model_architecture (function): Function that returns a PyTorch model given the number of features.
        hyperparams (dict): Dictionary of hyperparameters for model training.
        start_date (str): Start date for historical stock data (optional).
        end_date (str): End date for historical stock data (optional).
        trainers (dict): Dictionary to store DataProcessor, ModelEditor, and Trainer objects for each ticker.
    """

    def __init__(self, tickers, model_architecture, hyperparams, start_date=None, end_date=None):
        """
        Initialize the MultiStockTrainer with the specified tickers, model architecture, and hyperparameters.

        Args:
            tickers (list): List of stock ticker symbols.
            model_architecture (function): Function that returns a PyTorch model given the number of features.
            hyperparams (dict): Dictionary of hyperparameters for model training.
            start_date (str, optional): Start date for historical stock data. Defaults to None.
            end_date (str, optional): End date for historical stock data. Defaults to None.
        """
        self.tickers = tickers
        self.model_architecture = model_architecture
        self.hyperparams = hyperparams
        self.start_date = start_date
        self.end_date = end_date
        self.trainers = {}

    def prepare_data(self):
        """
        Prepare the data for each ticker symbol and store it in the trainers dictionary.
        """
        for ticker in self.tickers:
            # Create StockHistorical object for each ticker
            stock_data = StockHistorical(ticker, self.start_date, self.end_date)
            
            # Create and configure DataProcessor for each ticker
            data_processor = DataProcessor(stock_data)
            data_processor.set_features(['Open', 'High', 'Low', 'Volume'])
            data_processor.set_labels(['Close'], shift=1)
            
            # Store the DataProcessor in the trainers dictionary
            self.trainers[ticker] = {'data_processor': data_processor}

    def create_models(self):
        """
        Create a ModelEditor object for each ticker, add a model using the specified model architecture, and set the hyperparameters. 
        Store the ModelEditors in the trainers dictionary.
        """
        for ticker, components in self.trainers.items():
            # Get the number of features from the DataProcessor
            num_features = components['data_processor'].get_num_features()
            
            # Create ModelEditor and add model for each ticker
            model_editor = ModelEditor()
            model_name = f"{ticker}_model"
            model_editor.add_model(model_name, self.model_architecture(num_features))
            model_editor.set_hyperparams(model_name, self.hyperparams)
            
            # Store the ModelEditor in the trainers dictionary
            self.trainers[ticker]['model_editor'] = model_editor

    def train_models(self):
        """
        Create a Trainer object for each ticker, set up the training process, and execute training.
        """
        for ticker, components in self.trainers.items():
            # Create and configure Trainer for each ticker
            trainer = Trainer(f"{ticker}_model", components['data_processor'], components['model_editor'])
            trainer.setup_training()
            
            # Execute the training process
            trainer.train()
            
            # Store the Trainer in the trainers dictionary
            self.trainers[ticker]['trainer'] = trainer

    def run(self):
        """
        Execute the full process of data preparation, model creation, and model training.
        """
        self.prepare_data()
        self.create_models()
        self.train_models()

    def get_trainer(self, ticker):
        """
        Retrieve the Trainer object for a specific ticker.

        Args:
            ticker (str): The ticker symbol to get the trainer for.

        Returns:
            Trainer: The Trainer object for the specified ticker, or None if not found.
        """
        return self.trainers.get(ticker, {}).get('trainer')