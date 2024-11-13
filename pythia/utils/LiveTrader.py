import yfinance as yf
import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta
from pythia.datasets.timeseries.StockHistorical import StockHistorical
from pythia.utils.DataProcessor import DataProcessor

class LiveTrader:
    """
    This class manages live trading using a trained model, including data fetching, prediction, and signal generation.

    Attributes:
        trainer (Trainer): The trained model's Trainer object.
        threshold (float): The confidence threshold for making trades.
        transaction_cost (float): The fixed transaction cost per trade.
        model (nn.Module): The trained PyTorch model.
        data_processor (DataProcessor): The data processing object.
        device (str): The device to run the model on ('cuda' or 'cpu').
    """

    def __init__(self, trainer, threshold: float = 0.0, transaction_cost: float = 0.0):
        """
        Initialize the LiveTrader with a trained model and trading parameters.

        Args:
            trainer (Trainer): A trained Trainer object.
            threshold (float): The confidence threshold for making trades. Defaults to 0.0.
            transaction_cost (float): The fixed transaction cost per trade. Defaults to 0.0.
        """
        self.trainer = trainer
        self.threshold = threshold
        self.transaction_cost = transaction_cost
        self.model = trainer.model
        self.data_processor = trainer.data_processor
        self.device = trainer.device

    def get_latest_data(self, ticker: str, days: int = 30) -> StockHistorical:
        """
        Fetch the latest data for the given ticker.

        Args:
            ticker (str): The stock ticker symbol.
            days (int): Number of past days to fetch data for. Defaults to 30.

        Returns:
            StockHistorical: The fetched stock data as a StockHistorical object.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        return StockHistorical(ticker, start_date, end_date)

    def prepare_data(self, stock_data: StockHistorical) -> torch.Tensor:
        """
        Prepare the data for prediction.

        Args:
            stock_data (StockHistorical): Raw stock data.

        Returns:
            torch.Tensor: Processed data ready for model input.
        """
        processor = DataProcessor(stock_data)
        processor.set_features(self.data_processor.features.columns.tolist())
        processor.set_labels(self.data_processor.labels.columns.tolist(), shift=1)

        if self.trainer.is_lstm:
            sequence_length = self.trainer.hyperparams.get('sequence_length', 30)
            features = processor.features.iloc[-sequence_length:].values
            return torch.FloatTensor(features).unsqueeze(0).to(self.device)
        else:
            return torch.FloatTensor(processor.features.iloc[-1:].values).to(self.device)

    def predict_next_move(self, ticker: str) -> float:
        """
        Predict the next price movement for the given ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            float: The predicted price movement.
        """
        stock_data = self.get_latest_data(ticker)
        latest_data = self.prepare_data(stock_data)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(latest_data)

        # Denormalize the prediction using the StockHistorical class method
        close_mean = stock_data.data['Close'].mean()
        close_std = stock_data.data['Close'].std()
        denormalized_prediction = prediction * close_std + close_mean
        
        current_price = stock_data.data['Close'].iloc[-1]
        predicted_move = denormalized_prediction - current_price

        return predicted_move

    def get_trading_signal(self, ticker: str) -> int:
        """
        Get the trading signal based on the prediction and threshold.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            int: The trading signal (1 for Buy, -1 for Sell, 0 for Hold).
        """
        predicted_move = self.predict_next_move(ticker)
        current_price = self.get_latest_data(ticker).data['Close'].iloc[-1]
        
        # Use percentage change for threshold comparison
        percentage_change = predicted_move / current_price
        
        if percentage_change > self.threshold:
            return 1  # Buy 
        elif percentage_change < -self.threshold:
            return -1  # Sell 
        else:
            return 0  # Hold

    def run_live_trading(self, ticker: str) -> tuple:
        """
        Run live trading for the given ticker.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            tuple: A tuple containing the recommended action (str) and predicted move (float).
        """
        signal = self.get_trading_signal(ticker)
        predicted_move = self.predict_next_move(ticker)
        current_price = self.get_latest_data(ticker).data['Close'].iloc[-1]
        predicted_change = predicted_move / current_price * 100

        
        if signal == 1:
            action = "BUY"
        elif signal == -1:
            action = "SELL"
        else:
            action = "HOLD"

        print(f"Live Trading Signal for {ticker}:")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Action: {action}")
        print(f"Predicted Move: ${predicted_move:.2f}")
        print(f"Predicted Price: ${(current_price + predicted_move):.2f}")
        print(f"Predicted Change: {predicted_change:.2f}%")

        return action, predicted_change