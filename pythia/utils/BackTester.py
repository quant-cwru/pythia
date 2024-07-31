import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any

class Backtester:
    """
    This class backtests a trained model, including performance evaluation
    and comparison against a buy-and-hold strategy.

    Attributes:
        trainer (Trainer): The trained model's Trainer object.
        threshold (float): The confidence threshold for making trades.
        transaction_cost (float): The fixed transaction cost per trade.
        model (nn.Module): The trained PyTorch model.
        data_processor (DataProcessor): The data processing object.
        device (str): The device to run the model on ('cuda' or 'cpu').
        is_lstm (bool): Whether the model is an LSTM.
        sequence_length (int): The input sequence length for LSTM models.
        backtest_features (pd.DataFrame): Features for backtesting.
        backtest_labels (pd.DataFrame): Labels for backtesting.
    """

    def __init__(self, trainer, threshold: float = 0.0, transaction_cost: float = 0.0):
        """
        Initialize with a trained model and trading parameters.

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
        self.is_lstm = trainer.is_lstm
        self.sequence_length = trainer.hyperparams.get('sequence_length', 10)
        
        self.split_data()
    
    def split_data(self):
        """
        Split the data so we don't backtest on the training data.
        """
        split_index = int(len(self.data_processor.features) * (1 - self.trainer.hyperparams['validation_split']))
        self.backtest_features = self.data_processor.features.iloc[split_index:]
        self.backtest_labels = self.data_processor.labels.iloc[split_index:]
            
    def run_backtest(self) -> Dict[str, Any]:
        """
        Backtest the trained model.

        Returns:
            Dict[str, Any]: A dictionary containing the backtest results.

        Raises:
            ValueError: If there's no valid data for backtesting after processing.
        """
        self.model.eval()
        
        actual_returns = self.backtest_labels.pct_change().dropna()
        model_returns, trade_count, positions, predictions, prediction_changes = self.calculate_model_returns(actual_returns)
        buy_hold_returns = actual_returns.squeeze()
        
        if len(model_returns) == 0 or len(buy_hold_returns) == 0:
            raise ValueError("No valid data for backtesting after processing. Check your data and alignment.")
        
        common_index = model_returns.index.intersection(buy_hold_returns.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between model returns and buy-hold returns.")
        
        model_returns = model_returns.loc[common_index]
        buy_hold_returns = buy_hold_returns.loc[common_index]
        positions = positions.loc[common_index]
        predictions = predictions.loc[common_index]
        prediction_changes = prediction_changes.loc[common_index]
        
        cum_model_returns = (1 + model_returns).cumprod()
        cum_buy_hold_returns = (1 + buy_hold_returns).cumprod()
                
        return {
            'sharpe_ratio_model': self.calculate_sharpe_ratio(model_returns),
            'sharpe_ratio_buy_hold': self.calculate_sharpe_ratio(buy_hold_returns),
            'max_drawdown_model': self.calculate_max_drawdown(cum_model_returns),
            'max_drawdown_buy_hold': self.calculate_max_drawdown(cum_buy_hold_returns),
            'cum_model_returns': cum_model_returns,
            'cum_buy_hold_returns': cum_buy_hold_returns,
            'total_trades': trade_count,
            'positions': positions,
            'predictions': predictions,
            'prediction_changes': prediction_changes,
            'actual_returns': actual_returns
        }
    
    def _get_predictions(self) -> pd.Series:
        """
        Generate predictions using the trained model.

        Returns:
            pd.Series: A series of predictions.
        """
        predictions = []
        self.model.eval()
        with torch.no_grad():
            if self.is_lstm:
                for i in range(len(self.backtest_features) - self.sequence_length + 1):
                    sequence = self.backtest_features.iloc[i:i+self.sequence_length].values
                    sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
                    prediction = self.model(sequence).squeeze().item()
                    predictions.append(prediction)
            else:
                for features in self.backtest_features.values:
                    features = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Add batch dimension
                    prediction = self.model(features).squeeze().item()
                    predictions.append(prediction)
        
        return pd.Series(predictions, index=self.backtest_features.index[-len(predictions):])
    
    def calculate_model_returns(self, actual_returns: pd.Series) -> tuple:
        """
        Calculate returns based on the model's predictions and threshold.

        Args:
            actual_returns (pd.Series): The actual returns of the asset.

        Returns:
            tuple: (pd.Series model_returns, int trade_count, pd.Series positions, pd.Series predictions, pd.Series prediction_changes)
        """
        predictions = self._get_predictions()
        
        common_index = predictions.index.intersection(actual_returns.index)
        
        predictions = predictions.loc[common_index]
        actual_returns = actual_returns.loc[common_index].squeeze()
        
        # Calculate the percentage change of predictions
        prediction_changes = predictions.pct_change()

        # If above threshold, go long (1), if below negative threshold, go short (-1), otherwise hold cash (0)
        positions = pd.Series(np.where(prediction_changes.abs() > self.threshold, np.sign(prediction_changes), 0), index=common_index)
    
        # Calculate trades (position changes)
        trades = positions.diff().abs()
        trade_count = trades.sum()
        
        # Calculate transaction costs
        transaction_costs = trades * self.transaction_cost
        
        # Calculate returns
        model_returns = positions.shift(1) * actual_returns - transaction_costs
        model_returns = model_returns.dropna()
        
        return model_returns, trade_count, positions, predictions, prediction_changes
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate the Sharpe ratio of a series of returns.

        Args:
            returns (pd.Series): A series of returns.

        Returns:
            float: The Sharpe ratio.
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        if len(returns) == 0:
            return None
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    def calculate_max_drawdown(self, cum_returns: pd.Series) -> float:
        """
        Calculate the maximum drawdown of a series of cumulative returns.

        Args:
            cum_returns (pd.Series): A series of cumulative returns.

        Returns:
            float: The maximum drawdown.
        """
        if isinstance(cum_returns, pd.DataFrame):
            cum_returns = cum_returns.squeeze()
        if len(cum_returns) == 0:
            return None
        return (cum_returns / cum_returns.cummax() - 1).min()
    
    def plot_results(self, results: Dict[str, Any]):
        """
        Plot the cumulative returns of the model against the buy-and-hold strategy.

        Args:
            results (Dict[str, Any]): The results dictionary from run_backtest.
        """
        if results['cum_model_returns'] is None or results['cum_buy_hold_returns'] is None:
            print("Not enough data to plot results.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot cumulative returns
        ax1.plot(results['cum_buy_hold_returns'], label='Buy and Hold', color='blue', linewidth=2)
        ax1.plot(results['cum_model_returns'], label='Model Strategy', color='green', linewidth=2)
        ax1.set_title('Cumulative Returns: Model vs Buy and Hold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Returns')
        ax1.legend()
        ax1.grid(True)
    
        # Plot prediction changes
        ax2.plot(results['prediction_changes'].index, results['prediction_changes'], label='Prediction Changes', color='purple', linewidth=1)
        ax2.set_title('Prediction Changes and Positions')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Prediction Change')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.show()