import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from pythia.datasets.timeseries.StockHistorical import StockHistorical
from pythia.utils.DataProcessor import DataProcessor
from pythia.utils.Trainer import Trainer
from pythia.utils.ModelEditor import ModelEditor
from pythia.utils.BackTester import Backtester
from pythia.utils.LiveTrader import LiveTrader
from pythia.models.lstm import LSTMModel
from pythia.models.simple import SimpleNN

app = FastAPI(title="Pythia API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingConfig(BaseModel):
    model_type: str = Field(..., description="Model type: 'lstm' or 'simple'")
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: str = Field(..., description="Training start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Training end date (YYYY-MM-DD)")
    features: List[str] = Field(..., description="Features to use for training")
    target: str = Field(..., description="Target variable to predict")
    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    model_type: str = Field(..., description="Model type for prediction")

class BacktestConfig(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    model_type: str = Field(..., description="Model type to backtest")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Backtest end date (YYYY-MM-DD)")
    threshold: float = Field(0.0, description="Trading signal threshold")
    transaction_cost: float = Field(0.0, description="Transaction cost per trade")

# Store trained models in memory
trained_models = {}

@app.post("/train")
async def train_model(config: TrainingConfig):
    """Train a new stock prediction model"""
    try:
        # Initialize stock dataset
        dataset = StockHistorical(config.ticker, config.start_date, config.end_date)

        # Process data
        data_processor = DataProcessor(dataset)
        data_processor.set_features(config.features)
        data_processor.set_labels([config.target], shift=1)

        # Create model
        model_editor = ModelEditor()
        input_size = len(config.features)
        
        if config.model_type == "lstm":
            model = LSTMModel(
                input_size=input_size,
                hidden_size=config.hyperparameters.get("hidden_size", 64),
                num_layers=config.hyperparameters.get("num_layers", 2),
                output_size=1
            )
        elif config.model_type == "simple":
            model = SimpleNN(
                input_size=input_size,
                hidden_size=config.hyperparameters.get("hidden_size", 64),
                output_size=1
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")

        model_name = f"{config.ticker}_{config.model_type}"
        model_editor.add_model(model_name, model)
        model_editor.set_hyperparams(model_name, config.hyperparameters)

        # Train model
        trainer = Trainer(model_name, data_processor, model_editor)
        trainer.setup_training()
        trainer.train()

        # Store trained model
        trained_models[model_name] = trainer

        return {"message": "Model trained successfully", "model_name": model_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Get stock predictions using a trained model"""
    try:
        model_name = f"{request.ticker}_{request.model_type}"
        if model_name not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found. Please train first.")

        trainer = trained_models[model_name]
        live_trader = LiveTrader(trainer, threshold=0.02, transaction_cost=0)

        action, predicted_move = live_trader.run_live_trading(request.ticker)

        return {
            "ticker": request.ticker,
            "action": action,
            "predicted_percent_change": float(predicted_move),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def backtest(config: BacktestConfig):
    """Run backtesting on a trained model"""
    try:
        model_name = f"{config.ticker}_{config.model_type}"
        if model_name not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found. Please train first.")

        trainer = trained_models[model_name]
        backtester = Backtester(
            trainer,
            threshold=config.threshold,
            transaction_cost=config.transaction_cost
        )

        results = backtester.run_backtest()

        return {
            "sharpe_ratio_model": float(results["sharpe_ratio_model"]),
            "sharpe_ratio_buy_hold": float(results["sharpe_ratio_buy_hold"]),
            "max_drawdown_model": float(results["max_drawdown_model"]),
            "max_drawdown_buy_hold": float(results["max_drawdown_buy_hold"]),
            "total_trades": int(results["total_trades"]),
            "cumulative_returns_model": results["cum_model_returns"].tolist(),
            "cumulative_returns_buy_hold": results["cum_buy_hold_returns"].tolist(),
            "positions": results["positions"].tolist(),
            "dates": results["positions"].index.strftime('%Y-%m-%d').tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all trained models"""
    return {
        "models": [
            {
                "name": name,
                "type": name.split("_")[-1],
                "ticker": "_".join(name.split("_")[:-1])
            }
            for name in trained_models.keys()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)