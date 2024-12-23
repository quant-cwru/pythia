import React, { useState, useEffect } from 'react'
import Settings from '../../images/Union.png'
import { Helmet } from 'react-helmet'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'
import './data.css'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const Data = (props) => {
  const[ModelActive, setModelActive] = useState(0);
  const[DataActive, setDataActive] = useState(0);
  const[ResultsActive,setResultsActive] = useState(0);

  const intervals = ['1d', '1wk', '1mo', '3mo'];
  const pricePoints = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume'];
  const datasets = ['Yahoo Finance', 'Alpha Vantage', 'IEX Cloud'];
  const modelTypes = ['LSTM', 'SimpleNN'];
  const imputationMethods = ['Forward Fill', 'Backward Fill', 'Linear Interpolation', 'Mean'];

  const [formData, setFormData] = useState({
    ticker: '',
    startDate: '',
    endDate: '',
    interval: '1d',
    pricePoint: 'Close',
    dataset: 'yahoo-finance',
    modelType: 'lstm',
    features: [],
    target: 'Close',
    hyperparameters: {
      hiddenSize: 64,
      numLayers: 2,
      learningRate: 0.001,
      batchSize: 32,
      validation_split: 0.2
    }
  });
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chartData, setChartData] = useState(null);

  useEffect(() => {
    setModelActive(1);
    setDataActive(0);
    setResultsActive(0);
  }, []);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleHyperparamChange = (param, value) => {
    setFormData(prev => ({
      ...prev,
      hyperparameters: {
        ...prev.hyperparameters,
        [param]: value
      }
    }));
  };

  const formatDate = (dateString) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    return date.toISOString().split('T')[0]; 
  };

  const trainModel = async () => {
    setLoading(true);
    setError(null);
    
    if (!formData.ticker || !formData.startDate || !formData.endDate) {
      setError('Please fill in all required fields');
      setLoading(false);
      return;
    }

    try {
      const features = [formData.pricePoint];  
      
      console.log('Sending training request with data:', {
        model_type: formData.modelType,
        ticker: formData.ticker.toUpperCase(),
        start_date: formatDate(formData.startDate),
        end_date: formatDate(formData.endDate),
        features: features,
        target: formData.pricePoint, 
        hyperparameters: {
          hidden_size: parseInt(formData.hyperparameters.hiddenSize),
          num_layers: parseInt(formData.hyperparameters.numLayers),
          learning_rate: parseFloat(formData.hyperparameters.learningRate),
          batch_size: parseInt(formData.hyperparameters.batchSize),
          validation_split: parseFloat(formData.hyperparameters.validation_split)
        }
      });

      const response = await fetch('http://localhost:8000/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_type: formData.modelType,
          ticker: formData.ticker.toUpperCase(),
          start_date: formatDate(formData.startDate),
          end_date: formatDate(formData.endDate),
          features: features,
          target: formData.pricePoint, 
          hyperparameters: {
            hidden_size: parseInt(formData.hyperparameters.hiddenSize),
            num_layers: parseInt(formData.hyperparameters.numLayers),
            learning_rate: parseFloat(formData.hyperparameters.learningRate),
            batch_size: parseInt(formData.hyperparameters.batchSize),
            validation_split: parseFloat(formData.hyperparameters.validation_split)
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Training failed');
      }

      const data = await response.json();
      console.log('Model trained:', data);
      
      getPredictions();
    } catch (err) {
      console.error('Full error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runBacktest = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log('Starting backtest with params:', {
        ticker: formData.ticker.toUpperCase(),
        model_type: formData.modelType,
        start_date: formatDate(formData.startDate),
        end_date: formatDate(formData.endDate),
        threshold: 0.02,
        transaction_cost: 0.001,
        validation_split: 0.2 
      });

      const response = await fetch('http://localhost:8000/backtest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: formData.ticker.toUpperCase(),
          model_type: formData.modelType,
          start_date: formatDate(formData.startDate),
          end_date: formatDate(formData.endDate),
          threshold: 0.02,
          transaction_cost: 0.001,
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Backtest failed');
      }

      const data = await response.json();
      console.log('Backtest results:', data);
      setResults(data);

      setChartData({
        labels: data.dates,
        datasets: [
          {
            label: 'Model Returns',
            data: data.cumulative_returns_model,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
          },
          {
            label: 'Buy & Hold Returns',
            data: data.cumulative_returns_buy_hold,
            borderColor: 'rgb(255, 99, 132)',
            tension: 0.1
          }
        ]
      });

    } catch (err) {
      console.error('Backtest error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDateChange = (field, value) => {
    if (value && !isNaN(new Date(value))) {
      setFormData(prev => ({
        ...prev,
        [field]: value
      }));
    }
  };

  const getPredictions = async () => {
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: formData.ticker.toUpperCase(),
          model_type: formData.modelType
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message);
    }
  };

  const handleModelTypeChange = (value) => {
    setFormData(prev => ({
      ...prev,
      modelType: value.toLowerCase()
    }));
  };

  function Model(){
    setModelActive(1);
    setDataActive(0);
    setResultsActive(0);
    setError(null);
  };

  function Results(){
    setResultsActive(1);
    setDataActive(0);
    setModelActive(0);
  }

  function LiveTrading(){
    console.log("B");
  }

  function BigData(){
    setDataActive(1);
    setResultsActive(0);
    setModelActive(0);
  }

  return (
    <div>
      <div className="data-menu2">
        <div className="data-menulist">
          <span className="data-text12 menu-items action-button">
            <span onClick={BigData}>Data</span>
          </span>
          <span className="data-text12 menu-items action-button ">
            <span onClick={Model}>Model</span>
          </span>
          <span className="data-text12 menu-items action-button">
            <span onClick={Results}>Results</span>
          </span>
          <span className="data-text12 menu-items action-button">
            <span>Live Trading</span>
          </span>
        </div>
      </div>

      <div className="data-union1">
        <div className="data-union2">
          <img src={Settings} className="data-settings" alt="settings" />
        </div>
      </div>

      <div className="LogoContent">
        <span className="data-text18 M3bodylarge">
          <span>AI Finance Logo Name</span>
        </span>
      </div>

      <div className="Column_2" style={{zIndex:DataActive}}>
        <div className="data-input-field10">
          <span className="data-text20 BodyBase">
            <span>Ticker</span>
          </span>
          <input
            type="text"
            placeholder="e.g. AAPL"
            className="data-input10"
            value={formData.ticker}
            onChange={(e) => handleInputChange('ticker', e.target.value)}
          />
        </div>

        <div className="data-input-field11">
          <span className="data-text22 BodyBase">
            <span className="label-size">Start Date</span>
          </span>
          <input
            type="date"
            className="data-input10"
            value={formData.startDate}
            onChange={(e) => handleDateChange('startDate', e.target.value)}
            required
          />
        </div>

        <div className="data-input-field12">
          <span className="data-text24 BodyBase">
            <span>End Date</span>
          </span>
          <input
            type="date"
            className="data-input10"
            value={formData.endDate}
            onChange={(e) => handleDateChange('endDate', e.target.value)}
            required
          />
        </div>

        <div className="data-input-field13">
          <span className="data-text26 BodyBase">
            <span>Interval</span>
          </span>
          <select className="data-input10">
            {intervals.map(interval => (
              <option key={interval} value={interval}>{interval}</option>
            ))}
          </select>
        </div>

        <div className="data-input-field14">
          <span className="data-text28 BodyBase">
            <span>Price point</span>
          </span>
          <select 
            className="data-input10"
            value={formData.pricePoint}
            onChange={(e) => handleInputChange('pricePoint', e.target.value)}
          >
            {pricePoints.map(point => (
              <option key={point} value={point}>{point}</option>
            ))}
          </select>
        </div>

        <div className="data-input-field15">
          <span className="data-text30 BodyBase">
            <span>Dataset</span>
          </span>
          <select className="data-input10">
            {datasets.map(dataset => (
              <option key={dataset} value={dataset.toLowerCase().replace(' ', '-')}>
                {dataset}
              </option>
            ))}
          </select>
        </div>

        {/* Update error and loading displays */}
        {error && (
          <div style={{
            color: 'red',
            padding: '10px',
            position: 'absolute',
            bottom: '20px',
            left: '35px',
            backgroundColor: 'rgba(255,255,255,0.9)',
            borderRadius: '4px'
          }}>
            {error}
          </div>
        )}
        {loading && (
          <div style={{
            position: 'absolute',
            bottom: '20px',
            left: '35px',
            backgroundColor: 'rgba(255,255,255,0.9)',
            padding: '10px',
            borderRadius: '4px'
          }}>
            Processing...
          </div>
        )}
      </div>

        <span className="data-text32 M3bodylarge" style={{zIndex:DataActive}}>Data choices</span>

      <div className="Column_3" style={{zIndex:(ModelActive*200)}}>
        <div className="data-input-field16">
          <span className="data-text34 BodyBase">
            <span>Imputation method</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field17">
          <span className="data-text36 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field18">
          <span className="data-text38 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field19">
          <span className="data-text40 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field20">
          <span className="data-text42 BodyBase">
            <span>Label</span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
        <div className="data-input-field21">
          <span className="data-text44">
            <span>
              Scale
              <span
                
              />
            </span>
          </span>
          <input type="text" placeholder="Value" className="data-input10" />
        </div>
          <span className="data-text46 M3bodylarge">Data Scaling and Imputation</span>
      </div>

      <div className="Column_4">

      </div>

      <div className="Model_Choice" style={{zIndex:ModelActive}}>
          <span className="Model-Header">Model Choices</span>
        <div className="data-input-field10">
          <span className="data-text20 BodyBase">
            <span>Model Type</span>
          </span>
          <select 
            className="data-input10"
            value={formData.modelType}
            onChange={(e) => handleModelTypeChange(e.target.value)}
          >
            {modelTypes.map(type => (
              <option key={type} value={type.toLowerCase()}>{type}</option>
            ))}
          </select>
        </div>
        
        <div className="data-input-field11">
          <span className="data-text22 BodyBase">
            <span className="label-size">Hidden Size</span>
          </span>
          <input 
            type="number" 
            placeholder="e.g. 64" 
            min="1" 
            className="data-input10"
            value={formData.hyperparameters.hiddenSize}
            onChange={(e) => handleHyperparamChange('hiddenSize', e.target.value)}
          />
        </div>

        <div className="data-input-field12">
          <span className="data-text24 BodyBase">
            <span>Number of Layers</span>
          </span>
          <input 
            type="number" 
            placeholder="e.g. 2" 
            min="1" 
            className="data-input10"
            value={formData.hyperparameters.numLayers}
            onChange={(e) => handleHyperparamChange('numLayers', e.target.value)}
          />
        </div>

        <div className="data-input-field13">
          <span className="data-text26 BodyBase">
            <span>Learning Rate</span>
          </span>
          <input 
            type="number" 
            placeholder="e.g. 0.001" 
            step="0.001" 
            min="0" 
            className="data-input10"
            value={formData.hyperparameters.learningRate}
            onChange={(e) => handleHyperparamChange('learningRate', e.target.value)}
          />
        </div>

        <div className="data-input-field14">
          <span className="data-text28 BodyBase">
            <span>Batch Size</span>
          </span>
          <input 
            type="number" 
            placeholder="e.g. 32" 
            min="1" 
            className="data-input10"
            value={formData.hyperparameters.batchSize}
            onChange={(e) => handleHyperparamChange('batchSize', e.target.value)}
          />
        </div>

        <div className="data-input-field15">
          <span className="data-text30 BodyBase">
            <span>Validation Split</span>
          </span>
          <input 
            type="number" 
            placeholder="e.g. 0.2" 
            step="0.1" 
            min="0"
            max="0.5"
            className="data-input10"
            value={formData.hyperparameters.validation_split}
            onChange={(e) => handleHyperparamChange('validation_split', e.target.value)}
          />
        </div>

        <div className="generate-button" style={{top: '650px'}} onClick={trainModel}>
          <span className="generate-text">
            Train Model
          </span>
        </div>
      </div>

      <div className="graph-settings" style={{zIndex:ResultsActive*1000}}>
        {chartData ? (
          <Line
            data={{
              ...chartData,
              datasets: chartData.datasets.map(dataset => ({
                ...dataset,
                backgroundColor: 'white'
              }))
            }}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                title: {
                  display: true,
                  text: `${formData.ticker} - Model vs Buy & Hold Returns`
                },
                legend: {
                  position: 'top'
                }
              },
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Cumulative Returns'
                  },
                  grid: {
                    color: '#E5E5E5'
                  },
                  ticks: {
                    callback: value => `${(value * 100).toFixed(0)}%`
                  }
                },
                x: {
                  title: {
                    display: true,
                    text: 'Date'
                  },
                  grid: {
                    color: '#E5E5E5'
                  }
                }
              },
              layout: {
                padding: 20
              },
              backgroundColor: 'white',
              elements: {
                line: {
                  borderWidth: 2
                }
              }
            }}
          />
        ) : (
          <div style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100%',
            fontSize: '1.2em',
            color: '#666'
          }}>
            Run backtest to view performance comparison
          </div>
        )}
      </div>

      <div className="Column_2" style={{zIndex:ResultsActive}}>
        <div className="results-page">
          <span className="ticker-1">
            {formData.ticker || 'Ticker 1'}
          </span>
          <div className="stats-container">
            {results && (
              <>
                <div className="stat-item">
                  <div className="stat-label">Sharpe Ratio</div>
                  <div className="stat-value">{results.sharpe_ratio_model?.toFixed(2)}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Max Drawdown</div>
                  <div className="stat-value">{(results.max_drawdown_model * 100).toFixed(2)}%</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Total Trades</div>
                  <div className="stat-value">{results.total_trades}</div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Current Action</div>
                  <div className="stat-value" style={{
                    color: results.action === 'BUY' ? '#4CAF50' : 
                           results.action === 'SELL' ? '#f44336' : '#666'
                  }}>
                    {results.action}
                  </div>
                </div>
                <div className="stat-item">
                  <div className="stat-label">Predicted Move</div>
                  <div className="stat-value" style={{
                    color: results.predicted_percent_change >= 0 ? '#4CAF50' : '#f44336'
                  }}>
                    {(results.predicted_percent_change * 100).toFixed(2)}%
                  </div>
                </div>
              </>
            )}
            <div className="action-buttons">
              <div className="stat-item action-button" onClick={runBacktest}>
                <span className="stat-value">Run Backtest</span>
              </div>
              <div className="stat-item action-button" onClick={getPredictions}>
                <span className="stat-value">Get Prediction</span>
              </div>
            </div>
          </div>
        </div>

          <span  className="Results-Header" style={{zIndex:ResultsActive}}>Summary</span>
      </div>
    </div>
  )
}

export default Data
