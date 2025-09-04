# 1-Hour Trading Model

This is a specialized AI trading model designed for 1-hour cryptocurrency trading with the following key features:

## 🎯 Model Specifications

- **Primary Timeframe**: 1-hour bars
- **Trend Analysis**: Uses 4-hour and daily trends for signal confirmation
- **Signal Types**: Only BUY and NO-TRADE signals (no sell signals)
- **Trade Duration**: Exactly 1 hour (buy at open, sell at close)
- **Minimum Gain**: 3% minimum gain requirement for trade validation
- **Risk Management**: Conservative approach with trend confirmation

## 🏗️ Architecture

### Multi-Timeframe Analysis
- **1h Bars**: Primary trading timeframe with detailed technical indicators
- **4h Bars**: Medium-term trend analysis and momentum confirmation
- **1d Bars**: Long-term trend direction and market context

### Technical Indicators
- **Price Indicators**: SMA (5,10,20), EMA (5,10,20), Bollinger Bands
- **Momentum**: RSI, MACD, Price momentum (1,3,5 periods)
- **Volume**: Volume ratio, Volume SMA
- **Volatility**: Rolling standard deviation, High-Low spread
- **Trend Confirmation**: 4h and 1d trend alignment

### Neural Network
- **Architecture**: 4-layer MLP with dropout regularization
- **Input**: 30+ engineered features from multi-timeframe analysis
- **Output**: Binary classification (Buy vs No-Trade)
- **Optimization**: Adam optimizer with learning rate scheduling

## 📊 Training Process

### Data Preparation
1. Load 1h, 4h, and 1d historical data
2. Align timeframes and forward-fill trend data
3. Engineer technical indicators for each timeframe
4. Create labels based on 3% minimum gain requirement

### Feature Engineering
- **1h Features**: Primary technical indicators and price action
- **4h Features**: Trend direction, momentum, and volatility
- **1d Features**: Long-term trend and market context
- **Time Features**: Hour of day, day of week patterns

### Model Training
- **Validation**: 20% holdout for testing
- **Regularization**: Dropout and early stopping
- **Optimization**: Adam with learning rate reduction
- **Monitoring**: Loss and accuracy tracking

## 🚀 Usage

### Quick Start
```bash
# Train the 1-hour trading model
python train_1h_model.py

# Or use the main script with custom config
python main.py --config config_1h_trading.yaml --train
```

### Configuration
Edit `config_1h_trading.yaml` to customize:
- Symbol and timeframes
- Model parameters (epochs, learning rate, etc.)
- Backtest settings (threshold, fees, etc.)
- Risk management parameters

### Output Files
After training, you'll find:
- `artifacts/trading_model.pth` - Trained PyTorch model
- `artifacts/scaler.pkl` - Feature scaler
- `artifacts/backtest_results.csv` - Backtest performance
- `artifacts/training_history.png` - Training progress plot

## 📈 Backtesting

The model includes comprehensive backtesting with:
- **Signal Generation**: Based on probability threshold
- **Trade Execution**: Buy at bar open, sell at bar close
- **Performance Metrics**: Win rate, total return, average return
- **Risk Analysis**: Trade distribution and market condition analysis

## ⚙️ Configuration Options

### Model Parameters
```yaml
train:
  epochs: 50              # Training epochs
  batch_size: 256         # Batch size
  lr: 0.0005             # Learning rate
  hidden_size: 256        # Network size
  dropout: 0.3           # Dropout rate
```

### Trading Parameters
```yaml
backtest:
  threshold: 0.7         # Buy signal probability threshold
  min_gain_pct: 3.0      # Minimum 3% gain requirement
  fee_bps: 10           # Trading fees (0.1%)
```

## 🔍 Model Validation

The model is validated using:
- **Classification Metrics**: Precision, recall, F1-score
- **Confusion Matrix**: Buy vs No-Trade performance
- **Backtesting**: Historical performance simulation
- **Market Condition Analysis**: Performance across different market states

## 📋 Requirements

All dependencies are listed in `requirements.txt`:
- PyTorch for neural network
- scikit-learn for preprocessing
- pandas/numpy for data handling
- matplotlib/seaborn for visualization

## 🎯 Trading Strategy

### Signal Generation
1. **Feature Extraction**: Calculate all technical indicators
2. **Trend Confirmation**: Check 4h and 1d trend alignment
3. **Probability Calculation**: Neural network prediction
4. **Signal Decision**: Buy if probability > threshold

### Risk Management
- **Trend Alignment**: Only buy when higher timeframes support the trade
- **Minimum Gain**: 3% minimum gain requirement filters out low-probability trades
- **Position Sizing**: Fixed position size per trade
- **Time Management**: Strict 1-hour holding period

## 📊 Performance Expectations

The model is designed for:
- **Conservative Trading**: High probability, low frequency trades
- **Trend Following**: Aligns with higher timeframe trends
- **Risk Control**: 3% minimum gain requirement
- **Market Adaptation**: Learns from different market conditions

## 🔧 Customization

You can customize the model by:
- Adjusting technical indicator parameters
- Modifying the neural network architecture
- Changing the minimum gain requirement
- Adding new features or timeframes
- Adjusting the probability threshold

## 📝 Notes

- The model is trained on historical data and may not perform the same in live markets
- Always test with paper trading before using real funds
- Market conditions can change, requiring model retraining
- Consider transaction costs and slippage in live trading
