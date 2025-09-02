# AiETHTrader

An intelligent Python-based **AI Crypto Trading Bot** designed for automated Ethereum trading using machine learning. The bot analyzes multiple timeframes, trains a PyTorch neural network to predict price movements, and executes trades with profit-taking and stop-loss capabilities.

## 🚀 Features

- **🤖 AI-Powered Trading**: Uses PyTorch neural networks to predict price direction
- **📊 Multi-Timeframe Analysis**: Analyzes 1m, 5m, 15m, 30m, 1h, 4h, 1d intervals
- **📈 Technical Indicators**: RSI, MACD, EMA, volume analysis, and more
- **🔒 Long-Only Strategy**: Focuses on buying ETH and selling for profit or stop loss
- **⚡ Serial Execution**: Trades are executed one at a time, ensuring controlled risk management
- **🧪 Backtesting**: Comprehensive backtesting with performance metrics
- **🌐 Live Trading**: Optional live trading loop with Binance integration
- **🛡️ Safety Features**: Testnet support, dry-run mode, and configurable risk parameters

## 🏗️ Architecture

The bot uses a sophisticated pipeline:

1. **Data Collection**: Downloads historical klines from Binance
2. **Feature Engineering**: Creates technical indicators across multiple timeframes
3. **Model Training**: Trains a PyTorch MLP classifier on historical data
4. **Backtesting**: Validates strategy performance on out-of-sample data
5. **Live Trading**: Executes trades based on real-time predictions

## 📋 Requirements

- Python 3.8+
- Internet connection for real-time market data
- Binance API access (testnet recommended for development)
- Sufficient funds for trading

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/KristofEvaert/AiETHTrader.git
cd AiETHTrader
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your Binance API keys
# Get testnet keys from: https://testnet.binance.vision/
```

### 4. Train the Model
```bash
# Download data, train model, and run backtest
python main.py --train
```

### 5. Run Live Trading (Optional)
```bash
# Start live trading loop (uses testnet by default)
python main.py --live
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_key_here
USE_TESTNET=true          # Use testnet for safety
SYMBOL=ETHUSDC           # Trading pair
TRADE_QUOTE_SIZE=50      # Trade size in USDC
DRY_RUN=true            # Dry run mode (no real trades)
```

### YAML Configuration (config.yaml)
```yaml
symbol: "ETHUSDC"
timeframes: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
base_timeframe: "15m"
train:
  epochs: 20
  batch_size: 512
  lr: 0.001
backtest:
  threshold: 0.55    # Probability threshold for entry
  fee_bps: 10        # Trading fee (0.10%)
```

## 📊 Usage Examples

### Training and Backtesting
```bash
# Use default configuration
python main.py --train

# Use custom configuration
python main.py --config config.yaml --train
```

### Live Trading
```bash
# Start live trading (loads trained model)
python main.py --live

# Use custom config for live trading
python main.py --config config.yaml --live
```

### Custom Configuration
```bash
# Create your own config file
cp config.yaml my_config.yaml
# Edit my_config.yaml with your preferences
python main.py --config my_config.yaml --train
```

## 🔧 Advanced Usage

### Model Architecture
The bot uses a Multi-Layer Perceptron (MLP) with:
- Input layer: Technical indicators from multiple timeframes
- Hidden layers: Configurable size with ReLU activation and dropout
- Output layer: Sigmoid activation for probability prediction

### Feature Engineering
- **Price-based**: Returns, ranges, moving averages
- **Technical indicators**: RSI, MACD, EMA
- **Volume analysis**: Volume moving averages
- **Multi-timeframe alignment**: Features from different intervals

### Trading Strategy
1. **Entry**: Buy when model predicts >55% probability of price increase
2. **Hold**: Keep position for configurable number of bars
3. **Exit**: Sell at market price after hold period
4. **Risk Management**: Configurable position sizes and testnet support

## 📁 Project Structure

```
AiETHTrader/
├── main.py                 # Main entry point
├── crypto_ai_trader.py     # Core AI trading pipeline
├── config.yaml             # Configuration template
├── env.example             # Environment variables template
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── data/                  # Downloaded market data
└── artifacts/             # Trained models and results
```

## 🛡️ Safety Features

- **No Short Selling**: Only long positions are taken
- **Serial Execution**: Maximum one trade at a time
- **Testnet Support**: Safe testing environment
- **Dry Run Mode**: Test without real money
- **Configurable Limits**: Adjustable position sizes and thresholds
- **API Key Security**: Sensitive credentials excluded from version control

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

**Always test on testnet first and never invest more than you can afford to lose.**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- Bug reports
- Feature requests
- Pull requests
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or support:
1. Check the documentation above
2. Review the code comments
3. Open an issue on GitHub
4. Test with testnet first

## 🔄 Updates

The bot is actively maintained and updated with:
- Latest machine learning techniques
- Improved feature engineering
- Enhanced risk management
- Better performance monitoring

---

**Happy Trading! 🚀📈**
