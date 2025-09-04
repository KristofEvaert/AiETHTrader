# 🚀 AiETHTrader Live Trading System

## Overview

This live trading system implements your AI-powered cryptocurrency trading strategy with efficient data management and real-time execution. The system is designed to:

- **Efficiently manage historical data** with intelligent caching
- **Update data every hour** to keep models current
- **Execute trades based on AI predictions** with proper risk management
- **Support multiple cryptocurrencies** with individual trained models
- **Provide comprehensive monitoring** and logging

## 🏗️ System Architecture

### Data Management Strategy

1. **Initial Load**: Download 2+ years of historical data for all timeframes (1h, 4h, 1d)
2. **Cache Management**: Store data in memory with automatic size limits
3. **Hourly Updates**: Update cache every hour with new data
4. **Rolling Window**: Remove oldest data when adding new to maintain constant cache size
5. **File Persistence**: Save cache to disk for quick startup

### Trading Engine

- **Multi-Model Support**: Load individual AI models for each cryptocurrency
- **Real-Time Predictions**: Generate buy signals every minute
- **Risk Management**: Enforce position limits and stop-losses
- **Trade Execution**: Execute trades with proper error handling
- **Performance Monitoring**: Track PnL and trade history

## 📁 File Structure

```
AiETHTrader/
├── data_manager.py              # Core data management system
├── live_trading_engine.py       # Main trading engine
├── start_live_trading.py        # Simple startup script
├── test_data_manager.py         # Test suite
├── config_live.yaml            # Live trading configuration
├── requirements_live.txt       # Additional dependencies
├── README_LIVE_TRADING.md      # This file
├── data/                       # Historical data cache
├── cache/                      # Live data cache
├── artifacts/                  # Trained models
└── logs/                       # Trading logs
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install additional live trading dependencies
pip install -r requirements_live.txt
```

### 2. Test the System

```bash
# Run the test suite
python test_data_manager.py
```

### 3. Start Live Trading (Dry Run)

```bash
# Start in dry-run mode (no real trades)
python start_live_trading.py
```

### 4. Start Live Trading (Real Trades)

```bash
# Start with real trades (BE CAREFUL!)
python start_live_trading.py --live --api-key YOUR_API_KEY --api-secret YOUR_API_SECRET
```

## ⚙️ Configuration

### Key Settings in `config_live.yaml`

```yaml
# Trading Parameters
live:
  dry_run: true                 # Start with dry run for safety
  trade_quote_size: 25.0        # Position size in USDC
  poll_seconds: 60              # Check signals every 60 seconds
  max_positions: 1              # Maximum concurrent positions

# Risk Management
risk_management:
  max_daily_trades: 10          # Maximum trades per day
  max_drawdown: 0.1             # 10% maximum drawdown
  stop_loss_pct: 0.05           # 5% stop loss

# Data Management
data_management:
  cache_size_limits:
    "1h": 17520                 # 2 years of hourly data
    "4h": 4380                  # 2 years of 4-hour data
    "1d": 730                   # 2 years of daily data
  update_frequency: 3600        # Update every hour
```

## 🔧 Data Management Features

### Intelligent Caching

- **Memory Efficient**: Maintains constant cache size
- **Fast Access**: Data stored in memory for quick model predictions
- **Persistent Storage**: Cache saved to disk for quick restarts
- **Automatic Updates**: Background thread updates data every hour

### Cache Size Limits

| Timeframe | Max Candles | Equivalent |
|-----------|-------------|------------|
| 1h        | 17,520      | 2 years    |
| 4h        | 4,380       | 2 years    |
| 1d        | 730         | 2 years    |

### Data Update Strategy

1. **Startup**: Load existing cache or download historical data
2. **Background Updates**: Update every hour at the top of the hour
3. **Rolling Window**: Remove oldest data when adding new
4. **Error Handling**: Retry failed updates with exponential backoff

## 🤖 Trading Engine Features

### Multi-Model Support

- **Individual Models**: Each cryptocurrency has its own trained AI model
- **Model Loading**: Automatic loading of all available models
- **Prediction Generation**: Real-time predictions every minute
- **Signal Filtering**: Only trade when probability > threshold

### Risk Management

- **Position Limits**: Maximum 1 position at a time
- **Daily Limits**: Maximum 10 trades per day
- **Drawdown Protection**: Stop trading if drawdown exceeds 10%
- **Stop Loss**: 5% stop loss on positions (if implemented)

### Trade Execution

- **Dry Run Mode**: Test without real money
- **Real Trading**: Execute actual trades on exchange
- **Error Handling**: Robust error handling and retry logic
- **Trade Logging**: Comprehensive trade history

## 📊 Monitoring and Logging

### Real-Time Status

The system provides real-time status updates including:
- Current bankroll
- Active positions
- Total trades executed
- Model status
- Cache status

### Log Files

- `live_trading.log`: Main trading log
- `data_manager.log`: Data management log
- `live_trade_history.csv`: Trade history

### Performance Tracking

- PnL tracking
- Win/loss ratios
- Drawdown monitoring
- Trade frequency analysis

## 🛡️ Safety Features

### Dry Run Mode

- **Default Mode**: Starts in dry run by default
- **No Real Trades**: Simulates trades without executing
- **Full Logging**: Logs all simulated trades
- **Easy Testing**: Test strategy without risk

### Risk Controls

- **Position Limits**: Maximum 1 position at a time
- **Daily Limits**: Maximum 10 trades per day
- **Drawdown Protection**: Automatic stop if drawdown too high
- **API Safety**: Uses testnet by default

### Error Handling

- **Robust Error Handling**: Continues running despite errors
- **Automatic Retries**: Retries failed operations
- **Graceful Degradation**: Continues with available data
- **Comprehensive Logging**: Logs all errors and warnings

## 🔄 Data Flow

### 1. Initialization
```
Start → Load Config → Initialize Data Manager → Load Models → Start Trading
```

### 2. Data Updates
```
Every Hour → Download Latest Data → Update Cache → Save to Disk
```

### 3. Trading Loop
```
Every Minute → Check Signals → Execute Trades → Update Positions → Log Results
```

## 📈 Performance Optimization

### Memory Management

- **Efficient Caching**: Only stores necessary data
- **Rolling Windows**: Maintains constant memory usage
- **Garbage Collection**: Automatic cleanup of old data

### API Optimization

- **Rate Limiting**: Respects exchange API limits
- **Batch Requests**: Efficient data retrieval
- **Error Recovery**: Handles API errors gracefully

### Model Optimization

- **Model Caching**: Models loaded once at startup
- **Efficient Predictions**: Fast inference with PyTorch
- **Feature Caching**: Reuse calculated features

## 🚨 Important Notes

### Safety First

1. **Always start in dry run mode**
2. **Test thoroughly before live trading**
3. **Use testnet for initial testing**
4. **Monitor logs carefully**
5. **Set appropriate risk limits**

### API Keys

- **Testnet**: Safe to use without real money
- **Live Trading**: Requires real API keys
- **Permissions**: Only enable trading permissions
- **Security**: Keep API keys secure

### Monitoring

- **Check logs regularly**
- **Monitor performance metrics**
- **Watch for errors**
- **Verify trade execution**

## 🆘 Troubleshooting

### Common Issues

1. **No Data**: Check internet connection and API keys
2. **Model Errors**: Verify models are trained and saved
3. **Cache Issues**: Delete cache files to force refresh
4. **API Errors**: Check API key permissions and limits

### Debug Mode

```bash
# Run with debug logging
python start_live_trading.py --config config_live.yaml --force
```

### Reset Cache

```bash
# Delete cache files to force fresh download
rm -rf data/* cache/*
```

## 📞 Support

If you encounter issues:

1. Check the logs in `live_trading.log`
2. Run the test suite: `python test_data_manager.py`
3. Verify your configuration in `config_live.yaml`
4. Check your internet connection and API keys

## 🎯 Next Steps

1. **Test in dry run mode** for several hours
2. **Monitor performance** and adjust parameters
3. **Start with small position sizes** when going live
4. **Gradually increase** position sizes as confidence grows
5. **Monitor and optimize** based on real performance

---

**⚠️ Disclaimer**: This is a trading system that involves financial risk. Always test thoroughly and start with small amounts. The authors are not responsible for any financial losses.
