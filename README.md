# AiETHTrader

An intelligent Python-based Ethereum trading bot designed for automated cryptocurrency trading with profit-taking and stop-loss capabilities.

## Features

- **Long-only trading**: Focuses on buying ETH and selling for profit or stop loss
- **Serial execution**: Trades are executed one at a time, ensuring controlled risk management
- **Automated trading**: No manual intervention required once configured
- **Profit taking**: Automatically sells when profit targets are reached
- **Stop loss protection**: Minimizes losses with configurable stop-loss levels
- **Real-time monitoring**: Continuous market analysis and trade execution

## Requirements

- Python 3.8+
- Internet connection for real-time market data
- Cryptocurrency exchange API access
- Sufficient funds for trading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KristofEvaert/AiETHTrader.git
cd AiETHTrader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API keys and trading parameters in `config/config.py`

## Configuration

Create a `config/config.py` file with your trading parameters:

```python
# Exchange API configuration
EXCHANGE_API_KEY = "your_api_key_here"
EXCHANGE_SECRET_KEY = "your_secret_key_here"

# Trading parameters
TRADE_AMOUNT = 0.01  # Amount of ETH to trade
PROFIT_TARGET = 0.05  # 5% profit target
STOP_LOSS = 0.03     # 3% stop loss
```

## Usage

Run the trading bot:

```bash
python main.py
```

## Safety Features

- **No short selling**: Only long positions are taken
- **Serial execution**: Maximum one trade at a time
- **Stop loss protection**: Automatic loss limitation
- **API key security**: Sensitive credentials are excluded from version control

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Use at your own risk and never invest more than you can afford to lose.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support, please open an issue on GitHub.
