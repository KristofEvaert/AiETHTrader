#!/usr/bin/env python3
"""
AiETHTrader - Main Entry Point
==============================

This is the main entry point for the AiETHTrader bot.
It provides a simple interface to run the AI crypto trading pipeline.

Usage:
    python main.py --train          # Train the model and run backtest
    python main.py --live           # Run live trading loop
    python main.py --config config.yaml --train  # Use custom config
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main trading pipeline
from crypto_ai_trader import main

if __name__ == "__main__":
    main()
