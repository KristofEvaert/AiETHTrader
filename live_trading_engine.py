#!/usr/bin/env python3
"""
Live Trading Engine
==================

This module implements the live trading engine that:
1. Manages data updates and model predictions
2. Executes trades based on AI signals
3. Handles risk management and position sizing
4. Provides real-time monitoring and logging

Features:
- Real-time data updates
- Multi-coin model support
- Risk management
- Trade execution
- Performance monitoring
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import torch
import joblib
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import LiveDataManager
from crypto_ai_trader import TradingModel, FeatureEngineer

class LiveTradingEngine:
    """Main live trading engine."""
    
    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        """
        Initialize the live trading engine.
        
        Args:
            config: Configuration dictionary
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        
        # Trading parameters
        self.position_size = config['live']['trade_quote_size']
        self.threshold = config['backtest']['threshold']
        self.dry_run = config['live']['dry_run']
        self.poll_seconds = config['live']['poll_seconds']
        self.signal_check_interval = config['live'].get('signal_check_interval', 3600)  # Check signals every hour
        self.use_full_bankroll = config['live'].get('use_full_bankroll', True)  # Use entire bankroll
        self.fee_bps = config['backtest']['fee_bps']  # Trading fee in basis points (0.10% = 10 bps)
        
        # Signal timing
        self.last_signal_check = None
        self.next_signal_check = None
        
        # Initialize components
        from multi_coin_data_manager import LiveMultiCoinDataManager
        self.data_manager = LiveMultiCoinDataManager(config, api_key, api_secret)
        self.feature_engineer = FeatureEngineer()
        
        # Model cache
        self.models = {}
        self.scalers = {}
        
        # Trading state
        self.current_positions = {}
        self.trade_history = []
        self.bankroll = self.get_initial_bankroll()  # Get real or simulated bankroll
        self.is_running = False
        
        # Setup logging first
        self.setup_logging()
        
        # Load models
        self.load_models()
    
    def setup_logging(self):
        """Setup logging for live trading."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LiveTradingEngine')
    
    def load_models(self):
        """Load all trained models."""
        print("Loading trained models...")
        
        artifacts_dir = self.config['storage']['artifacts_dir']
        available_coins = [d for d in os.listdir(artifacts_dir) 
                          if os.path.isdir(os.path.join(artifacts_dir, d))]
        
        for coin in available_coins:
            try:
                # Load model
                model_path = f"{artifacts_dir}/{coin}/trading_model.pth"
                scaler_path = f"{artifacts_dir}/{coin}/scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = TradingModel(input_size=35, hidden_size=256, dropout=0.3)
                    model.load_state_dict(torch.load(model_path))
                    model.eval()
                    
                    scaler = joblib.load(scaler_path)
                    
                    self.models[coin] = model
                    self.scalers[coin] = scaler
                    
                    print(f"Loaded model for {coin}")
                else:
                    print(f"WARNING: Model files not found for {coin}")
                    
            except Exception as e:
                print(f"ERROR: Error loading model for {coin}: {e}")
        
        print(f"Loaded {len(self.models)} models")
    
    def get_initial_bankroll(self) -> float:
        """
        Get initial bankroll - real from exchange or simulated.
        
        Returns:
            Initial bankroll amount
        """
        if self.dry_run:
            # Simulated bankroll for dry run
            return 26.0
        else:
            # Get real USDC balance from exchange
            try:
                from binance.client import Client
                client = Client(self.api_key, self.api_secret, testnet=self.config.get('use_testnet', True))
                account = client.get_account()
                
                # Find USDC balance
                for balance in account['balances']:
                    if balance['asset'] == 'USDC':
                        usdc_balance = float(balance['free'])
                        print(f"Real USDC balance: {usdc_balance:.2f}")  # Use print instead of logger
                        return usdc_balance
                
                # If no USDC found, return 0
                print("WARNING: No USDC balance found on exchange")
                return 0.0
                
            except Exception as e:
                print(f"ERROR: Error fetching real bankroll: {e}")
                # Fallback to simulated bankroll if API fails
                print("WARNING: Falling back to simulated bankroll")
                return 26.0
    
    def add_technical_indicators(self, df):
        """Add technical indicators to dataframe."""
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def create_multi_timeframe_features(self, df_1h, df_4h, df_1d):
        """Create multi-timeframe features."""
        # Align timeframes
        df_1h['datetime'] = pd.to_datetime(df_1h['open_time'])
        df_4h['datetime'] = pd.to_datetime(df_4h['open_time'])
        df_1d['datetime'] = pd.to_datetime(df_1d['open_time'])
        
        # Add technical indicators
        df_1h = self.add_technical_indicators(df_1h)
        df_4h = self.add_technical_indicators(df_4h)
        df_1d = self.add_technical_indicators(df_1d)
        
        # Create trend features
        df_4h['4h_trend'] = np.where(df_4h['close'] > df_4h['sma_20'], 1, 0)
        df_4h['4h_momentum'] = df_4h['close'].pct_change(1)
        
        df_1d['1d_trend'] = np.where(df_1d['close'] > df_1d['sma_20'], 1, 0)
        df_1d['1d_momentum'] = df_1d['close'].pct_change(1)
        
        # Merge timeframes
        df_1h['hour'] = df_1h['datetime'].dt.hour
        df_1h['day_of_week'] = df_1h['datetime'].dt.dayofweek
        
        # Forward fill 4h and 1d data
        df_4h_aligned = df_4h.set_index('datetime').reindex(df_1h['datetime'], method='ffill')
        df_1d_aligned = df_1d.set_index('datetime').reindex(df_1h['datetime'], method='ffill')
        
        # Add 4h features
        for col in ['4h_trend', '4h_momentum', 'rsi', 'macd', 'bb_position']:
            if col in df_4h_aligned.columns:
                df_1h[f'4h_{col}'] = df_4h_aligned[col].values
        
        # Add 1d features
        for col in ['1d_trend', '1d_momentum', 'rsi', 'macd', 'bb_position']:
            if col in df_1d_aligned.columns:
                df_1h[f'1d_{col}'] = df_1d_aligned[col].values
        
        return df_1h
    
    def prepare_features(self, df):
        """Prepare features for model prediction."""
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volume_ratio',
            'price_change_1', 'price_change_3', 'price_change_5',
            'volatility', 'hl_spread', 'hour', 'day_of_week',
            '4h_4h_trend', '4h_4h_momentum', '4h_rsi', '4h_macd', '4h_bb_position',
            '1d_1d_trend', '1d_1d_momentum', '1d_rsi', '1d_macd', '1d_bb_position'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0)
        
        return X
    
    def get_current_price(self, coin: str) -> float:
        """
        Get current price for a coin.
        In live trading, fetches real-time price from exchange.
        In dry run, uses simulated prices.
        
        Args:
            coin: Coin symbol
            
        Returns:
            Current price
        """
        if self.dry_run:
            # Simulated prices for dry run
            price_simulation = {
                'ETHUSDC': 2500.0,
                'ADAUSDC': 0.45,
                'BTCUSDC': 45000.0,
                'DOGEUSDC': 0.08,
                'XRPUSDC': 0.52
            }
            
            # Add some random variation for demo
            import random
            base_price = price_simulation.get(coin, 1.0)
            variation = random.uniform(0.95, 1.05)  # ±5% variation
            return base_price * variation
        else:
            # Real API call to get current price
            try:
                from binance.client import Client
                client = Client(self.api_key, self.api_secret, testnet=self.config.get('use_testnet', True))
                ticker = client.get_symbol_ticker(symbol=coin)
                return float(ticker['price'])
            except Exception as e:
                self.logger.error(f"Error fetching real price for {coin}: {e}")
                # Fallback to simulated price if API fails
                price_simulation = {
                    'ETHUSDC': 2500.0,
                    'ADAUSDC': 0.45,
                    'BTCUSDC': 45000.0,
                    'DOGEUSDC': 0.08,
                    'XRPUSDC': 0.52
                }
                return price_simulation.get(coin, 1.0)
    
    def get_prediction(self, coin: str) -> Tuple[float, bool]:
        """
        Get prediction for a specific coin.
        
        Args:
            coin: Coin symbol
            
        Returns:
            Tuple of (probability, should_trade)
        """
        try:
            if coin not in self.models:
                return 0.0, False
            
            # Get data for the specific coin
            df_1h = self.data_manager.get_model_data(coin, '1h')
            df_4h = self.data_manager.get_model_data(coin, '4h')
            df_1d = self.data_manager.get_model_data(coin, '1d')
            
            if df_1h.empty or df_4h.empty or df_1d.empty:
                return 0.0, False
            
            # Create features
            df_combined = self.create_multi_timeframe_features(df_1h, df_4h, df_1d)
            X = self.prepare_features(df_combined)
            
            if X.empty:
                return 0.0, False
            
            # Get latest features
            latest_features = X.iloc[-1:].values
            
            # Scale features
            scaler = self.scalers[coin]
            X_scaled = scaler.transform(latest_features)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Get prediction
            model = self.models[coin]
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                buy_prob = probabilities[0, 1].item()
            
            should_trade = buy_prob >= self.threshold
            
            return buy_prob, should_trade
            
        except Exception as e:
            self.logger.error(f"Error getting prediction for {coin}: {e}")
            return 0.0, False
    
    def calculate_fee(self, amount: float, is_buy: bool = True) -> float:
        """
        Calculate trading fee for a given amount.
        
        Args:
            amount: Amount to calculate fee for
            is_buy: True for buy orders, False for sell orders
            
        Returns:
            Fee amount
        """
        fee_rate = self.fee_bps / 10000  # Convert basis points to decimal
        return amount * fee_rate
    
    def execute_trade(self, coin: str, action: str, price: float, amount: float = None) -> tuple:
        """
        Execute a trade with proper fee calculation.
        In live mode, executes real trades on exchange.
        In dry run mode, simulates trades.
        
        Args:
            coin: Coin symbol
            action: 'buy' or 'sell'
            price: Current price
            amount: Amount to trade (if None, uses full bankroll logic)
            
        Returns:
            Tuple of (success, actual_amount_bought/sold, fee_paid)
        """
        try:
            if self.dry_run:
                # Simulated trade execution
                return self._execute_simulated_trade(coin, action, price, amount)
            else:
                # Real trade execution
                return self._execute_real_trade(coin, action, price, amount)
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False, 0, 0
    
    def _execute_simulated_trade(self, coin: str, action: str, price: float, amount: float = None) -> tuple:
        """Execute simulated trade for dry run mode."""
        if action == 'buy':
            if self.use_full_bankroll:
                # Use entire bankroll to buy as much coin as possible
                usdc_available = self.bankroll
                fee = self.calculate_fee(usdc_available, is_buy=True)
                usdc_after_fee = usdc_available - fee
                coin_amount = usdc_after_fee / price
                
                self.logger.info(f"[DRY RUN] BUY {coin_amount:.6f} {coin} for {usdc_available:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
                return True, coin_amount, fee
            else:
                # Use fixed position size
                usdc_amount = self.position_size
                fee = self.calculate_fee(usdc_amount, is_buy=True)
                usdc_after_fee = usdc_amount - fee
                coin_amount = usdc_after_fee / price
                
                self.logger.info(f"[DRY RUN] BUY {coin_amount:.6f} {coin} for {usdc_amount:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
                return True, coin_amount, fee
            
        elif action == 'sell':
            # Sell the exact amount of coin we have
            coin_amount = amount  # Amount of coin we have
            usdc_before_fee = coin_amount * price
            fee = self.calculate_fee(usdc_before_fee, is_buy=False)
            usdc_after_fee = usdc_before_fee - fee
            
            self.logger.info(f"[DRY RUN] SELL {coin_amount:.6f} {coin} for {usdc_before_fee:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
            return True, usdc_after_fee, fee
    
    def _execute_real_trade(self, coin: str, action: str, price: float, amount: float = None) -> tuple:
        """Execute real trade on exchange."""
        try:
            from binance.client import Client
            client = Client(self.api_key, self.api_secret, testnet=self.config.get('use_testnet', True))
            
            if action == 'buy':
                if self.use_full_bankroll:
                    # Get current USDC balance
                    account = client.get_account()
                    usdc_balance = 0.0
                    for balance in account['balances']:
                        if balance['asset'] == 'USDC':
                            usdc_balance = float(balance['free'])
                            break
                    
                    if usdc_balance <= 0:
                        self.logger.error("Insufficient USDC balance for trade")
                        return False, 0, 0
                    
                    # Calculate amount to buy (accounting for fees)
                    fee_rate = self.fee_bps / 10000
                    usdc_after_fee = usdc_balance / (1 + fee_rate)
                    coin_amount = usdc_after_fee / price
                    fee = usdc_balance - usdc_after_fee
                    
                    # Execute market buy order
                    order = client.order_market_buy(
                        symbol=coin,
                        quoteOrderQty=usdc_balance  # Buy with all available USDC
                    )
                    
                    self.logger.info(f"[LIVE] BUY {coin_amount:.6f} {coin} for {usdc_balance:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
                    return True, coin_amount, fee
                else:
                    # Use fixed position size
                    usdc_amount = self.position_size
                    fee = self.calculate_fee(usdc_amount, is_buy=True)
                    usdc_after_fee = usdc_amount - fee
                    coin_amount = usdc_after_fee / price
                    
                    # Execute market buy order
                    order = client.order_market_buy(
                        symbol=coin,
                        quoteOrderQty=usdc_amount
                    )
                    
                    self.logger.info(f"[LIVE] BUY {coin_amount:.6f} {coin} for {usdc_amount:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
                    return True, coin_amount, fee
                    
            elif action == 'sell':
                # Sell the exact amount of coin we have
                coin_amount = amount
                
                # Execute market sell order
                order = client.order_market_sell(
                    symbol=coin,
                    quantity=coin_amount
                )
                
                # Calculate fees and proceeds
                usdc_before_fee = coin_amount * price
                fee = self.calculate_fee(usdc_before_fee, is_buy=False)
                usdc_after_fee = usdc_before_fee - fee
                
                self.logger.info(f"[LIVE] SELL {coin_amount:.6f} {coin} for {usdc_before_fee:.2f} USDC (fee: {fee:.2f} USDC) at ${price:.2f}")
                return True, usdc_after_fee, fee
                
        except Exception as e:
            self.logger.error(f"Error executing real trade: {e}")
            return False, 0, 0
    
    def check_trading_signals(self):
        """Check for trading signals across all coins and select the best one."""
        self.logger.info("Checking trading signals at bar open...")
        
        signals = {}
        buy_signals = []
        
        # Collect all signals first
        for coin in self.models.keys():
            try:
                prob, should_trade = self.get_prediction(coin)
                signals[coin] = {
                    'probability': prob,
                    'should_trade': should_trade,
                    'timestamp': datetime.now()
                }
                
                if should_trade:
                    buy_signals.append({
                        'coin': coin,
                        'probability': prob,
                        'timestamp': datetime.now()
                    })
                    self.logger.info(f"BUY signal for {coin}: {prob:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error checking signals for {coin}: {e}")
        
        # If we have buy signals and can open a position
        if buy_signals and len(self.current_positions) == 0 and self.bankroll >= self.position_size:
            # Sort by probability (highest first)
            buy_signals.sort(key=lambda x: x['probability'], reverse=True)
            
            # Take the signal with highest probability
            best_signal = buy_signals[0]
            coin = best_signal['coin']
            prob = best_signal['probability']
            
            self.logger.info(f"SELECTED BEST SIGNAL: {coin} with probability {prob:.3f}")
            
            # Log all available signals for comparison
            if len(buy_signals) > 1:
                self.logger.info("Available signals:")
                for i, signal in enumerate(buy_signals):
                    status = "SELECTED" if i == 0 else "SKIPPED"
                    self.logger.info(f"  {i+1}. {signal['coin']}: {signal['probability']:.3f} {status}")
            
            # Get current price (would come from exchange in live trading)
            current_price = self.get_current_price(coin)
            
            # Execute buy order for the best signal
            success, coin_amount, buy_fee = self.execute_trade(coin, 'buy', current_price)
            if success:
                if self.use_full_bankroll:
                    # Use entire bankroll
                    usdc_spent = self.bankroll
                    if self.dry_run:
                        self.bankroll = 0  # All USDC converted to coin (simulated)
                    else:
                        # In live mode, bankroll will be updated by real trade execution
                        pass
                else:
                    # Use fixed position size
                    usdc_spent = self.position_size
                    if self.dry_run:
                        self.bankroll -= self.position_size
                    else:
                        # In live mode, bankroll will be updated by real trade execution
                        pass
                
                self.current_positions[coin] = {
                    'entry_time': datetime.now(),
                    'entry_price': current_price,
                    'coin_amount': coin_amount,  # Amount of coin we actually bought (after fees)
                    'usdc_spent': usdc_spent,    # Amount of USDC we spent (including fees)
                    'buy_fee': buy_fee,          # Fee paid on buy
                    'signal_probability': prob,
                    'competing_signals': len(buy_signals) - 1  # Number of other signals we skipped
                }
                self.logger.info(f"Opened position in {coin}: {coin_amount:.6f} {coin} for {usdc_spent:.2f} USDC (fee: {buy_fee:.2f} USDC, probability: {prob:.3f})")
        
        elif buy_signals and len(self.current_positions) > 0:
            # We have signals but already have a position
            self.logger.info(f"WARNING: {len(buy_signals)} signals available but already have position")
            for signal in buy_signals:
                self.logger.info(f"  - {signal['coin']}: {signal['probability']:.3f}")
        
        elif buy_signals and self.bankroll < self.position_size:
            # We have signals but insufficient funds
            self.logger.info(f"WARNING: {len(buy_signals)} signals available but insufficient funds (${self.bankroll:.2f})")
            for signal in buy_signals:
                self.logger.info(f"  - {signal['coin']}: {signal['probability']:.3f}")
        
        return signals
    
    def check_exit_signals(self):
        """Check for exit signals on current positions."""
        current_time = datetime.now()
        
        for coin, position in list(self.current_positions.items()):
            try:
                # Check if position should be closed (1 hour has passed)
                entry_time = position['entry_time']
                if (current_time - entry_time).total_seconds() >= 3600:  # 1 hour
                    
                    # Get current price (would come from exchange in live trading)
                    exit_price = self.get_current_price(coin)
                    coin_amount = position['coin_amount']
                    
                    # Execute sell order
                    success, usdc_received, sell_fee = self.execute_trade(coin, 'sell', exit_price, coin_amount)
                    if success:
                        # Calculate PnL
                        entry_price = position['entry_price']
                        usdc_spent = position['usdc_spent']
                        total_fees = position['buy_fee'] + sell_fee
                        pnl = usdc_received - usdc_spent
                        
                        # Add USDC back to bankroll (after sell fees)
                        if self.dry_run:
                            self.bankroll = usdc_received
                        else:
                            # In live mode, get updated balance from exchange
                            self.bankroll = self.get_initial_bankroll()
                        
                        # Record trade
                        trade_record = {
                            'coin': coin,
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'coin_amount': coin_amount,
                            'usdc_spent': usdc_spent,
                            'usdc_received': usdc_received,
                            'buy_fee': position['buy_fee'],
                            'sell_fee': sell_fee,
                            'total_fees': total_fees,
                            'pnl': pnl,
                            'pnl_percentage': (pnl / usdc_spent) * 100,
                            'bankroll_after': self.bankroll,
                            'signal_probability': position.get('signal_probability', 0.0),
                            'competing_signals': position.get('competing_signals', 0)
                        }
                        self.trade_history.append(trade_record)
                        
                        # Remove position
                        del self.current_positions[coin]
                        
                        self.logger.info(f"Closed position in {coin}: {coin_amount:.6f} {coin} -> {usdc_received:.2f} USDC (PnL: ${pnl:.2f}, {trade_record['pnl_percentage']:.2f}%, fees: ${total_fees:.2f})")
                
            except Exception as e:
                self.logger.error(f"Error checking exit for {coin}: {e}")
    
    def get_status(self) -> dict:
        """Get current trading status."""
        return {
            'is_running': self.is_running,
            'bankroll': self.bankroll,
            'positions': len(self.current_positions),
            'total_trades': len(self.trade_history),
            'models_loaded': len(self.models),
            'current_positions': self.current_positions,
            'cache_status': self.data_manager.get_cache_status()
        }
    
    def log_status(self):
        """Log current status."""
        status = self.get_status()
        current_time = datetime.now()
        next_check = self.next_signal_check.strftime("%H:%M:%S") if self.next_signal_check else "N/A"
        
        self.logger.info(f"Status: Bankroll=${status['bankroll']:.2f}, Positions={status['positions']}, Models={status['models_loaded']}, Next Signal Check: {next_check}")
    
    def start_trading(self):
        """Start the live trading loop."""
        self.logger.info("Starting live trading engine...")
        
        # Initialize data manager
        if not self.data_manager.initialize():
            self.logger.error("Failed to initialize data manager")
            return False
        
        # Set initial signal check time (2 minutes after the hour to align with data updates)
        now = datetime.now()
        self.next_signal_check = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=2)
        self.logger.info(f"Next signal check scheduled for: {self.next_signal_check.strftime('%H:%M:%S')}")
        
        # Check trading signals at bar open (initial check)
        self.logger.info("Checking trading signals at bar open...")
        self.check_trading_signals()
        
        self.is_running = True
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check if it's time for signal checking (at the top of the hour)
                if self.next_signal_check and current_time >= self.next_signal_check:
                    self.logger.info("Checking trading signals at bar open...")
                    self.check_trading_signals()
                    
                    # Update next signal check time (2 minutes after the hour to align with data updates)
                    self.next_signal_check = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=2)
                    self.logger.info(f"Next signal check scheduled for: {self.next_signal_check.strftime('%H:%M:%S')}")
                
                # Always check exit signals (every minute)
                self.check_exit_signals()
                
                # Log status
                self.log_status()
                
                # Wait for next check
                time.sleep(self.poll_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal, stopping...")
        except Exception as e:
            self.logger.error(f"Error in trading loop: {e}")
        finally:
            self.is_running = False
            self.logger.info("Live trading engine stopped")
        
        return True
    
    def stop_trading(self):
        """Stop the live trading engine."""
        self.logger.info("Stopping live trading engine...")
        self.is_running = False
    
    def save_trade_history(self):
        """Save trade history to file."""
        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv('live_trade_history.csv', index=False)
            self.logger.info(f"Saved {len(self.trade_history)} trades to live_trade_history.csv")

def main():
    """Main function for live trading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Trading Engine')
    parser.add_argument('--config', default='config_1h_trading.yaml', help='Configuration file')
    parser.add_argument('--api-key', help='Binance API key')
    parser.add_argument('--api-secret', help='Binance API secret')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override dry-run if specified
    if args.dry_run:
        config['live']['dry_run'] = True
    
    # Initialize trading engine
    engine = LiveTradingEngine(config, args.api_key, args.api_secret)
    
    # Start trading
    try:
        engine.start_trading()
    except KeyboardInterrupt:
        print("Stopping trading engine...")
    finally:
        engine.save_trade_history()
        print("Trading engine stopped")

if __name__ == "__main__":
    main()
