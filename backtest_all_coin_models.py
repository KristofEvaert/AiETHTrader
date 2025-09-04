#!/usr/bin/env python3
"""
Backtest All Coin Models
========================

This script runs backtests on all the newly trained coin-specific models
to compare their performance with the same constraints:
- Maximum 1 trade at a time
- 25 USDC position size per trade
- 26 USDC initial bankroll
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
import joblib
import glob
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ai_trader import TradingModel

class AllCoinBacktester:
    """Backtester for all coin-specific models."""
    
    def __init__(self, config):
        self.config = config
        self.initial_bankroll = 26.0  # USDC
        self.position_size = 25.0     # USDC per trade
        self.fee_rate = config['backtest']['fee_bps'] / 10000  # Convert bps to decimal
        self.threshold = config['backtest']['threshold']
        
    def get_available_coins(self, artifacts_dir):
        """Get list of available coin models."""
        coin_dirs = [d for d in os.listdir(artifacts_dir) 
                    if os.path.isdir(os.path.join(artifacts_dir, d)) and d != '__pycache__']
        return sorted(coin_dirs)
    
    def load_coin_model(self, coin, artifacts_dir):
        """Load the trained model and scaler for a specific coin."""
        try:
            # Load scaler
            scaler = joblib.load(f"{artifacts_dir}/{coin}/scaler.pkl")
            
            # Load model
            model = TradingModel(input_size=35, hidden_size=256, dropout=0.3)
            model.load_state_dict(torch.load(f"{artifacts_dir}/{coin}/trading_model.pth"))
            model.eval()
            
            return model, scaler
        except Exception as e:
            print(f"Error loading model for {coin}: {e}")
            return None, None
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
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
        """Create features using multiple timeframes."""
        # Align timeframes
        df_1h['datetime'] = pd.to_datetime(df_1h['open_time'])
        df_4h['datetime'] = pd.to_datetime(df_4h['open_time'])
        df_1d['datetime'] = pd.to_datetime(df_1d['open_time'])
        
        # Add technical indicators to each timeframe
        df_1h = self.add_technical_indicators(df_1h)
        df_4h = self.add_technical_indicators(df_4h)
        df_1d = self.add_technical_indicators(df_1d)
        
        # Create 4h trend features
        df_4h['4h_trend'] = np.where(df_4h['close'] > df_4h['sma_20'], 1, 0)
        df_4h['4h_momentum'] = df_4h['close'].pct_change(1)
        
        # Create 1d trend features
        df_1d['1d_trend'] = np.where(df_1d['close'] > df_1d['sma_20'], 1, 0)
        df_1d['1d_momentum'] = df_1d['close'].pct_change(1)
        
        # Merge timeframes
        df_1h['hour'] = df_1h['datetime'].dt.hour
        df_1h['day_of_week'] = df_1h['datetime'].dt.dayofweek
        
        # Forward fill 4h and 1d data to align with 1h
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
        """Prepare features for prediction."""
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
    
    def run_backtest_for_coin(self, coin, model, scaler, data_dir):
        """Run backtest for a specific coin using its trained model."""
        print(f"\n{'='*60}")
        print(f"BACKTESTING {coin} WITH ITS OWN MODEL")
        print(f"{'='*60}")
        
        try:
            # Load data for different timeframes
            df_1h = pd.read_csv(f"{data_dir}/{coin}_1h.csv")
            df_4h = pd.read_csv(f"{data_dir}/{coin}_4h.csv")
            df_1d = pd.read_csv(f"{data_dir}/{coin}_1d.csv")
            
            print(f"Loaded data: {len(df_1h)} 1h bars, {len(df_4h)} 4h bars, {len(df_1d)} 1d bars")
            
            # Create multi-timeframe features
            df_combined = self.create_multi_timeframe_features(df_1h, df_4h, df_1d)
            
            # Prepare features
            X = self.prepare_features(df_combined)
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                buy_prob = probabilities[:, 1].numpy()
            
            # Initialize tracking variables
            bankroll = self.initial_bankroll
            current_position = None
            trades = []
            trade_count = 0
            
            # Process each bar
            for i in range(len(df_combined)):
                current_bar = df_combined.iloc[i]
                current_time = pd.to_datetime(current_bar['open_time'])
                
                # Check if we have an open position
                if current_position is not None:
                    # Close the position at the end of the current bar
                    entry_price = current_position['entry_price']
                    exit_price = current_bar['close']
                    
                    # Calculate return
                    price_return = (exit_price - entry_price) / entry_price
                    
                    # Calculate fees (entry + exit)
                    entry_fee = self.position_size * self.fee_rate
                    exit_fee = self.position_size * (1 + price_return) * self.fee_rate
                    total_fees = entry_fee + exit_fee
                    
                    # Calculate PnL
                    gross_pnl = self.position_size * price_return
                    net_pnl = gross_pnl - total_fees
                    
                    # Update bankroll
                    bankroll += net_pnl
                    
                    # Record trade
                    trade = {
                        'coin': coin,
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': self.position_size,
                        'price_return': price_return,
                        'gross_pnl': gross_pnl,
                        'fees': total_fees,
                        'net_pnl': net_pnl,
                        'bankroll_after': bankroll,
                        'trade_number': trade_count + 1
                    }
                    trades.append(trade)
                    trade_count += 1
                    
                    print(f"Trade #{trade_count}: {current_position['entry_time'].strftime('%Y-%m-%d %H:%M')} -> {current_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"  Entry: ${entry_price:.2f} | Exit: ${exit_price:.2f}")
                    print(f"  Return: {price_return:.2%} | PnL: ${net_pnl:.2f} | Bankroll: ${bankroll:.2f}")
                    
                    # Close position
                    current_position = None
                
                # Check for new entry signal (only if no position open)
                if current_position is None and buy_prob[i] >= self.threshold:
                    # Check if we have enough capital
                    if bankroll >= self.position_size:
                        # Open new position
                        current_position = {
                            'entry_time': current_time,
                            'entry_price': current_bar['open']
                        }
                        print(f"Opening position at {current_time.strftime('%Y-%m-%d %H:%M')} - Price: ${current_bar['open']:.2f} (Prob: {buy_prob[i]:.3f})")
            
            # Calculate results
            total_trades = len(trades)
            if total_trades > 0:
                winning_trades = len([t for t in trades if t['net_pnl'] > 0])
                win_rate = winning_trades / total_trades
                total_pnl = sum(t['net_pnl'] for t in trades)
                total_return = (bankroll - self.initial_bankroll) / self.initial_bankroll
                avg_pnl = total_pnl / total_trades
            else:
                winning_trades = 0
                win_rate = 0
                total_pnl = 0
                total_return = 0
                avg_pnl = 0
            
            print(f"\n{coin} Results:")
            print(f"  Total Trades: {total_trades}")
            print(f"  Win Rate: {win_rate:.2%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Final Bankroll: ${bankroll:.2f}")
            
            return {
                'coin': coin,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'total_return': total_return,
                'final_bankroll': bankroll,
                'avg_pnl': avg_pnl,
                'trades': trades
            }
            
        except Exception as e:
            print(f"Error backtesting {coin}: {e}")
            return {
                'coin': coin,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'final_bankroll': self.initial_bankroll,
                'avg_pnl': 0,
                'trades': []
            }
    
    def run_all_backtests(self, artifacts_dir, data_dir):
        """Run backtests on all available coin models."""
        coins = self.get_available_coins(artifacts_dir)
        print(f"Found coin models: {coins}")
        
        all_results = []
        all_trades = []
        
        for coin in coins:
            # Load the coin-specific model
            model, scaler = self.load_coin_model(coin, artifacts_dir)
            
            if model is not None and scaler is not None:
                result = self.run_backtest_for_coin(coin, model, scaler, data_dir)
                all_results.append(result)
                all_trades.extend(result['trades'])
            else:
                print(f"Skipping {coin} due to model loading error")
        
        return all_results, all_trades
    
    def create_summary_report(self, all_results):
        """Create a summary report of all backtests."""
        print(f"\n{'='*80}")
        print("ALL COIN MODELS BACKTEST SUMMARY")
        print(f"{'='*80}")
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Coin': result['coin'],
                'Total Trades': result['total_trades'],
                'Win Rate': f"{result['win_rate']:.1%}",
                'Total PnL': f"${result['total_pnl']:.2f}",
                'Total Return': f"{result['total_return']:.1%}",
                'Final Bankroll': f"${result['final_bankroll']:.2f}",
                'Avg PnL/Trade': f"${result['avg_pnl']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Overall statistics
        total_trades_all = sum(r['total_trades'] for r in all_results)
        total_pnl_all = sum(r['total_pnl'] for r in all_results)
        avg_return = np.mean([r['total_return'] for r in all_results if r['total_trades'] > 0])
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Total Trades Across All Coins: {total_trades_all}")
        print(f"Total PnL Across All Coins: ${total_pnl_all:.2f}")
        print(f"Average Return per Coin: {avg_return:.1%}")
        
        # Best and worst performers
        if all_results:
            best_coin = max(all_results, key=lambda x: x['total_return'])
            worst_coin = min(all_results, key=lambda x: x['total_return'])
            
            print(f"\nBEST PERFORMER: {best_coin['coin']} ({best_coin['total_return']:.1%})")
            print(f"WORST PERFORMER: {worst_coin['coin']} ({worst_coin['total_return']:.1%})")
        
        return summary_df

def main():
    """Main function to run backtests on all coin models."""
    print("All Coin Models Backtest")
    print("=" * 60)
    print("Testing each coin with its own trained model")
    print("Constraints: Max 1 trade at a time, $25 position size, $26 bankroll")
    print("=" * 60)
    
    # Load configuration
    with open('config_1h_trading.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize backtester
    backtester = AllCoinBacktester(config)
    
    # Run backtests on all coin models
    all_results, all_trades = backtester.run_all_backtests(
        config['storage']['artifacts_dir'], 
        config['storage']['data_dir']
    )
    
    # Create summary report
    summary_df = backtester.create_summary_report(all_results)
    
    # Save results
    summary_df.to_csv('all_coin_models_backtest_summary.csv', index=False)
    
    # Save all trades
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv('all_coin_models_backtest_all_trades.csv', index=False)
        print(f"\nAll trade details saved to: all_coin_models_backtest_all_trades.csv")
    
    print(f"Summary saved to: all_coin_models_backtest_summary.csv")

if __name__ == "__main__":
    main()
