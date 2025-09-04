#!/usr/bin/env python3
"""
Single Trade Backtest for 1-Hour Trading Model
==============================================

This script runs a backtest with the following constraints:
- Maximum 1 trade at a time
- 25 USDC position size per trade
- 26 USDC initial bankroll
- Shows detailed PnL progression
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ai_trader import TradingModel

class SingleTradeBacktester:
    """Backtester that enforces maximum 1 trade at a time."""
    
    def __init__(self, config):
        self.config = config
        self.initial_bankroll = 26.0  # USDC
        self.position_size = 25.0     # USDC per trade
        self.fee_rate = config['backtest']['fee_bps'] / 10000  # Convert bps to decimal
        self.threshold = config['backtest']['threshold']
        
    def load_model(self, artifacts_dir):
        """Load the trained model and scaler."""
        # Load scaler
        scaler = joblib.load(f"{artifacts_dir}/scaler.pkl")
        
        # Load model
        model = TradingModel(input_size=35, hidden_size=256, dropout=0.3)
        model.load_state_dict(torch.load(f"{artifacts_dir}/trading_model.pth"))
        model.eval()
        
        return model, scaler
    
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
    
    def run_backtest(self, df, model, scaler):
        """Run backtest with single trade constraint."""
        print("Running Single Trade Backtest...")
        print(f"Initial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"Position Size: ${self.position_size:.2f}")
        print(f"Fee Rate: {self.fee_rate:.4f}")
        print(f"Probability Threshold: {self.threshold}")
        print("=" * 60)
        
        # Prepare features
        X = self.prepare_features(df)
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
        bankroll_history = [bankroll]
        trade_count = 0
        
        # Process each bar
        for i in range(len(df)):
            current_bar = df.iloc[i]
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
            
            # Record bankroll history
            bankroll_history.append(bankroll)
        
        return trades, bankroll_history
    
    def analyze_results(self, trades, bankroll_history):
        """Analyze and display backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        if not trades:
            print("No trades executed!")
            return
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        total_pnl = trades_df['net_pnl'].sum()
        avg_pnl = trades_df['net_pnl'].mean()
        best_trade = trades_df['net_pnl'].max()
        worst_trade = trades_df['net_pnl'].min()
        
        # Return statistics
        total_return = (bankroll_history[-1] - self.initial_bankroll) / self.initial_bankroll
        avg_return_per_trade = trades_df['price_return'].mean()
        
        # Fees
        total_fees = trades_df['fees'].sum()
        
        print(f"Initial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"Final Bankroll: ${bankroll_history[-1]:.2f}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Total Fees Paid: ${total_fees:.2f}")
        print()
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print()
        print(f"Average PnL per Trade: ${avg_pnl:.2f}")
        print(f"Best Trade: ${best_trade:.2f}")
        print(f"Worst Trade: ${worst_trade:.2f}")
        print(f"Average Return per Trade: {avg_return_per_trade:.2%}")
        
        # Show individual trades
        print("\n" + "=" * 60)
        print("INDIVIDUAL TRADES")
        print("=" * 60)
        for _, trade in trades_df.iterrows():
            print(f"Trade #{trade['trade_number']}: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')} -> {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}")
            print(f"  ${trade['entry_price']:.2f} -> ${trade['exit_price']:.2f} | Return: {trade['price_return']:.2%} | PnL: ${trade['net_pnl']:.2f}")
        
        return trades_df, bankroll_history
    
    def plot_results(self, trades_df, bankroll_history):
        """Plot backtest results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Bankroll progression
        ax1.plot(bankroll_history, linewidth=2, color='blue')
        ax1.axhline(y=self.initial_bankroll, color='red', linestyle='--', alpha=0.7, label='Initial Bankroll')
        ax1.set_title('Bankroll Progression', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (Bars)')
        ax1.set_ylabel('Bankroll (USDC)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add trade markers
        if not trades_df.empty:
            for _, trade in trades_df.iterrows():
                ax1.axvline(x=trade['trade_number'], color='green', alpha=0.3, linestyle=':')
        
        # PnL per trade
        if not trades_df.empty:
            colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['net_pnl']]
            ax2.bar(range(len(trades_df)), trades_df['net_pnl'], color=colors, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_title('PnL per Trade', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('PnL (USDC)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('single_trade_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main backtest function."""
    print("Single Trade Backtest - 1-Hour Trading Model")
    print("=" * 60)
    print("Constraints:")
    print("- Maximum 1 trade at a time")
    print("- 25 USDC position size")
    print("- 26 USDC initial bankroll")
    print("=" * 60)
    
    # Load configuration
    with open('config_1h_trading.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load backtest data
    backtest_df = pd.read_csv('artifacts/backtest_results.csv')
    print(f"Loaded {len(backtest_df)} bars for backtesting")
    
    # Initialize backtester
    backtester = SingleTradeBacktester(config)
    
    # Load model
    model, scaler = backtester.load_model(config['storage']['artifacts_dir'])
    
    # Run backtest
    trades, bankroll_history = backtester.run_backtest(backtest_df, model, scaler)
    
    # Analyze results
    trades_df, bankroll_history = backtester.analyze_results(trades, bankroll_history)
    
    # Plot results
    backtester.plot_results(trades_df, bankroll_history)
    
    # Save results
    if not trades_df.empty:
        trades_df.to_csv('single_trade_backtest_trades.csv', index=False)
        print(f"\nTrade details saved to: single_trade_backtest_trades.csv")
    
    print(f"Backtest chart saved to: single_trade_backtest_results.png")

if __name__ == "__main__":
    main()
