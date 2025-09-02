#!/usr/bin/env python3
"""
Backtest Comparison Script
Compares the 50 USDC vs 25 USDC trading setups
"""

import json
import pandas as pd

def load_backtest_stats(filepath):
    """Load backtest statistics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def compare_backtests():
    """Compare the two backtest configurations"""
    
    print("🔄 BACKTEST CONFIGURATION COMPARISON")
    print("=" * 60)
    
    # Load both backtest results
    config_50 = {
        'name': '50 USDC Setup',
        'trade_size': 50,
        'stats': load_backtest_stats('artifacts/backtest_stats.json')
    }
    
    config_25 = {
        'name': '25 USDC Setup', 
        'trade_size': 25,
        'stats': load_backtest_stats('artifacts/backtest_stats.json')
    }
    
    # Display comparison
    print(f"{'Metric':<20} {'50 USDC':<15} {'25 USDC':<15} {'Change':<15}")
    print("-" * 60)
    
    if config_50['stats'] and config_25['stats']:
        # Total Return
        ret_50 = config_50['stats']['total_return']
        ret_25 = config_25['stats']['total_return']
        ret_change = ((ret_25 - ret_50) / ret_50) * 100
        print(f"{'Total Return':<20} {ret_50:>12.2%} {ret_25:>12.2%} {ret_change:>+12.1f}%")
        
        # Sharpe Ratio
        sharpe_50 = config_50['stats']['sharpe']
        sharpe_25 = config_25['stats']['sharpe']
        sharpe_change = ((sharpe_25 - sharpe_50) / sharpe_50) * 100
        print(f"{'Sharpe Ratio':<20} {sharpe_50:>12.2f} {sharpe_25:>12.2f} {sharpe_change:>+12.1f}%")
        
        # Max Drawdown
        dd_50 = config_50['stats']['max_drawdown']
        dd_25 = config_25['stats']['max_drawdown']
        dd_change = ((dd_25 - dd_50) / abs(dd_50)) * 100
        print(f"{'Max Drawdown':<20} {dd_50:>12.2%} {dd_25:>12.2%} {dd_change:>+12.1f}%")
        
        # Total Trades
        trades_50 = config_50['stats']['trades']
        trades_25 = config_25['stats']['trades']
        trades_change = ((trades_25 - trades_50) / trades_50) * 100
        print(f"{'Total Trades':<20} {trades_50:>12,} {trades_25:>12,} {trades_change:>+12.1f}%")
        
        # Risk-Adjusted Return per Trade
        risk_adj_50 = ret_50 / abs(dd_50) if dd_50 != 0 else 0
        risk_adj_25 = ret_25 / abs(dd_25) if dd_25 != 0 else 0
        print(f"{'Risk-Adj Return':<20} {risk_adj_50:>12.2f} {risk_adj_25:>12.2f}")
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        if ret_25 > ret_50:
            print("✅ 25 USDC setup shows BETTER total returns")
        else:
            print("⚠️  25 USDC setup shows LOWER total returns")
            
        if sharpe_25 > sharpe_50:
            print("✅ 25 USDC setup shows BETTER risk-adjusted returns")
        else:
            print("⚠️  25 USDC setup shows LOWER risk-adjusted returns")
            
        if abs(dd_25) < abs(dd_50):
            print("✅ 25 USDC setup shows LOWER risk (smaller drawdown)")
        else:
            print("⚠️  25 USDC setup shows HIGHER risk (larger drawdown)")
            
        # Calculate absolute dollar returns
        initial_capital = 1000  # Assume $1000 starting capital for comparison
        dollar_return_50 = initial_capital * ret_50
        dollar_return_25 = initial_capital * ret_25
        
        print(f"\n💰 DOLLAR RETURNS (Starting with ${initial_capital:,})")
        print(f"50 USDC setup: ${dollar_return_50:.2f}")
        print(f"25 USDC setup: ${dollar_return_25:.2f}")
        print(f"Difference: ${dollar_return_25 - dollar_return_50:+.2f}")
        
        # Trading frequency analysis
        print(f"\n📈 TRADING FREQUENCY")
        print(f"50 USDC: {trades_50:,} trades over ~1 year")
        print(f"25 USDC: {trades_25:,} trades over ~1 year")
        print(f"Trades per day (50 USDC): {trades_50/365:.1f}")
        print(f"Trades per day (25 USDC): {trades_25/365:.1f}")
        
    else:
        print("❌ Could not load backtest statistics for comparison")

if __name__ == "__main__":
    compare_backtests()
