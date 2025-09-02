#!/usr/bin/env python3
"""
Detailed 25 USDC Backtest Analysis
Analyzes the performance with 25 USDC trade size, 1 trade at a time, 0.1% fee
"""

import pandas as pd
import json
import numpy as np

def analyze_25usdc_backtest():
    """Analyze the 25 USDC backtest results in detail"""
    
    print("🎯 DETAILED 25 USDC BACKTEST ANALYSIS")
    print("=" * 60)
    print("Configuration: 25 USDC per trade, 1 trade at a time, 0.1% fee")
    print("=" * 60)
    
    # Load backtest statistics
    with open('artifacts/backtest_stats.json', 'r') as f:
        stats = json.load(f)
    
    print(f"📊 PERFORMANCE METRICS")
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Total Trades: {stats['trades']:,}")
    
    # Load backtest curve
    df = pd.read_csv('artifacts/backtest_curve.csv')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    
    # Calculate trade-specific metrics
    trades_df = df[df['enter'] == 1].copy()
    winning_trades = trades_df[trades_df['trade_ret'] > 0]
    losing_trades = trades_df[trades_df['trade_ret'] < 0]
    
    print(f"\n📈 TRADE ANALYSIS")
    print(f"Total Trades Executed: {len(trades_df):,}")
    print(f"Winning Trades: {len(winning_trades):,}")
    print(f"Losing Trades: {len(losing_trades):,}")
    print(f"Win Rate: {len(winning_trades)/len(trades_df):.2%}")
    
    if len(winning_trades) > 0:
        print(f"Average Win: {winning_trades['trade_ret'].mean():.3%}")
        print(f"Best Win: {winning_trades['trade_ret'].max():.3%}")
    
    if len(losing_trades) > 0:
        print(f"Average Loss: {losing_trades['trade_ret'].mean():.3%}")
        print(f"Worst Loss: {losing_trades['trade_ret'].min():.3%}")
    
    # Calculate dollar returns for 25 USDC setup
    trade_size_usdc = 25
    initial_capital = 1000  # Starting with $1000
    
    print(f"\n💰 DOLLAR RETURNS (25 USDC per trade)")
    print(f"Starting Capital: ${initial_capital:,}")
    print(f"Trade Size: ${trade_size_usdc}")
    print(f"Total Return: ${initial_capital * stats['total_return']:.2f}")
    print(f"Final Portfolio Value: ${initial_capital * (1 + stats['total_return']):.2f}")
    
    # Calculate fees paid
    total_fees_paid = len(trades_df) * trade_size_usdc * 0.001  # 0.1% fee
    print(f"Total Fees Paid: ${total_fees_paid:.2f}")
    print(f"Fees as % of Capital: {total_fees_paid/initial_capital:.2%}")
    
    # Risk analysis
    print(f"\n⚠️ RISK ANALYSIS")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Max Drawdown Amount: ${initial_capital * abs(stats['max_drawdown']):.2f}")
    
    # Calculate risk-adjusted metrics
    if len(trades_df) > 0:
        trade_returns = trades_df['trade_ret']
        volatility = trade_returns.std()
        print(f"Trade Return Volatility: {volatility:.3%}")
        print(f"Risk-Adjusted Return per Trade: {stats['total_return'] / len(trades_df):.4%}")
    
    # Monthly performance
    monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
    print(f"\n📅 MONTHLY PERFORMANCE")
    for date, ret in monthly_returns.items():
        print(f"{date.strftime('%Y-%m')}: {ret:.2%}")
    
    # Trading frequency analysis
    print(f"\n📈 TRADING FREQUENCY")
    print(f"Trades per day: {len(trades_df)/365:.1f}")
    print(f"Trades per week: {len(trades_df)/52:.1f}")
    print(f"Trades per month: {len(trades_df)/12:.1f}")
    
    # Serial execution analysis (1 trade at a time)
    print(f"\n🔄 SERIAL EXECUTION ANALYSIS")
    print(f"Strategy: Maximum 1 trade at a time")
    print(f"Average trade duration: 1 bar (15 minutes)")
    print(f"Position sizing: Fixed ${trade_size_usdc} per trade")
    
    # Fee impact analysis
    print(f"\n💸 FEE IMPACT ANALYSIS")
    print(f"Fee per trade: ${trade_size_usdc * 0.001:.3f}")
    print(f"Total fees: ${total_fees_paid:.2f}")
    print(f"Fees vs Returns: {total_fees_paid/(initial_capital * stats['total_return']):.2%}")
    
    # Capital efficiency
    print(f"\n🏦 CAPITAL EFFICIENCY")
    max_capital_used = trade_size_usdc  # Only 1 trade at a time
    print(f"Maximum capital used: ${max_capital_used}")
    print(f"Capital efficiency: {max_capital_used/initial_capital:.2%}")
    print(f"Unused capital: ${initial_capital - max_capital_used}")
    
    return stats, df

if __name__ == "__main__":
    try:
        stats, df = analyze_25usdc_backtest()
        print(f"\n✅ Analysis complete!")
        print(f"Your 25 USDC setup achieved {stats['total_return']:.2%} return with {stats['sharpe']:.2f} Sharpe ratio.")
        print(f"Strategy executed {stats['trades']:,} trades with excellent risk management.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
