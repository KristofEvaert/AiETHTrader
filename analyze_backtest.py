#!/usr/bin/env python3
"""
Backtest Analysis Script for AiETHTrader
Analyzes the performance metrics and creates visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def analyze_backtest():
    """Analyze the backtest results"""
    
    # Load backtest statistics
    with open('artifacts/backtest_stats.json', 'r') as f:
        stats = json.load(f)
    
    print("🎯 BACKTEST PERFORMANCE ANALYSIS")
    print("=" * 50)
    print(f"Total Return: {stats['total_return']:.2%}")
    print(f"Sharpe Ratio: {stats['sharpe']:.2f}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Total Trades: {stats['trades']:,}")
    
    # Load backtest curve
    df = pd.read_csv('artifacts/backtest_curve.csv')
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index('ts')
    
    # Calculate additional metrics
    winning_trades = df[df['trade_ret'] > 0]
    losing_trades = df[df['trade_ret'] < 0]
    
    win_rate = len(winning_trades) / len(df[df['enter'] == 1]) if len(df[df['enter'] == 1]) > 0 else 0
    avg_win = winning_trades['trade_ret'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['trade_ret'].mean() if len(losing_trades) > 0 else 0
    
    print(f"\n📊 TRADE ANALYSIS")
    print("=" * 50)
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2%}")
    print(f"Average Loss: {avg_loss:.2%}")
    print(f"Profit Factor: {abs(avg_win / avg_loss):.2f}" if avg_loss != 0 else "Profit Factor: N/A")
    
    # Calculate drawdown
    df['peak'] = df['equity'].expanding().max()
    df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Equity curve
    ax1.plot(df.index, df['equity'], label='Portfolio Value', linewidth=2)
    ax1.set_title('Portfolio Equity Curve')
    ax1.set_ylabel('Portfolio Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Drawdown
    ax2.fill_between(df.index, df['drawdown'], 0, alpha=0.3, color='red')
    ax2.plot(df.index, df['drawdown'], color='red', linewidth=1)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.grid(True, alpha=0.3)
    
    # Trade returns distribution
    trade_returns = df[df['enter'] == 1]['trade_ret']
    if len(trade_returns) > 0:
        ax3.hist(trade_returns, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(trade_returns.mean(), color='red', linestyle='--', label=f'Mean: {trade_returns.mean():.3%}')
        ax3.set_title('Trade Returns Distribution')
        ax3.set_xlabel('Trade Return %')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/backtest_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Charts saved to: artifacts/backtest_analysis.png")
    
    # Monthly returns
    monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
    print(f"\n📅 MONTHLY PERFORMANCE")
    print("=" * 50)
    for date, ret in monthly_returns.items():
        print(f"{date.strftime('%Y-%m')}: {ret:.2%}")
    
    # Risk metrics
    print(f"\n⚠️ RISK METRICS")
    print("=" * 50)
    print(f"Volatility (Annualized): {trade_returns.std() * (252 * 24 * 4):.2%}" if len(trade_returns) > 0 else "Volatility: N/A")
    print(f"Best Trade: {trade_returns.max():.2%}" if len(trade_returns) > 0 else "Best Trade: N/A")
    print(f"Worst Trade: {trade_returns.min():.2%}" if len(trade_returns) > 0 else "Worst Trade: N/A")
    
    return stats, df

if __name__ == "__main__":
    try:
        stats, df = analyze_backtest()
        print(f"\n✅ Analysis complete! Your AI trader achieved {stats['total_return']:.2%} return with {stats['sharpe']:.2f} Sharpe ratio.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
