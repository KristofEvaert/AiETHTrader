#!/usr/bin/env python3
"""
Probability Threshold Analysis
Analyzes different threshold levels and their impact on trading performance
"""

import pandas as pd
import json
import numpy as np

def analyze_thresholds():
    """Analyze different probability thresholds"""
    
    print("🎯 PROBABILITY THRESHOLD ANALYSIS")
    print("=" * 60)
    
    # Load backtest data
    df = pd.read_csv('artifacts/backtest_curve.csv')
    df['ts'] = pd.to_datetime(df['ts'])
    
    print(f"📊 PROBABILITY RANGE OVERVIEW")
    print(f"Min Probability: {df['prob'].min():.3f}")
    print(f"Max Probability: {df['prob'].max():.3f}")
    print(f"Mean Probability: {df['prob'].mean():.3f}")
    print(f"Standard Deviation: {df['prob'].std():.3f}")
    
    # Test different thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    print(f"\n📈 THRESHOLD IMPACT ANALYSIS")
    print(f"{'Threshold':<12} {'Signals':<8} {'Trades':<8} {'Win Rate':<10} {'Avg Return':<12}")
    print("-" * 60)
    
    results = []
    
    for threshold in thresholds:
        # Filter signals above threshold
        signals = df[df['prob'] > threshold]
        trades = signals[signals['enter'] == 1]
        
        if len(trades) > 0:
            winning_trades = trades[trades['trade_ret'] > 0]
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            avg_return = trades['trade_ret'].mean() if len(trades) > 0 else 0
            
            print(f"{threshold:<12.2f} {len(signals):<8,} {len(trades):<8,} {win_rate:<10.2%} {avg_return:<12.3%}")
            
            results.append({
                'threshold': threshold,
                'signals': len(signals),
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return
            })
        else:
            print(f"{threshold:<12.2f} {len(signals):<8,} {len(trades):<8,} {'N/A':<10} {'N/A':<12}")
    
    # Find optimal threshold
    if results:
        best_win_rate = max(results, key=lambda x: x['win_rate'])
        best_return = max(results, key=lambda x: x['avg_return'])
        
        print(f"\n🏆 OPTIMAL THRESHOLDS")
        print(f"Best Win Rate: {best_win_rate['threshold']:.2f} (Win Rate: {best_win_rate['win_rate']:.2%})")
        print(f"Best Avg Return: {best_return['threshold']:.2f} (Return: {best_return['avg_return']:.3%})")
    
    # Current threshold analysis
    current_threshold = 0.55
    current_signals = df[df['prob'] > current_threshold]
    current_trades = current_signals[current_signals['enter'] == 1]
    
    print(f"\n🎯 CURRENT THRESHOLD ANALYSIS (0.55)")
    print(f"Total Signals: {len(current_signals):,}")
    print(f"Total Trades: {len(current_trades):,}")
    print(f"Signal to Trade Ratio: {len(current_trades)/len(current_signals):.2%}")
    
    if len(current_trades) > 0:
        current_winning = current_trades[current_trades['trade_ret'] > 0]
        current_win_rate = len(current_winning) / len(current_trades)
        current_avg_return = current_trades['trade_ret'].mean()
        
        print(f"Current Win Rate: {current_win_rate:.2%}")
        print(f"Current Avg Return: {current_avg_return:.3%}")
    
    # Risk vs Reward analysis
    print(f"\n⚖️ RISK VS REWARD ANALYSIS")
    print(f"Lower thresholds (0.50-0.60): More trades, lower win rate, higher volume")
    print(f"Medium thresholds (0.60-0.75): Balanced approach, moderate win rate")
    print(f"Higher thresholds (0.75+): Fewer trades, higher win rate, lower volume")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print(f"Conservative: 0.70+ (fewer trades, higher quality)")
    print(f"Balanced: 0.60-0.70 (good balance of quantity/quality)")
    print(f"Aggressive: 0.50-0.60 (more trades, lower quality)")
    print(f"Current (0.55): Good balance for 25 USDC setup")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_thresholds()
        print(f"\n✅ Analysis complete! Use this to optimize your trading threshold.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
