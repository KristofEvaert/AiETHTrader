#!/usr/bin/env python3
"""
Analyze Simultaneous Buy Signals
================================

This script analyzes the backtest data to identify simultaneous buy signals
between different cryptocurrencies and their potential impact on portfolio management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_simultaneous_signals():
    """Analyze simultaneous buy signals across all coins."""
    
    # Load the trade data
    df = pd.read_csv('all_coin_models_backtest_all_trades.csv')
    
    print("SIMULTANEOUS BUY SIGNALS ANALYSIS")
    print("=" * 60)
    print(f"Total trades across all coins: {len(df)}")
    print(f"Coins analyzed: {df['coin'].unique()}")
    print()
    
    # Convert entry times to datetime
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Sort by entry time
    df = df.sort_values('entry_time').reset_index(drop=True)
    
    # Find simultaneous signals (trades starting within the same hour)
    df['entry_hour'] = df['entry_time'].dt.floor('H')
    
    # Group by entry hour and count unique coins
    simultaneous_signals = df.groupby('entry_hour').agg({
        'coin': ['count', 'nunique', list],
        'entry_time': 'first',
        'exit_time': 'last'
    }).reset_index()
    
    # Flatten column names
    simultaneous_signals.columns = ['entry_hour', 'total_trades', 'unique_coins', 'coins_list', 'first_entry', 'last_exit']
    
    # Filter for hours with multiple trades
    multiple_trades = simultaneous_signals[simultaneous_signals['total_trades'] > 1]
    
    print("HOURS WITH MULTIPLE TRADES:")
    print("-" * 40)
    
    if len(multiple_trades) > 0:
        for _, row in multiple_trades.iterrows():
            print(f"Time: {row['entry_hour']}")
            print(f"  Trades: {row['total_trades']}")
            print(f"  Coins: {row['coins_list']}")
            print(f"  Duration: {row['first_entry']} to {row['last_exit']}")
            print()
        
        print(f"Total hours with multiple trades: {len(multiple_trades)}")
        print(f"Percentage of trading hours with conflicts: {len(multiple_trades) / len(simultaneous_signals) * 100:.1f}%")
    else:
        print("No simultaneous trades found!")
    
    # Analyze trade overlap patterns
    print("\nTRADE OVERLAP ANALYSIS:")
    print("-" * 30)
    
    # Create a detailed analysis of overlapping trades
    overlaps = []
    
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            trade1 = df.iloc[i]
            trade2 = df.iloc[j]
            
            # Check if trades overlap in time
            if (trade1['entry_time'] < trade2['exit_time'] and 
                trade2['entry_time'] < trade1['exit_time']):
                
                overlap_start = max(trade1['entry_time'], trade2['entry_time'])
                overlap_end = min(trade1['exit_time'], trade2['exit_time'])
                overlap_duration = overlap_end - overlap_start
                
                overlaps.append({
                    'coin1': trade1['coin'],
                    'coin2': trade2['coin'],
                    'trade1_entry': trade1['entry_time'],
                    'trade1_exit': trade1['exit_time'],
                    'trade2_entry': trade2['entry_time'],
                    'trade2_exit': trade2['exit_time'],
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration_minutes': overlap_duration.total_seconds() / 60,
                    'trade1_return': trade1['price_return'],
                    'trade2_return': trade2['price_return']
                })
    
    if overlaps:
        overlaps_df = pd.DataFrame(overlaps)
        
        print(f"Total overlapping trades: {len(overlaps)}")
        print(f"Percentage of trades with overlaps: {len(overlaps) / len(df) * 100:.1f}%")
        
        # Analyze overlap patterns by coin pairs
        print("\nOVERLAP PATTERNS BY COIN PAIRS:")
        print("-" * 35)
        
        coin_pairs = overlaps_df.groupby(['coin1', 'coin2']).size().reset_index(name='overlap_count')
        coin_pairs = coin_pairs.sort_values('overlap_count', ascending=False)
        
        for _, row in coin_pairs.iterrows():
            print(f"{row['coin1']} <-> {row['coin2']}: {row['overlap_count']} overlaps")
        
        # Analyze overlap duration
        print(f"\nOVERLAP DURATION STATISTICS:")
        print(f"Average overlap: {overlaps_df['overlap_duration_minutes'].mean():.1f} minutes")
        print(f"Maximum overlap: {overlaps_df['overlap_duration_minutes'].max():.1f} minutes")
        print(f"Minimum overlap: {overlaps_df['overlap_duration_minutes'].min():.1f} minutes")
        
        # Show detailed overlap examples
        print(f"\nDETAILED OVERLAP EXAMPLES (Top 10):")
        print("-" * 40)
        
        top_overlaps = overlaps_df.nlargest(10, 'overlap_duration_minutes')
        for _, row in top_overlaps.iterrows():
            print(f"{row['coin1']} ({row['trade1_entry'].strftime('%Y-%m-%d %H:%M')} - {row['trade1_exit'].strftime('%Y-%m-%d %H:%M')})")
            print(f"  vs {row['coin2']} ({row['trade2_entry'].strftime('%Y-%m-%d %H:%M')} - {row['trade2_exit'].strftime('%Y-%m-%d %H:%M')})")
            print(f"  Overlap: {row['overlap_duration_minutes']:.1f} minutes")
            print(f"  Returns: {row['coin1']} {row['trade1_return']:.2%}, {row['coin2']} {row['trade2_return']:.2%}")
            print()
        
        return overlaps_df, multiple_trades
    else:
        print("No overlapping trades found!")
        return pd.DataFrame(), multiple_trades

def analyze_portfolio_impact(overlaps_df):
    """Analyze the impact of simultaneous signals on portfolio management."""
    
    if overlaps_df.empty:
        print("\nPORTFOLIO IMPACT ANALYSIS:")
        print("-" * 30)
        print("No overlapping trades to analyze.")
        return
    
    print("\nPORTFOLIO IMPACT ANALYSIS:")
    print("-" * 30)
    
    # Calculate potential missed opportunities
    total_overlaps = len(overlaps_df)
    unique_trades_with_overlaps = len(set(overlaps_df['coin1'].tolist() + overlaps_df['coin2'].tolist()))
    
    print(f"Total overlapping trade pairs: {total_overlaps}")
    print(f"Unique trades involved in overlaps: {unique_trades_with_overlaps}")
    
    # Analyze return correlation during overlaps
    overlap_returns = overlaps_df[['trade1_return', 'trade2_return']].corr()
    print(f"\nReturn correlation during overlaps: {overlap_returns.iloc[0, 1]:.3f}")
    
    # Calculate potential portfolio impact
    # If we could only take one trade at a time, what would be the impact?
    print(f"\nSINGLE TRADE CONSTRAINT IMPACT:")
    print("-" * 35)
    
    # For each overlap, calculate which trade had better returns
    overlaps_df['better_return'] = overlaps_df.apply(
        lambda row: row['coin1'] if row['trade1_return'] > row['trade2_return'] else row['coin2'], axis=1
    )
    overlaps_df['better_return_value'] = overlaps_df.apply(
        lambda row: max(row['trade1_return'], row['trade2_return']), axis=1
    )
    overlaps_df['worse_return_value'] = overlaps_df.apply(
        lambda row: min(row['trade1_return'], row['trade2_return']), axis=1
    )
    
    # Calculate opportunity cost
    opportunity_cost = (overlaps_df['better_return_value'] - overlaps_df['worse_return_value']).sum()
    print(f"Total opportunity cost from overlaps: {opportunity_cost:.2%}")
    print(f"Average opportunity cost per overlap: {opportunity_cost / len(overlaps_df):.2%}")
    
    # Show which coin would have been better in each overlap
    better_coin_counts = overlaps_df['better_return'].value_counts()
    print(f"\nBetter performing coin in overlaps:")
    for coin, count in better_coin_counts.items():
        print(f"  {coin}: {count} times")

def create_visualization(overlaps_df, multiple_trades):
    """Create visualizations of the simultaneous signals."""
    
    if overlaps_df.empty and len(multiple_trades) == 0:
        print("\nNo overlapping data to visualize.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Timeline of simultaneous signals
    if len(multiple_trades) > 0:
        ax1 = axes[0, 0]
        multiple_trades['entry_hour'] = pd.to_datetime(multiple_trades['entry_hour'])
        ax1.scatter(multiple_trades['entry_hour'], multiple_trades['total_trades'], alpha=0.7)
        ax1.set_title('Simultaneous Signals Over Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of Simultaneous Trades')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Overlap duration distribution
    if not overlaps_df.empty:
        ax2 = axes[0, 1]
        ax2.hist(overlaps_df['overlap_duration_minutes'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribution of Overlap Durations')
        ax2.set_xlabel('Overlap Duration (minutes)')
        ax2.set_ylabel('Frequency')
    
    # 3. Coin pair overlap heatmap
    if not overlaps_df.empty:
        ax3 = axes[1, 0]
        coin_pairs = overlaps_df.groupby(['coin1', 'coin2']).size().unstack(fill_value=0)
        sns.heatmap(coin_pairs, annot=True, fmt='d', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Overlap Count by Coin Pairs')
    
    # 4. Return correlation during overlaps
    if not overlaps_df.empty:
        ax4 = axes[1, 1]
        ax4.scatter(overlaps_df['trade1_return'], overlaps_df['trade2_return'], alpha=0.6)
        ax4.set_title('Return Correlation During Overlaps')
        ax4.set_xlabel('Trade 1 Return')
        ax4.set_ylabel('Trade 2 Return')
        
        # Add correlation line
        z = np.polyfit(overlaps_df['trade1_return'], overlaps_df['trade2_return'], 1)
        p = np.poly1d(z)
        ax4.plot(overlaps_df['trade1_return'], p(overlaps_df['trade1_return']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('simultaneous_signals_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as: simultaneous_signals_analysis.png")

def main():
    """Main analysis function."""
    print("Analyzing Simultaneous Buy Signals Across All Coins")
    print("=" * 60)
    
    # Load and analyze the data
    overlaps_df, multiple_trades = analyze_simultaneous_signals()
    
    # Analyze portfolio impact
    analyze_portfolio_impact(overlaps_df)
    
    # Create visualizations
    create_visualization(overlaps_df, multiple_trades)
    
    # Save detailed results
    if not overlaps_df.empty:
        overlaps_df.to_csv('simultaneous_signals_detailed.csv', index=False)
        print(f"\nDetailed overlap data saved to: simultaneous_signals_detailed.csv")
    
    if len(multiple_trades) > 0:
        multiple_trades.to_csv('simultaneous_signals_summary.csv', index=False)
        print(f"Simultaneous signals summary saved to: simultaneous_signals_summary.csv")

if __name__ == "__main__":
    main()
