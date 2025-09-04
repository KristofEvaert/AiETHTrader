#!/usr/bin/env python3
"""
Multi-Signal Strategy for Live Trading
=====================================

This module implements the strategy for handling multiple simultaneous buy signals
during live trading. The strategy selects the signal with the highest probability
and logs all available signals for analysis.

Strategy:
1. Collect all buy signals from all coins
2. Sort by probability (highest first)
3. Select the signal with highest probability
4. Log all available signals for comparison
5. Track competing signals for analysis
"""

import logging
from typing import List, Dict, Tuple
from datetime import datetime

class MultiSignalStrategy:
    """Handles multiple simultaneous buy signals."""
    
    def __init__(self, logger=None):
        """Initialize the multi-signal strategy."""
        self.logger = logger or logging.getLogger('MultiSignalStrategy')
        self.signal_history = []
    
    def select_best_signal(self, buy_signals: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """
        Select the best signal from multiple buy signals.
        
        Args:
            buy_signals: List of buy signals with coin, probability, timestamp
            
        Returns:
            Tuple of (best_signal, all_signals_sorted)
        """
        if not buy_signals:
            return None, []
        
        # Sort by probability (highest first)
        sorted_signals = sorted(buy_signals, key=lambda x: x['probability'], reverse=True)
        
        best_signal = sorted_signals[0]
        
        # Log the selection
        self.logger.info(f"🎯 SELECTED BEST SIGNAL: {best_signal['coin']} with probability {best_signal['probability']:.3f}")
        
        # Log all available signals for comparison
        if len(sorted_signals) > 1:
            self.logger.info("📊 Available signals:")
            for i, signal in enumerate(sorted_signals):
                status = "✅ SELECTED" if i == 0 else "❌ SKIPPED"
                self.logger.info(f"  {i+1}. {signal['coin']}: {signal['probability']:.3f} {status}")
        
        # Record in history
        self.record_signal_selection(best_signal, sorted_signals)
        
        return best_signal, sorted_signals
    
    def record_signal_selection(self, selected_signal: Dict, all_signals: List[Dict]):
        """Record signal selection for analysis."""
        record = {
            'timestamp': datetime.now(),
            'selected_coin': selected_signal['coin'],
            'selected_probability': selected_signal['probability'],
            'total_signals': len(all_signals),
            'competing_signals': len(all_signals) - 1,
            'all_signals': all_signals.copy()
        }
        
        self.signal_history.append(record)
        
        # Keep only last 1000 records to prevent memory bloat
        if len(self.signal_history) > 1000:
            self.signal_history = self.signal_history[-1000:]
    
    def get_signal_statistics(self) -> Dict:
        """Get statistics about signal selections."""
        if not self.signal_history:
            return {}
        
        # Count how often each coin was selected
        coin_selections = {}
        total_selections = len(self.signal_history)
        
        for record in self.signal_history:
            coin = record['selected_coin']
            coin_selections[coin] = coin_selections.get(coin, 0) + 1
        
        # Calculate selection rates
        selection_rates = {}
        for coin, count in coin_selections.items():
            selection_rates[coin] = count / total_selections
        
        # Calculate average competing signals
        avg_competing = sum(record['competing_signals'] for record in self.signal_history) / total_selections
        
        return {
            'total_selections': total_selections,
            'coin_selections': coin_selections,
            'selection_rates': selection_rates,
            'average_competing_signals': avg_competing
        }
    
    def log_signal_statistics(self):
        """Log current signal statistics."""
        stats = self.get_signal_statistics()
        
        if not stats:
            self.logger.info("No signal selections recorded yet.")
            return
        
        self.logger.info("📊 Signal Selection Statistics:")
        self.logger.info(f"  Total selections: {stats['total_selections']}")
        self.logger.info(f"  Average competing signals: {stats['average_competing_signals']:.1f}")
        
        self.logger.info("  Coin selection rates:")
        for coin, rate in stats['selection_rates'].items():
            count = stats['coin_selections'][coin]
            self.logger.info(f"    {coin}: {count} times ({rate:.1%})")

def test_multi_signal_strategy():
    """Test the multi-signal strategy."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Test')
    
    # Create strategy
    strategy = MultiSignalStrategy(logger)
    
    # Test with multiple signals
    test_signals = [
        {'coin': 'ETHUSDC', 'probability': 0.75, 'timestamp': datetime.now()},
        {'coin': 'ADAUSDC', 'probability': 0.82, 'timestamp': datetime.now()},
        {'coin': 'BTCUSDC', 'probability': 0.68, 'timestamp': datetime.now()},
        {'coin': 'DOGEUSDC', 'probability': 0.79, 'timestamp': datetime.now()},
        {'coin': 'XRPUSDC', 'probability': 0.71, 'timestamp': datetime.now()}
    ]
    
    print("🧪 Testing Multi-Signal Strategy")
    print("="*50)
    
    # Test selection
    best_signal, all_signals = strategy.select_best_signal(test_signals)
    
    print(f"\nBest signal: {best_signal['coin']} ({best_signal['probability']:.3f})")
    print(f"Total signals: {len(all_signals)}")
    print(f"Competing signals: {len(all_signals) - 1}")
    
    # Test statistics
    strategy.log_signal_statistics()
    
    # Test with single signal
    print("\n" + "="*50)
    print("Testing with single signal:")
    
    single_signal = [{'coin': 'ETHUSDC', 'probability': 0.85, 'timestamp': datetime.now()}]
    best_signal, all_signals = strategy.select_best_signal(single_signal)
    
    print(f"Best signal: {best_signal['coin']} ({best_signal['probability']:.3f})")
    print(f"Total signals: {len(all_signals)}")
    print(f"Competing signals: {len(all_signals) - 1}")
    
    # Final statistics
    print("\n" + "="*50)
    print("Final Statistics:")
    strategy.log_signal_statistics()

if __name__ == "__main__":
    test_multi_signal_strategy()
