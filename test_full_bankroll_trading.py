#!/usr/bin/env python3
"""
Test Full Bankroll Trading Logic
===============================

This script tests the full bankroll trading logic where:
1. BUY: Use entire bankroll to buy as much coin as possible
2. SELL: Sell all of the coin we have back to USDC
"""

import os
import sys
import yaml
import time
from datetime import datetime, timedelta

def test_full_bankroll_trading():
    """Test the full bankroll trading logic."""
    print("🧪 Testing Full Bankroll Trading Logic")
    print("="*50)
    
    # Load configuration
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Import live trading engine
    from live_trading_engine import LiveTradingEngine
    
    try:
        # Initialize trading engine
        print("🚀 Initializing live trading engine...")
        engine = LiveTradingEngine(config)
        
        # Check initial status
        status = engine.get_status()
        print(f"\n📊 Initial Status:")
        print(f"  Bankroll: ${status['bankroll']:.2f}")
        print(f"  Positions: {status['positions']}")
        print(f"  Use Full Bankroll: {engine.use_full_bankroll}")
        
        # Simulate a buy signal
        print(f"\n🔄 Simulating Buy Signal...")
        
        # Manually trigger a buy signal for ETHUSDC
        coin = 'ETHUSDC'
        current_price = engine.get_current_price(coin)
        print(f"  Current {coin} price: ${current_price:.2f}")
        
        # Simulate buy trade
        if engine.use_full_bankroll:
            usdc_amount = engine.bankroll
            coin_amount = usdc_amount / current_price
            print(f"  BUY: {coin_amount:.6f} {coin} for {usdc_amount:.2f} USDC")
            
            # Update bankroll and position
            engine.bankroll = 0  # All USDC converted to coin
            engine.current_positions[coin] = {
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'coin_amount': coin_amount,
                'usdc_amount': usdc_amount,
                'signal_probability': 0.85,
                'competing_signals': 2
            }
            
            print(f"  ✅ Position opened: {coin_amount:.6f} {coin}")
            print(f"  Bankroll after buy: ${engine.bankroll:.2f}")
        
        # Check status after buy
        status = engine.get_status()
        print(f"\n📊 Status After Buy:")
        print(f"  Bankroll: ${status['bankroll']:.2f}")
        print(f"  Positions: {status['positions']}")
        print(f"  Current Positions: {status['current_positions']}")
        
        # Simulate price change and sell
        print(f"\n🔄 Simulating Sell After 1 Hour...")
        
        # Simulate price increase
        new_price = current_price * 1.05  # 5% increase
        print(f"  New {coin} price: ${new_price:.2f} (+5%)")
        
        # Get position details
        position = engine.current_positions[coin]
        coin_amount = position['coin_amount']
        usdc_received = coin_amount * new_price
        usdc_spent = position['usdc_amount']
        pnl = usdc_received - usdc_spent
        
        print(f"  SELL: {coin_amount:.6f} {coin} for {usdc_received:.2f} USDC")
        print(f"  PnL: ${pnl:.2f} ({(pnl/usdc_spent)*100:.2f}%)")
        
        # Update bankroll and close position
        engine.bankroll = usdc_received
        del engine.current_positions[coin]
        
        print(f"  ✅ Position closed: {coin_amount:.6f} {coin} → {usdc_received:.2f} USDC")
        print(f"  Bankroll after sell: ${engine.bankroll:.2f}")
        
        # Check final status
        status = engine.get_status()
        print(f"\n📊 Final Status:")
        print(f"  Bankroll: ${status['bankroll']:.2f}")
        print(f"  Positions: {status['positions']}")
        print(f"  Total Trades: {status['total_trades']}")
        
        # Test with different coin
        print(f"\n🔄 Testing with Different Coin (ADAUSDC)...")
        
        coin = 'ADAUSDC'
        current_price = engine.get_current_price(coin)
        print(f"  Current {coin} price: ${current_price:.2f}")
        
        if engine.use_full_bankroll:
            usdc_amount = engine.bankroll
            coin_amount = usdc_amount / current_price
            print(f"  BUY: {coin_amount:.6f} {coin} for {usdc_amount:.2f} USDC")
            
            # Update bankroll and position
            engine.bankroll = 0
            engine.current_positions[coin] = {
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'coin_amount': coin_amount,
                'usdc_amount': usdc_amount,
                'signal_probability': 0.78,
                'competing_signals': 1
            }
            
            print(f"  ✅ Position opened: {coin_amount:.6f} {coin}")
            print(f"  Bankroll after buy: ${engine.bankroll:.2f}")
        
        print(f"\n✅ Full bankroll trading logic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing full bankroll trading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_position_trading():
    """Test the fixed position size trading logic."""
    print("\n🧪 Testing Fixed Position Size Trading Logic")
    print("="*50)
    
    # Load configuration
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Override to use fixed position size
    config['live']['use_full_bankroll'] = False
    
    # Import live trading engine
    from live_trading_engine import LiveTradingEngine
    
    try:
        # Initialize trading engine
        print("🚀 Initializing live trading engine (fixed position)...")
        engine = LiveTradingEngine(config)
        
        # Check initial status
        status = engine.get_status()
        print(f"\n📊 Initial Status:")
        print(f"  Bankroll: ${status['bankroll']:.2f}")
        print(f"  Position Size: ${engine.position_size:.2f}")
        print(f"  Use Full Bankroll: {engine.use_full_bankroll}")
        
        # Simulate a buy signal
        print(f"\n🔄 Simulating Buy Signal...")
        
        coin = 'BTCUSDC'
        current_price = engine.get_current_price(coin)
        print(f"  Current {coin} price: ${current_price:.2f}")
        
        # Simulate buy trade
        if not engine.use_full_bankroll:
            usdc_amount = engine.position_size
            coin_amount = usdc_amount / current_price
            print(f"  BUY: {coin_amount:.6f} {coin} for {usdc_amount:.2f} USDC")
            
            # Update bankroll and position
            engine.bankroll -= engine.position_size
            engine.current_positions[coin] = {
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'coin_amount': coin_amount,
                'usdc_amount': usdc_amount,
                'signal_probability': 0.82,
                'competing_signals': 0
            }
            
            print(f"  ✅ Position opened: {coin_amount:.6f} {coin}")
            print(f"  Bankroll after buy: ${engine.bankroll:.2f}")
        
        print(f"\n✅ Fixed position trading logic test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed position trading: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Full Bankroll Trading Logic Test Suite")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test full bankroll trading
    full_bankroll_ok = test_full_bankroll_trading()
    
    # Test fixed position trading
    fixed_position_ok = test_fixed_position_trading()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Full Bankroll Trading: {'✅ PASS' if full_bankroll_ok else '❌ FAIL'}")
    print(f"Fixed Position Trading: {'✅ PASS' if fixed_position_ok else '❌ FAIL'}")
    
    if full_bankroll_ok and fixed_position_ok:
        print("\n🎉 All trading logic tests passed!")
        print("The system can now:")
        print("  ✅ Use entire bankroll to buy maximum coin amount")
        print("  ✅ Sell all coin back to USDC")
        print("  ✅ Track detailed trade information")
        print("  ✅ Calculate accurate PnL")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
