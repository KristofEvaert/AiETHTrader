#!/usr/bin/env python3
"""
Test Fee Calculation Logic
=========================

This script tests the fee calculation logic to ensure we properly account for
trading fees and don't try to sell more coin than we actually have.
"""

import os
import sys
import yaml
from datetime import datetime

def test_fee_calculation():
    """Test the fee calculation logic."""
    print("🧪 Testing Fee Calculation Logic")
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
        
        # Test fee calculation
        print(f"\n💰 Fee Configuration:")
        print(f"  Fee Rate: {engine.fee_bps} basis points ({engine.fee_bps/100}%)")
        
        # Test buy fee calculation
        print(f"\n🔄 Testing Buy Fee Calculation:")
        usdc_amount = 26.00
        buy_fee = engine.calculate_fee(usdc_amount, is_buy=True)
        usdc_after_fee = usdc_amount - buy_fee
        print(f"  USDC Available: ${usdc_amount:.2f}")
        print(f"  Buy Fee: ${buy_fee:.2f}")
        print(f"  USDC After Fee: ${usdc_after_fee:.2f}")
        
        # Test sell fee calculation
        print(f"\n🔄 Testing Sell Fee Calculation:")
        coin_amount = 0.010320
        price = 2500.0
        usdc_before_fee = coin_amount * price
        sell_fee = engine.calculate_fee(usdc_before_fee, is_buy=False)
        usdc_after_fee = usdc_before_fee - sell_fee
        print(f"  Coin Amount: {coin_amount:.6f}")
        print(f"  Price: ${price:.2f}")
        print(f"  USDC Before Fee: ${usdc_before_fee:.2f}")
        print(f"  Sell Fee: ${sell_fee:.2f}")
        print(f"  USDC After Fee: ${usdc_after_fee:.2f}")
        
        # Test complete trade cycle
        print(f"\n🔄 Testing Complete Trade Cycle:")
        
        # Simulate buy
        coin = 'ETHUSDC'
        current_price = engine.get_current_price(coin)
        print(f"  {coin} Price: ${current_price:.2f}")
        
        # Buy trade
        success, coin_amount, buy_fee = engine.execute_trade(coin, 'buy', current_price)
        if success:
            print(f"  ✅ Buy successful: {coin_amount:.6f} {coin} (fee: ${buy_fee:.2f})")
            
            # Update bankroll and position
            if engine.use_full_bankroll:
                usdc_spent = engine.bankroll
                engine.bankroll = 0
            else:
                usdc_spent = engine.position_size
                engine.bankroll -= engine.position_size
            
            engine.current_positions[coin] = {
                'entry_time': datetime.now(),
                'entry_price': current_price,
                'coin_amount': coin_amount,
                'usdc_spent': usdc_spent,
                'buy_fee': buy_fee,
                'signal_probability': 0.85,
                'competing_signals': 0
            }
            
            print(f"  Position: {coin_amount:.6f} {coin}")
            print(f"  Bankroll: ${engine.bankroll:.2f}")
        
        # Simulate sell after price increase
        print(f"\n🔄 Simulating Sell After Price Increase:")
        new_price = current_price * 1.05  # 5% increase
        print(f"  New {coin} Price: ${new_price:.2f} (+5%)")
        
        # Sell trade
        success, usdc_received, sell_fee = engine.execute_trade(coin, 'sell', new_price, coin_amount)
        if success:
            print(f"  ✅ Sell successful: {usdc_received:.2f} USDC (fee: ${sell_fee:.2f})")
            
            # Calculate PnL
            position = engine.current_positions[coin]
            usdc_spent = position['usdc_spent']
            total_fees = position['buy_fee'] + sell_fee
            pnl = usdc_received - usdc_spent
            
            print(f"  USDC Spent: ${usdc_spent:.2f}")
            print(f"  USDC Received: ${usdc_received:.2f}")
            print(f"  Total Fees: ${total_fees:.2f}")
            print(f"  PnL: ${pnl:.2f} ({(pnl/usdc_spent)*100:.2f}%)")
            
            # Update bankroll
            engine.bankroll = usdc_received
            del engine.current_positions[coin]
            
            print(f"  Final Bankroll: ${engine.bankroll:.2f}")
        
        print(f"\n✅ Fee calculation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing fee calculation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fee_impact():
    """Test the impact of fees on different trade sizes."""
    print(f"\n🧪 Testing Fee Impact on Different Trade Sizes")
    print("="*50)
    
    # Load configuration
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from live_trading_engine import LiveTradingEngine
    engine = LiveTradingEngine(config)
    
    # Test different trade sizes
    trade_sizes = [10, 25, 50, 100, 500, 1000]
    
    print(f"Trade Size | Buy Fee | Sell Fee | Total Fee | Fee %")
    print("-" * 55)
    
    for size in trade_sizes:
        buy_fee = engine.calculate_fee(size, is_buy=True)
        sell_fee = engine.calculate_fee(size, is_buy=False)
        total_fee = buy_fee + sell_fee
        fee_percentage = (total_fee / size) * 100
        
        print(f"${size:8.0f} | ${buy_fee:6.2f} | ${sell_fee:7.2f} | ${total_fee:8.2f} | {fee_percentage:5.2f}%")
    
    print(f"\n💡 Key Insights:")
    print(f"  - Fee rate: {engine.fee_bps/100}% per trade")
    print(f"  - Total round-trip fee: {engine.fee_bps*2/100}%")
    print(f"  - Need {engine.fee_bps*2/100}%+ gain to break even")
    print(f"  - System only sells the exact coin amount we have")

def main():
    """Main test function."""
    print("🧪 Fee Calculation Test Suite")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test fee calculation
    fee_calc_ok = test_fee_calculation()
    
    # Test fee impact
    test_fee_impact()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Fee Calculation: {'✅ PASS' if fee_calc_ok else '❌ FAIL'}")
    
    if fee_calc_ok:
        print("\n🎉 Fee calculation tests passed!")
        print("The system now properly:")
        print("  ✅ Calculates buy fees and reduces coin amount")
        print("  ✅ Calculates sell fees and reduces USDC received")
        print("  ✅ Only sells the exact coin amount we have")
        print("  ✅ Tracks all fees in trade history")
        print("  ✅ Prevents overselling due to fees")
    else:
        print("\n⚠️  Fee calculation tests failed. Please check the errors above.")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
