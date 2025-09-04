#!/usr/bin/env python3
"""
Test Multi-Coin Data Manager
============================

Test script to verify the multi-coin data manager works correctly.
"""

import os
import sys
import yaml
import time
from datetime import datetime

def test_multi_coin_data_manager():
    """Test the multi-coin data manager functionality."""
    print("🧪 Testing Multi-Coin Data Manager...")
    
    # Load configuration
    with open('config_multi_coin_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Import data manager
    from multi_coin_data_manager import LiveMultiCoinDataManager
    
    try:
        # Initialize data manager
        print("📊 Initializing multi-coin data manager...")
        data_manager = LiveMultiCoinDataManager(config)
        
        # Get available coins
        coins = data_manager.get_available_coins()
        print(f"📈 Available coins: {coins}")
        
        # Initialize
        if data_manager.initialize():
            print("✅ Multi-coin data manager initialized successfully!")
            
            # Get cache status
            status = data_manager.get_cache_status()
            print("\n📈 Cache Status:")
            for coin, timeframes in status.items():
                print(f"\n{coin}:")
                for tf, info in timeframes.items():
                    print(f"  {tf}: {info['count']} candles")
                    if info['latest_time']:
                        print(f"    Latest: {info['latest_time']}")
                    if info['oldest_time']:
                        print(f"    Oldest: {info['oldest_time']}")
            
            # Test data retrieval for each coin
            print("\n🔍 Testing data retrieval...")
            for coin in coins:
                print(f"\n{coin}:")
                for tf in ['1h', '4h', '1d']:
                    df = data_manager.get_model_data(coin, tf)
                    if not df.empty:
                        print(f"  ✅ {tf} data: {len(df)} candles")
                        print(f"    Latest: {df['open_time'].iloc[-1]}")
                    else:
                        print(f"  ❌ {tf} data: No data available")
            
            # Test cache update
            print("\n🔄 Testing cache update...")
            data_manager.force_update()
            
            # Check updated status
            status = data_manager.get_cache_status()
            print("\n📈 Updated Cache Status:")
            for coin, timeframes in status.items():
                total_candles = sum(info['count'] for info in timeframes.values())
                print(f"{coin}: {total_candles} total candles")
                for tf, info in timeframes.items():
                    print(f"  {tf}: {info['count']} candles")
            
            print("\n✅ Multi-coin data manager test completed successfully!")
            return True
            
        else:
            print("❌ Failed to initialize multi-coin data manager")
            return False
            
    except Exception as e:
        print(f"❌ Error testing multi-coin data manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 AiETHTrader Multi-Coin Data Manager Test Suite")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test multi-coin data manager
    data_manager_ok = test_multi_coin_data_manager()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Multi-Coin Data Manager: {'✅ PASS' if data_manager_ok else '❌ FAIL'}")
    
    if data_manager_ok:
        print("\n🎉 Multi-coin data manager test passed!")
        print("All coins are now cached and will update every hour.")
    else:
        print("\n⚠️  Multi-coin data manager test failed. Please check the errors above.")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
