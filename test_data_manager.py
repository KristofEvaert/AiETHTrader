#!/usr/bin/env python3
"""
Test Data Manager
================

Test script to verify the data manager works correctly before starting live trading.
"""

import os
import sys
import yaml
import time
from datetime import datetime

def test_data_manager():
    """Test the data manager functionality."""
    print("🧪 Testing Data Manager...")
    
    # Load configuration
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Import data manager
    from data_manager import LiveDataManager
    
    try:
        # Initialize data manager
        print("📊 Initializing data manager...")
        data_manager = LiveDataManager(config)
        
        # Initialize
        if data_manager.initialize():
            print("✅ Data manager initialized successfully!")
            
            # Get cache status
            status = data_manager.get_cache_status()
            print("\n📈 Cache Status:")
            for tf, info in status.items():
                print(f"  {tf}: {info['count']} candles")
                if info['latest_time']:
                    print(f"    Latest: {info['latest_time']}")
                if info['oldest_time']:
                    print(f"    Oldest: {info['oldest_time']}")
            
            # Test data retrieval
            print("\n🔍 Testing data retrieval...")
            for tf in ['1h', '4h', '1d']:
                df = data_manager.get_model_data('ETHUSDC', tf)
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
            print("📈 Updated Cache Status:")
            for tf, info in status.items():
                print(f"  {tf}: {info['count']} candles")
            
            print("\n✅ Data manager test completed successfully!")
            return True
            
        else:
            print("❌ Failed to initialize data manager")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_live_trading_engine():
    """Test the live trading engine initialization."""
    print("\n🤖 Testing Live Trading Engine...")
    
    # Load configuration
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Import live trading engine
    from live_trading_engine import LiveTradingEngine
    
    try:
        # Initialize trading engine
        print("🚀 Initializing live trading engine...")
        engine = LiveTradingEngine(config)
        
        # Check if models are loaded
        print(f"📊 Models loaded: {len(engine.models)}")
        for coin in engine.models.keys():
            print(f"  ✅ {coin}")
        
        # Get status
        status = engine.get_status()
        print(f"\n📈 Engine Status:")
        print(f"  Bankroll: ${status['bankroll']:.2f}")
        print(f"  Positions: {status['positions']}")
        print(f"  Models: {status['models_loaded']}")
        print(f"  Running: {status['is_running']}")
        
        print("\n✅ Live trading engine test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing live trading engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 AiETHTrader Live Trading Test Suite")
    print("="*50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test data manager
    data_manager_ok = test_data_manager()
    
    # Test live trading engine
    engine_ok = test_live_trading_engine()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Data Manager: {'✅ PASS' if data_manager_ok else '❌ FAIL'}")
    print(f"Live Trading Engine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if data_manager_ok and engine_ok:
        print("\n🎉 All tests passed! Ready for live trading.")
        print("You can now run: python start_live_trading.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
