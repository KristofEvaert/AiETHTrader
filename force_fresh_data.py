#!/usr/bin/env python3
"""
Force Fresh Data Download Script
Diagnoses and forces download of truly fresh market data
"""

import os
import sys
from dotenv import load_dotenv
from binance.spot import Spot
from datetime import datetime, timezone, timedelta
import pandas as pd
import time

def force_fresh_data():
    """Force download of truly fresh data"""
    
    print("🔄 FORCING TRULY FRESH DATA DOWNLOAD")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Create client
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        client = Spot(api_key=api_key, api_secret=api_secret)
        print("✅ Binance client created")
        
        # Get current time
        current_time = datetime.now(timezone.utc)
        print(f"⏰ Current time: {current_time}")
        
        # Test with very recent data first
        print(f"\n🔍 TESTING WITH VERY RECENT DATA")
        print(f"📅 Requesting data from last 2 hours only")
        
        # Download data for each timeframe with very recent window
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        
        for tf in timeframes:
            print(f"\n📊 {tf}: Testing very recent data...")
            
            # Request only last 2 hours for testing
            start_time = current_time - timedelta(hours=2)
            
            print(f"   From: {start_time}")
            print(f"   To: {current_time}")
            
            # Get klines
            klines = client.klines(
                symbol="ETHUSDC",
                interval=tf,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(current_time.timestamp() * 1000),
                limit=1000
            )
            
            if not klines:
                print(f"    ⚠️  No data for {tf}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            print(f"    ✅ {tf}: {len(df)} bars")
            print(f"    📅 First: {df.index[0]}")
            print(f"    📅 Last: {df.index[-1]}")
            print(f"    💰 Latest price: {df['close'].iloc[-1]:.2f}")
            
            # Check if data is actually recent
            time_diff = current_time - df.index[-1]
            if time_diff.total_seconds() < 3600:  # Less than 1 hour
                print(f"    🟢 Data is recent: {time_diff}")
            else:
                print(f"    🔴 Data is OLD: {time_diff}")
            
            # Rate limiting
            time.sleep(0.2)
        
        print(f"\n🎯 DIAGNOSIS COMPLETE")
        print(f"📅 If data is still old, there's an API issue")
        print(f"📅 If data is recent, the previous system had a bug")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fresh data download: {e}")
        return False

if __name__ == "__main__":
    success = force_fresh_data()
    if success:
        print("\n🚀 Fresh data test complete!")
    else:
        print("\n💥 Fresh data test failed!")
