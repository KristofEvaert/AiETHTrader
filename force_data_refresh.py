#!/usr/bin/env python3
"""
Force Data Refresh Script
Downloads extra data to ensure live trading has enough recent data
"""

import os
import sys
from dotenv import load_dotenv
from binance.spot import Spot
from datetime import datetime, timezone, timedelta
import pandas as pd
import time

def force_data_refresh():
    """Force download of extra data for live trading"""
    
    print("🔄 FORCING DATA REFRESH FOR LIVE TRADING")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Create client
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        client = Spot(api_key=api_key, api_secret=api_secret)
        print("✅ Binance client created")
        
        # Calculate time range - get 2 extra hours
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=4)  # 4 hours instead of 2
        
        print(f"📅 Downloading data from {start_time} to {end_time}")
        print(f"📅 This will give us 2+ hours of extra data")
        
        # Download data for each timeframe
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        
        for tf in timeframes:
            print(f"\n📊 Downloading {tf} data...")
            
            # Get klines
            klines = client.klines(
                symbol="ETHUSDC",
                interval=tf,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=1000
            )
            
            if not klines:
                print(f"⚠️  No data for {tf}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Save to file
            filename = f"data/ETHUSDC_{tf}.csv"
            df.to_csv(filename)
            
            print(f"✅ {tf}: {len(df)} bars, latest: {df.index[-1]}")
            print(f"   Saved to: {filename}")
            
            # Rate limiting
            time.sleep(0.1)
        
        print(f"\n🎉 DATA REFRESH COMPLETE!")
        print(f"📅 Latest data now extends to: {end_time}")
        print(f"⏰ Ready for live trading!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data refresh: {e}")
        return False

if __name__ == "__main__":
    success = force_data_refresh()
    if success:
        print("\n🚀 You can now start live trading!")
    else:
        print("\n💥 Data refresh failed!")
