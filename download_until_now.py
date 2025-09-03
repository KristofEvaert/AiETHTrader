#!/usr/bin/env python3
"""
Download Data Until Now Script
Downloads data up to the current moment for live trading
"""

import os
import sys
from dotenv import load_dotenv
from binance.spot import Spot
from datetime import datetime, timezone, timedelta
import pandas as pd
import time

def download_until_now():
    """Download data up to the current moment"""
    
    print("🔄 DOWNLOADING DATA UNTIL NOW")
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
        
        # Download data for each timeframe up to now
        timeframes = ["5m", "15m", "30m", "1h", "4h", "1d"]
        
        for tf in timeframes:
            print(f"\n📊 Downloading {tf} data until now...")
            
            # For live trading, we need recent data
            if tf in ["5m", "15m", "30m"]:
                # Get more recent data for shorter timeframes
                start_time = current_time - timedelta(days=7)
            else:
                start_time = current_time - timedelta(days=30)
            
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
        
        print(f"\n🎉 DOWNLOAD COMPLETE!")
        print(f"📅 Data now extends to: {current_time}")
        print(f"⏰ Ready for live trading!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

if __name__ == "__main__":
    success = download_until_now()
    if success:
        print("\n🚀 You can now start live trading!")
    else:
        print("\n💥 Download failed!")
