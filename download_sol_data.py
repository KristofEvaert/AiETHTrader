#!/usr/bin/env python3
"""
Download SOLUSDC data with specific candle counts:
- 1h: 8810 candles
- 4h: 2201 candles  
- 1d: 366 candles
"""

import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta
import time

def download_klines_chunked(client, symbol, interval, target_limit):
    """Download klines data from Binance in chunks (max 1000 per request)."""
    try:
        print(f"  📥 Downloading {target_limit} {interval} candles for {symbol}...")
        
        all_klines = []
        remaining = target_limit
        end_time = None  # Start from most recent
        
        while remaining > 0:
            # Calculate how many candles to request this time (max 1000)
            chunk_size = min(remaining, 1000)
            
            print(f"    📦 Requesting {chunk_size} candles (remaining: {remaining})...")
            
            # Make API call
            if end_time:
                klines = client.get_klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=chunk_size,
                    endTime=end_time
                )
            else:
                klines = client.get_klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=chunk_size
                )
            
            if not klines:
                print(f"    ⚠️  No more data available")
                break
                
            all_klines.extend(klines)
            remaining -= len(klines)
            
            # Set end_time for next request (go backwards in time)
            if len(klines) > 0:
                end_time = klines[0][0] - 1  # Start before the first candle of this chunk
            
            print(f"    ✅ Got {len(klines)} candles, {remaining} remaining")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        if not all_klines:
            print(f"  ❌ No data received for {symbol} {interval}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to proper types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove the 'ignore' column
        df = df.drop('ignore', axis=1)
        
        # Sort by time (oldest first)
        df = df.sort_values('open_time').reset_index(drop=True)
        
        print(f"  ✅ Downloaded {len(df)} {interval} candles total")
        print(f"     Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        
        return df
        
    except Exception as e:
        print(f"  ❌ Error downloading {symbol} {interval}: {e}")
        return None

def download_coin_data(symbol, target_candles):
    """Download data for a specific coin."""
    print(f"🪙 Downloading {symbol} Data")
    print("=" * 40)
    
    # Load API keys from environment
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    use_testnet = os.getenv('USE_TESTNET', 'false').lower() == 'true'
    
    if not api_key or not api_secret:
        print("❌ API keys not found in environment variables")
        return
    
    # Initialize Binance client
    if use_testnet:
        client = Client(api_key, api_secret, testnet=True)
        print("🔧 Using Binance Testnet")
    else:
        client = Client(api_key, api_secret)
        print("🌐 Using Binance Mainnet")
    
    # Test connection
    try:
        client.ping()
        print("✅ Connected to Binance API")
    except Exception as e:
        print(f"❌ Failed to connect to Binance API: {e}")
        return
    
    # symbol and target_candles are now parameters
    
    # Create coin directory
    coin_dir = f"data/{symbol}"
    os.makedirs(coin_dir, exist_ok=True)
    print(f"📁 Created directory: {coin_dir}")
    print()
    
    # Download data for each timeframe
    for timeframe, target_count in target_candles.items():
        print(f"⏰ Downloading {timeframe} data...")
        
        # Download data
        df = download_klines_chunked(client, symbol, timeframe, target_count)
        
        if df is not None:
            # Save to file
            filename = f"{coin_dir}/{timeframe}.csv"
            df.to_csv(filename, index=False)
            print(f"  💾 Saved to: {filename}")
            
            # Verify candle count
            actual_count = len(df)
            if actual_count == target_count:
                print(f"  ✅ Perfect! Got exactly {actual_count} candles")
            else:
                print(f"  ⚠️  Got {actual_count} candles (target: {target_count})")
        else:
            print(f"  ❌ Failed to download {timeframe} data")
        
        print()
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"🎉 {symbol} data download complete!")
    
    # Show final structure
    print(f"\n📂 Final structure:")
    for timeframe in target_candles.keys():
        filename = f"{coin_dir}/{timeframe}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            size_kb = os.path.getsize(filename) / 1024
            print(f"  {timeframe}.csv: {len(df)} candles ({size_kb:.1f} KB)")

def main():
    """Download SOLUSDC data."""
    symbol = "SOLUSDC"
    target_candles = {
        '1h': 8810,
        '4h': 2201,
        '1d': 366
    }
    download_coin_data(symbol, target_candles)

if __name__ == "__main__":
    main()
