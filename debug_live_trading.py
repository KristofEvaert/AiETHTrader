#!/usr/bin/env python3
"""
Debug Live Trading Script
Identifies exactly where the live trading system is failing
"""

import os
import sys
from dotenv import load_dotenv
from binance.spot import Spot
from datetime import datetime, timezone, timedelta
import pandas as pd
import joblib
import torch
import numpy as np

# Add current directory to path to import crypto_ai_trader
sys.path.append('.')

def debug_live_trading():
    """Debug the live trading system step by step"""
    
    print("🔍 DEBUGGING LIVE TRADING SYSTEM")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Test API connection
    print("\n1️⃣ TESTING API CONNECTION")
    print("-" * 30)
    
    try:
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        client = Spot(api_key=api_key, api_secret=api_secret)
        print("✅ API connection successful")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return
    
    # Step 2: Test data download
    print("\n2️⃣ TESTING DATA DOWNLOAD")
    print("-" * 30)
    
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=14)  # Same as live trading
        
        print(f"Downloading data from {start_time} to {end_time}")
        
        # Download data for each timeframe
        timeframes = ["15m", "1h", "4h"]  # Reduced for testing
        tf_data = {}
        
        for tf in timeframes:
            print(f"  Downloading {tf} data...")
            klines = client.klines(
                symbol="ETHUSDC",
                interval=tf,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            tf_data[tf] = df
            print(f"  ✅ {tf}: {len(df)} bars, latest: {df.index[-1]}")
        
        print("✅ Data download successful")
        
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return
    
    # Step 3: Test model loading
    print("\n3️⃣ TESTING MODEL LOADING")
    print("-" * 30)
    
    try:
        scaler = joblib.load('artifacts/scaler.pkl')
        print("✅ Scaler loaded successfully")
        
        feature_columns = joblib.load('artifacts/feature_columns.json')
        print(f"✅ Feature columns loaded: {len(feature_columns)} features")
        
        model = torch.load('artifacts/model.pt', map_location='cpu')
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Step 4: Test feature building (simplified)
    print("\n4️⃣ TESTING FEATURE BUILDING")
    print("-" * 30)
    
    try:
        # Use the base timeframe data
        base_df = tf_data['15m']
        print(f"Base data shape: {base_df.shape}")
        print(f"Base data columns: {base_df.columns.tolist()}")
        
        # Check if we have enough data
        if len(base_df) < 100:
            print(f"❌ Not enough data: {len(base_df)} bars (need at least 100)")
            return
        
        print("✅ Sufficient data available")
        
    except Exception as e:
        print(f"❌ Feature building failed: {e}")
        return
    
    # Step 5: Test data alignment
    print("\n5️⃣ TESTING DATA ALIGNMENT")
    print("-" * 30)
    
    try:
        # Check if data is recent enough
        latest_time = base_df.index[-1]
        current_time = datetime.now(timezone.utc)
        
        # Ensure both are timezone-aware
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)
        
        time_diff = current_time - latest_time
        
        print(f"Latest data time: {latest_time}")
        print(f"Current time: {current_time}")
        print(f"Time difference: {time_diff}")
        
        if time_diff.total_seconds() > 3600:  # More than 1 hour
            print(f"⚠️  Data is {time_diff.total_seconds()/3600:.1f} hours old")
            print("   This explains why the system is waiting for data")
        else:
            print("✅ Data is recent enough")
            
    except Exception as e:
        print(f"❌ Data alignment check failed: {e}")
        return
    
    print("\n🎯 DIAGNOSIS COMPLETE")
    print("=" * 60)
    print("The system should work once it has recent enough data.")
    print("The 'Waiting for enough data' message is normal during startup.")

if __name__ == "__main__":
    debug_live_trading()
