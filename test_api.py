#!/usr/bin/env python3
"""
Simple API Test Script
Tests Binance API connection and data retrieval
"""

import os
from dotenv import load_dotenv
from binance.spot import Spot
from datetime import datetime, timezone, timedelta
import time

def test_binance_api():
    """Test Binance API connection and data retrieval"""
    
    print("🔍 TESTING BINANCE API CONNECTION")
    print("=" * 50)
    
    # Load environment variables from .env file
    print("📁 Loading .env file...")
    load_dotenv()
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API keys not found in environment")
        print(f"API Key: {api_key}")
        print(f"API Secret: {api_secret}")
        return False
    
    print(f"✅ API Key: {api_key[:10]}...")
    print(f"✅ API Secret: {api_secret[:10]}...")
    
    try:
        # Create client
        print("\n🔌 Creating Binance client...")
        client = Spot(api_key=api_key, api_secret=api_secret)
        print("✅ Client created successfully")
        
        # Test basic connection
        print("\n📡 Testing basic connection...")
        server_time = client.time()
        print(f"✅ Server time: {server_time}")
        
        # Test data retrieval
        print("\n📊 Testing data retrieval...")
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=2)
        
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        
        # Get recent klines
        klines = client.klines(
            symbol="ETHUSDC",
            interval="15m",
            startTime=int(start_time.timestamp() * 1000),
            endTime=int(end_time.timestamp() * 1000),
            limit=100
        )
        
        print(f"✅ Retrieved {len(klines)} klines")
        
        if klines:
            latest = klines[-1]
            latest_time = datetime.fromtimestamp(latest[0] / 1000, tz=timezone.utc)
            print(f"✅ Latest data: {latest_time}")
            print(f"✅ Latest price: {latest[4]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_binance_api()
    if success:
        print("\n🎉 API test successful! The connection is working.")
    else:
        print("\n💥 API test failed! There's a connection issue.")
