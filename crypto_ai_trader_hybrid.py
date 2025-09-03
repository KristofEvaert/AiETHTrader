#!/usr/bin/env python3
"""
Hybrid Crypto AI Trader
Uses existing data for immediate startup + periodic fresh data downloads
"""

import os
import sys
import json
import time
import yaml
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from binance.spot import Spot as SpotClient
from binance.error import ClientError
from sklearn.preprocessing import StandardScaler

# Import the original functions
from crypto_ai_trader import (
    load_env_file, make_client, build_tf_features, 
    MLP, place_market_buy, place_market_sell
)

def download_fresh_data(client: SpotClient, config: dict):
    """Download fresh data from Binance API"""
    
    print("🔄 Downloading fresh market data...")
    
    try:
        # Get current time
        current_time = datetime.now(timezone.utc)
        
        # Download recent data for each timeframe
        timeframes = config["timeframes"]
        
        for tf in timeframes:
            print(f"  📊 {tf}: Downloading fresh data...")
            
            # For all timeframes, get 7 days of data for proper feature building
            start_time = current_time - timedelta(days=7)  # Last 7 days
            
            # Special handling for 5m data - get more recent data
            if tf == "5m":
                start_time = current_time - timedelta(hours=24)  # Last 24 hours for 5m
                print(f"    ⚠️  5m: Using 24-hour window for more recent data")
            
            # Get klines
            klines = client.klines(
                symbol=config["symbol"],
                interval=tf,
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(current_time.timestamp() * 1000),
                limit=1000
            )
            
            if not klines:
                print(f"    ⚠️  No fresh data for {tf}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # Save to file (overwrite existing)
            filename = f"data/ETHUSDC_{tf}.csv"
            df.to_csv(filename)
            
            print(f"    ✅ {tf}: {len(df)} bars, latest: {df.index[-1]}")
            
            # Rate limiting
            time.sleep(0.1)
        
        print("✅ Fresh data download complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading fresh data: {e}")
        return False

def load_existing_data(config: dict):
    """Load existing data from files"""
    
    print("📁 Loading data from files...")
    
    tf_data = {}
    for tf in config["timeframes"]:
        filename = f"data/ETHUSDC_{tf}.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            tf_data[tf] = df
            print(f"✅ {tf}: {len(df)} bars, latest: {df.index[-1]}")
        else:
            print(f"❌ File not found: {filename}")
            return None
    
    return tf_data

def latest_features_and_signal_hybrid(client: SpotClient, config: dict, scaler: StandardScaler, model: nn.Module, last_refresh_time):
    """
    Hybrid version that uses existing data + checks for fresh data
    """
    symbol = config["symbol"]
    base_tf = config["base_timeframe"]
    current_time = datetime.now(timezone.utc)
    
    # Check if we need to refresh data (every 1 minute)
    refresh_interval = 60  # 1 minute in seconds
    if last_refresh_time is None or (current_time - last_refresh_time).total_seconds() > refresh_interval:
        print(f"⏰ Time for data refresh (every {refresh_interval/60:.0f} minute)")
        if download_fresh_data(client, config):
            last_refresh_time = current_time
            print(f"🕐 Data refreshed at: {last_refresh_time}")
        else:
            print("⚠️  Data refresh failed, using existing data")
    
    # Load data (either fresh or existing)
    tf_data = load_existing_data(config)
    if tf_data is None:
        return None, None, None, last_refresh_time
    
    # Check if we have recent enough data
    base_candles = tf_data[base_tf]
    latest_time = base_candles.index[-1]
    
    # Ensure both are timezone-aware
    if latest_time.tzinfo is None:
        latest_time = latest_time.replace(tzinfo=timezone.utc)
    
    time_diff = current_time - latest_time
    
    print(f"📅 Latest data: {latest_time}")
    print(f"📅 Current time: {current_time}")
    print(f"📅 Time difference: {time_diff}")
    
    # If data is too old (more than 2 hours), return None
    if time_diff.total_seconds() > 7200:  # 2 hours
        print(f"⚠️  Data is too old: {time_diff.total_seconds()/3600:.1f} hours")
        return None, None, None, last_refresh_time
    
    # Build features from data
    feats = {tf: build_tf_features(df, prefix=tf) for tf, df in tf_data.items()}
    others = {tf: feat for tf, feat in feats.items() if tf != base_tf}
    
    # Manual merge implementation to avoid the broken merge_asof_multi
    base_features = feats[base_tf].copy()
    base_features = base_features.sort_index().reset_index()
    base_features = base_features.rename(columns={base_features.index.name or "timestamp": "ts"})
    
    # Merge each timeframe manually with forward fill to avoid NaN
    merged = base_features.copy()
    for name, feat in others.items():
        f = feat.sort_index().reset_index()
        f = f.rename(columns={f.index.name or "timestamp": "ts"})
        # Use forward fill to propagate values and avoid NaN
        merged = pd.merge_asof(merged.sort_values("ts"), f.sort_values("ts"), on="ts", direction="backward")
        # Forward fill any NaN values from this merge
        merged = merged.ffill()
    
    merged = merged.set_index("ts").sort_index()
    
    # Additional forward fill to handle any remaining NaN
    merged = merged.ffill().bfill()
    
    if merged is None or merged.empty:
        print("❌ Feature merging failed")
        return None, None, None, last_refresh_time
    
    print(f"✅ Features merged: {merged.shape}")
    
    # Check for NaN values
    nan_count = merged.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠️  Warning: {nan_count} NaN values found, attempting to clean...")
        # Final cleanup of any remaining NaN
        merged = merged.fillna(0)  # Replace any remaining NaN with 0
    
    # Get the latest features
    X_now = merged.dropna()
    
    if X_now.empty:
        print("⚠️  No features after dropna")
        print("🔍 Debug: merged shape before dropna:", merged.shape)
        print("🔍 Debug: NaN count:", merged.isna().sum().sum())
        print("🔍 Debug: Sample of merged data:")
        print(merged.head())
        return None, None, None, last_refresh_time
    
    print(f"✅ Features built: {X_now.shape}")
    
    # Get prediction
    cols = X_now.columns.tolist()
    X_sc = scaler.transform(X_now.values)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        proba = model(torch.tensor(X_sc, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    
    proba_series = pd.Series(proba, index=X_now.index, name="proba")
    last_ts = proba_series.index[-1]
    last_proba = float(proba_series.iloc[-1])
    last_price = float(base_candles.loc[last_ts, "close"])
    
    return last_ts, last_proba, last_price, last_refresh_time

def run_live_loop_hybrid(config: dict):
    """Hybrid live loop with periodic data refresh"""
    
    env = load_env_file()
    api_key = env.get("BINANCE_API_KEY", os.environ.get("BINANCE_API_KEY", ""))
    api_secret = env.get("BINANCE_API_SECRET", os.environ.get("BINANCE_API_SECRET", ""))
    use_testnet = env.get("USE_TESTNET", str(config.get("use_testnet", True))).lower() == "true"
    dry_run = env.get("DRY_RUN", str(config["live"]["dry_run"])).lower() == "true"
    trade_quote_size = float(env.get("TRADE_QUOTE_SIZE", config["live"]["trade_quote_size"]))

    client = make_client(api_key, api_secret, use_testnet=use_testnet)

    # Load artifacts
    artifacts_dir = config["storage"]["artifacts_dir"]
    model_path = os.path.join(artifacts_dir, "model.pt")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        print("[ERROR] Train the model first to create artifacts.")
        return

    scaler = joblib.load(scaler_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-create model with correct input size
    sample_cols = joblib.load(os.path.join(artifacts_dir, "feature_columns.json"))
    model = MLP(in_features=len(sample_cols))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    print("🚀 Starting HYBRID live loop")
    print("📁 Data source: Existing files + periodic fresh downloads")
    print("⏰ Data refresh: Every 1 minute")
    print("⏰ Data window: Last 7 days for proper features")
    print("⏰ Press Ctrl+C to stop")
    
    position_file = os.path.join(artifacts_dir, "position_state.json")
    bankroll_file = os.path.join(artifacts_dir, "bankroll_state.json")
    
    # Load or initialize bankroll state
    if os.path.exists(bankroll_file):
        with open(bankroll_file, "r") as f:
            bankroll_state = json.load(f)
        print(f"💰 Loaded bankroll: {bankroll_state['current_usdc']:.2f} USDC")
    else:
        bankroll_state = {
            "starting_usdc": 100.0,
            "current_usdc": 100.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "peak_balance": 100.0,
            "trade_history": []
        }
        print("💰 Initialized new bankroll: 100.00 USDC")
    
    # Simple position state
    if os.path.exists(position_file):
        with open(position_file,"r") as f:
            state = json.load(f)
    else:
        state = {"has_position": False, "base_qty": 0.0, "entry_price": 0.0, "entry_time": None}

    threshold = config["backtest"]["threshold"]
    last_refresh_time = None

    try:
        while True:
            ts, proba, price, last_refresh_time = latest_features_and_signal_hybrid(client, config, scaler, model, last_refresh_time)
            if ts is None:
                print("⏳ Waiting for enough data...")
                time.sleep(config["live"]["poll_seconds"])
                continue
                
            print(f"🎯 [{ts}] proba_up={proba:.3f} price={price:.2f}")
            
            if (proba is not None) and (proba > threshold) and not state["has_position"]:
                # Enter long
                if dry_run:
                    print(f"🟢 DRY_RUN: would BUY ~{trade_quote_size} USDC of {config['symbol']}")
                    state["has_position"] = True
                    state["base_qty"] = (trade_quote_size / price) * 0.999  # fee slippage estimate
                else:
                    try:
                        order = place_market_buy(client, config["symbol"], trade_quote_size)
                        fills_qty = sum(float(f["qty"]) for f in order.get("fills", [])) if "fills" in order else 0.0
                        state["has_position"] = True
                        state["base_qty"] = fills_qty if fills_qty>0 else (trade_quote_size/price)
                        print("🟢 BUY order executed:", order)
                    except ClientError as e:
                        print("❌ BUY error:", e)
                        
            elif state["has_position"]:
                # Naive exit: after one base bar, just sell on next loop iteration
                if dry_run:
                    print(f"🔴 DRY_RUN: would SELL {state['base_qty']:.6f} {config['symbol'][:-4]}")
                    state["has_position"] = False
                    state["base_qty"] = 0.0
                else:
                    try:
                        order = place_market_sell(client, config["symbol"], state["base_qty"])
                        state["has_position"] = False
                        state["base_qty"] = 0.0
                        print("🔴 SELL order executed:", order)
                    except ClientError as e:
                        print("❌ SELL error:", e)

            # Save position state
            with open(position_file,"w") as f:
                json.dump(state, f)

            time.sleep(config["live"]["poll_seconds"])
            
    except KeyboardInterrupt:
        print("\n⏹️  Stopped.")
        print(f"\n💰 FINAL BANKROLL SUMMARY:")
        print(f"   Starting: ${bankroll_state['starting_usdc']:.2f} USDC")
        print(f"   Current: ${bankroll_state['current_usdc']:.2f} USDC")
        print(f"   Total PnL: ${bankroll_state['total_pnl']:.2f} USDC")
        print(f"   Total Trades: {bankroll_state['total_trades']}")
        if bankroll_state['total_trades'] > 0:
            print(f"   Win Rate: {bankroll_state['winning_trades']}/{bankroll_state['total_trades']} ({bankroll_state['winning_trades']/bankroll_state['total_trades']*100:.1f}%)")
        print(f"   Max Drawdown: {bankroll_state['max_drawdown']:.2f}%")
        print(f"   Peak Balance: ${bankroll_state['peak_balance']:.2f} USDC")

if __name__ == "__main__":
    print("🚀 Hybrid Crypto AI Trader - Live Trading Mode")
    print("=" * 60)
    
    # Load config
    config_path = "config_live_trading.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config: {config_path}")
    else:
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    # Start hybrid live loop
    run_live_loop_hybrid(config)
