#!/usr/bin/env python3
"""
Working Crypto AI Trader
Uses the existing working data for live trading
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

def latest_features_and_signal_simple(client: SpotClient, config: dict, scaler: StandardScaler, model: nn.Module):
    """
    Simple version that uses existing data without complex merging
    """
    symbol = config["symbol"]
    base_tf = config["base_timeframe"]
    
    # Load existing data
    tf_data = load_existing_data(config)
    if tf_data is None:
        return None, None, None
    
    # Check if we have recent enough data
    base_candles = tf_data[base_tf]
    latest_time = base_candles.index[-1]
    current_time = datetime.now(timezone.utc)
    
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
        return None, None, None
    
    # Build features from base timeframe only (simplified approach)
    print("🔧 Building features from base timeframe only...")
    base_features = build_tf_features(tf_data[base_tf], prefix=base_tf)
    
    if base_features.empty:
        print("⚠️  No base features built")
        return None, None, None
    
    print(f"✅ Base features built: {base_features.shape}")
    
    # Get prediction from base features only
    X_now = base_features.dropna()
    
    if X_now.empty:
        print("⚠️  No features after dropna")
        return None, None, None
    
    print(f"✅ Features ready: {X_now.shape}")
    
    # Get prediction
    X_sc = scaler.transform(X_now.values)
    device = next(model.parameters()).device
    
    with torch.no_grad():
        proba = model(torch.tensor(X_sc, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    
    proba_series = pd.Series(proba, index=X_now.index, name="proba")
    last_ts = proba_series.index[-1]
    last_proba = float(proba_series.iloc[-1])
    last_price = float(base_candles.loc[last_ts, "close"])
    
    return last_ts, last_proba, last_price

def run_live_loop_simple(config: dict):
    """Simple live loop that uses existing data"""
    
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

    print("🚀 Starting SIMPLE live loop")
    print("📁 Data source: Existing files (simplified approach)")
    print("⏰ Press Ctrl+C to stop")
    
    position_file = os.path.join(artifacts_dir, "position_state.json")
    
    # Simple position state
    if os.path.exists(position_file):
        with open(position_file,"r") as f:
            state = json.load(f)
    else:
        state = {"has_position": False, "base_qty": 0.0}

    threshold = config["backtest"]["threshold"]

    try:
        while True:
            ts, proba, price = latest_features_and_signal_simple(client, config, scaler, model)
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

if __name__ == "__main__":
    print("🚀 Simple Crypto AI Trader - Live Trading Mode")
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
    
    # Start simple live loop
    run_live_loop_simple(config)
