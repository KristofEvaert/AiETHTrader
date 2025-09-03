#!/usr/bin/env python3
"""
Debug Features Script
Examines the actual structure of features being built
"""

import os
import sys
import yaml
import pandas as pd
from crypto_ai_trader import build_tf_features

def debug_features():
    """Debug the feature structure"""
    
    print("🔍 DEBUGGING FEATURE STRUCTURE")
    print("=" * 50)
    
    # Load config
    config_path = "config_live_trading.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded config: {config_path}")
    else:
        print(f"❌ Config not found: {config_path}")
        return
    
    # Load data
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
            return
    
    # Build features for each timeframe
    print(f"\n🔧 BUILDING FEATURES")
    print("-" * 30)
    
    feats = {}
    for tf, df in tf_data.items():
        try:
            feat = build_tf_features(df, prefix=tf)
            feats[tf] = feat
            print(f"✅ {tf}: {feat.shape}")
            print(f"   Columns: {feat.columns.tolist()}")
            print(f"   Index name: {feat.index.name}")
            print(f"   First few rows:")
            print(f"   {feat.head(2)}")
            print()
        except Exception as e:
            print(f"❌ {tf}: Error: {e}")
    
    # Check the merge_asof_multi function
    print(f"🔧 TESTING MERGE_ASOF_MULTI")
    print("-" * 30)
    
    try:
        from crypto_ai_trader import merge_asof_multi
        
        base_tf = config["base_timeframe"]
        base_features = feats[base_tf]
        other_features = {tf: feat for tf, feat in feats.items() if tf != base_tf}
        
        print(f"Base features: {base_features.shape}")
        print(f"Other features: {list(other_features.keys())}")
        
        # Test the merge
        merged = merge_asof_multi(base_features, other_features)
        
        if merged is not None:
            print(f"✅ Merge successful: {merged.shape}")
            print(f"   Final columns: {merged.columns.tolist()}")
        else:
            print("❌ Merge returned None")
            
    except Exception as e:
        print(f"❌ Merge error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_features()
