#!/usr/bin/env python3
"""
Start Live Trading
=================

Simple script to start live trading with proper configuration and safety checks.
"""

import os
import sys
import yaml
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def check_requirements():
    """Check if all requirements are met for live trading."""
    print("Checking requirements...")
    
    # Check if models exist
    artifacts_dir = "./artifacts"
    if not os.path.exists(artifacts_dir):
        print("❌ Artifacts directory not found. Please train models first.")
        return False
    
    # Check for trained models
    model_dirs = [d for d in os.listdir(artifacts_dir) 
                 if os.path.isdir(os.path.join(artifacts_dir, d))]
    
    if not model_dirs:
        print("❌ No trained models found. Please train models first.")
        return False
    
    print(f"✅ Found {len(model_dirs)} trained models: {model_dirs}")
    
    # Check if data directory exists
    data_dir = "./data"
    if not os.path.exists(data_dir):
        print("❌ Data directory not found. Creating...")
        os.makedirs(data_dir, exist_ok=True)
    
    print("✅ Data directory ready")
    
    # Check if cache directory exists
    cache_dir = "./cache"
    if not os.path.exists(cache_dir):
        print("❌ Cache directory not found. Creating...")
        os.makedirs(cache_dir, exist_ok=True)
    
    print("✅ Cache directory ready")
    
    return True

def load_config(config_file):
    """Load configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from {config_file}")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def display_config_summary(config):
    """Display configuration summary."""
    print("\n" + "="*60)
    print("LIVE TRADING CONFIGURATION SUMMARY")
    print("="*60)
    
    print(f"Symbol: {config['symbol']}")
    print(f"Timeframes: {config['timeframes']}")
    print(f"Position Size: ${config['live']['trade_quote_size']}")
    print(f"Probability Threshold: {config['backtest']['threshold']}")
    print(f"Dry Run: {config['live']['dry_run']}")
    print(f"Poll Interval: {config['live']['poll_seconds']} seconds")
    print(f"Use Testnet: {config['exchange']['use_testnet']}")
    
    print("\nRisk Management:")
    print(f"  Max Positions: {config['live']['max_positions']}")
    print(f"  Max Daily Trades: {config['live']['risk_management']['max_daily_trades']}")
    print(f"  Max Drawdown: {config['live']['risk_management']['max_drawdown']*100}%")
    
    print("\nData Management:")
    print(f"  Cache Size Limits: {config['data_management']['cache_size_limits']}")
    print(f"  Update Frequency: {config['data_management']['update_frequency']} seconds")
    
    print("="*60)

def confirm_start():
    """Ask user to confirm starting live trading."""
    print("\n⚠️  LIVE TRADING WARNING ⚠️")
    print("This will start the live trading engine.")
    print("Make sure you understand the risks involved.")
    print("The system will start in DRY RUN mode by default.")
    
    response = input("\nDo you want to continue? (yes/no): ").lower().strip()
    return response in ['yes', 'y']

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Start Live Trading Engine')
    parser.add_argument('--config', default='config_live.yaml', 
                       help='Configuration file (default: config_live.yaml)')
    parser.add_argument('--api-key', help='Binance API key')
    parser.add_argument('--api-secret', help='Binance API secret')
    parser.add_argument('--live', action='store_true', 
                       help='Start in live mode (not dry run)')
    parser.add_argument('--force', action='store_true', 
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    print("🚀 AiETHTrader Live Trading Engine")
    print("="*50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed. Exiting.")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print("❌ Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # Override configuration if needed
    if args.live:
        config['live']['dry_run'] = False
        print("⚠️  LIVE MODE ENABLED - REAL TRADES WILL BE EXECUTED!")
    
    # Display configuration
    display_config_summary(config)
    
    # Confirm start
    if not args.force:
        if not confirm_start():
            print("❌ Trading cancelled by user.")
            sys.exit(0)
    
    # Start trading engine
    try:
        from live_trading_engine import LiveTradingEngine
        
        # Get API credentials from environment or command line
        api_key = args.api_key or os.getenv('BINANCE_API_KEY')
        api_secret = args.api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("❌ API credentials not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET in .env file or use --api-key and --api-secret arguments.")
            sys.exit(1)
        
        print("\n🔄 Starting live trading engine...")
        engine = LiveTradingEngine(config, api_key, api_secret)
        
        # Start trading
        engine.start_trading()
        
    except KeyboardInterrupt:
        print("\n⏹️  Trading stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting trading engine: {e}")
        sys.exit(1)
    finally:
        print("👋 Live trading engine stopped.")
        print(f"Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
