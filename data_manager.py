#!/usr/bin/env python3
"""
Data Manager for Live Trading
============================

This module handles efficient data retrieval and caching for live trading:
1. Initial data load with sufficient historical data for models
2. Hourly data updates to keep cache current
3. Automatic cache management to prevent memory bloat
4. Support for multiple timeframes (1h, 4h, 1d)

Strategy:
- Load 2+ years of historical data on startup
- Update cache every hour with new data
- Maintain rolling window (remove oldest data when adding new)
- Cache size stays constant after initial load
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple
import requests
import json
from binance.client import Client
from binance.exceptions import BinanceAPIException
import warnings
warnings.filterwarnings('ignore')

class DataManager:
    """Manages data retrieval and caching for live trading."""
    
    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary
            api_key: Binance API key (optional, uses testnet if not provided)
            api_secret: Binance API secret (optional, uses testnet if not provided)
        """
        self.config = config
        self.symbol = config['symbol']
        self.timeframes = config['timeframes']
        self.lookback_years = config.get('lookback_years', 2)
        self.cache_dir = config['storage']['data_dir']
        
        # Initialize Binance client
        if api_key and api_secret:
            self.client = Client(api_key, api_secret)
            self.use_testnet = False
        else:
            # Use testnet for safety
            self.client = Client(testnet=True)
            self.use_testnet = True
        
        # Data cache
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
        # Cache configuration
        self.max_cache_size = {
            '1h': 17520,  # 2 years of hourly data
            '4h': 4380,   # 2 years of 4-hour data
            '1d': 730     # 2 years of daily data
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize cache
        self.initialize_cache()
    
    def setup_logging(self):
        """Setup logging for data manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataManager')
    
    def initialize_cache(self):
        """Initialize data cache with historical data."""
        self.logger.info("Initializing data cache...")
        
        for timeframe in self.timeframes:
            self.logger.info(f"Loading {timeframe} data for {self.symbol}")
            
            # Load existing data if available
            cache_file = f"{self.cache_dir}/{self.symbol}_{timeframe}.csv"
            if os.path.exists(cache_file):
                self.logger.info(f"Loading existing {timeframe} data from cache")
                df = pd.read_csv(cache_file)
                df['open_time'] = pd.to_datetime(df['open_time'])
                self.data_cache[timeframe] = df
            else:
                # Download historical data
                self.logger.info(f"Downloading historical {timeframe} data")
                df = self.download_historical_data(timeframe)
                self.data_cache[timeframe] = df
                self.save_cache_to_file(timeframe)
        
        self.logger.info("Data cache initialization complete")
        self.logger.info(f"Cache status: {[(tf, len(df)) for tf, df in self.data_cache.items()]}")
    
    def download_historical_data(self, timeframe: str) -> pd.DataFrame:
        """
        Download historical data for a specific timeframe.
        
        Args:
            timeframe: Timeframe string (1h, 4h, 1d)
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)
            
            self.logger.info(f"Downloading {timeframe} data from {start_date} to {end_date}")
            
            # Download data in chunks to avoid API limits
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                try:
                    # Download chunk (max 1000 candles per request)
                    klines = self.client.get_historical_klines(
                        self.symbol,
                        timeframe,
                        current_start.strftime("%d %b %Y %H:%M:%S"),
                        end_date.strftime("%d %b %Y %H:%M:%S"),
                        limit=1000
                    )
                    
                    if not klines:
                        break
                    
                    # Convert to DataFrame
                    df_chunk = pd.DataFrame(klines, columns=[
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'trades',
                        'taker_buy_base', 'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert data types
                    df_chunk['open_time'] = pd.to_datetime(df_chunk['open_time'], unit='ms')
                    df_chunk['close_time'] = pd.to_datetime(df_chunk['close_time'], unit='ms')
                    
                    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
                        df_chunk[col] = pd.to_numeric(df_chunk[col])
                    
                    all_data.append(df_chunk)
                    
                    # Update start date for next chunk
                    current_start = df_chunk['open_time'].iloc[-1] + timedelta(hours=1)
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except BinanceAPIException as e:
                    self.logger.error(f"Binance API error: {e}")
                    time.sleep(1)
                    continue
                except Exception as e:
                    self.logger.error(f"Error downloading data: {e}")
                    time.sleep(1)
                    continue
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                self.logger.info(f"Downloaded {len(df)} {timeframe} candles")
                return df
            else:
                self.logger.error(f"No data downloaded for {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error downloading historical data: {e}")
            return pd.DataFrame()
    
    def save_cache_to_file(self, timeframe: str):
        """Save cache data to file."""
        with self.cache_lock:
            if timeframe in self.data_cache:
                cache_file = f"{self.cache_dir}/{self.symbol}_{timeframe}.csv"
                self.data_cache[timeframe].to_csv(cache_file, index=False)
                self.logger.info(f"Saved {timeframe} cache to {cache_file}")
    
    def get_latest_data(self, timeframe: str) -> pd.DataFrame:
        """
        Get the latest data for a specific timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            DataFrame with latest data
        """
        with self.cache_lock:
            if timeframe in self.data_cache:
                return self.data_cache[timeframe].copy()
            else:
                return pd.DataFrame()
    
    def update_cache(self):
        """Update cache with latest data from exchange."""
        self.logger.info("Updating data cache...")
        
        for timeframe in self.timeframes:
            try:
                # Get latest data from exchange
                latest_data = self.download_latest_data(timeframe)
                
                if not latest_data.empty:
                    with self.cache_lock:
                        if timeframe in self.data_cache:
                            # Merge with existing data
                            existing_df = self.data_cache[timeframe]
                            
                            # Ensure timezone consistency
                            if existing_df['open_time'].dt.tz is None:
                                existing_df['open_time'] = existing_df['open_time'].dt.tz_localize('UTC')
                            if latest_data['open_time'].dt.tz is None:
                                latest_data['open_time'] = latest_data['open_time'].dt.tz_localize('UTC')
                            
                            # Remove duplicates and sort
                            combined_df = pd.concat([existing_df, latest_data], ignore_index=True)
                            combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                            
                            # Maintain cache size limit
                            max_size = self.max_cache_size.get(timeframe, 10000)
                            if len(combined_df) > max_size:
                                combined_df = combined_df.tail(max_size)
                            
                            self.data_cache[timeframe] = combined_df
                            self.logger.info(f"Updated {timeframe} cache: {len(combined_df)} candles")
                        else:
                            self.data_cache[timeframe] = latest_data
                            self.logger.info(f"Initialized {timeframe} cache: {len(latest_data)} candles")
                    
                    # Save to file
                    self.save_cache_to_file(timeframe)
                
            except Exception as e:
                self.logger.error(f"Error updating {timeframe} cache: {e}")
    
    def download_latest_data(self, timeframe: str) -> pd.DataFrame:
        """
        Download the latest data for a specific timeframe.
        
        Args:
            timeframe: Timeframe string
            
        Returns:
            DataFrame with latest data
        """
        try:
            # Get latest klines
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=timeframe,
                limit=100  # Get last 100 candles to ensure we have latest
            )
            
            if not klines:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convert data types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading latest {timeframe} data: {e}")
            return pd.DataFrame()
    
    def get_data_for_model(self, timeframe: str, lookback_bars: int = 100) -> pd.DataFrame:
        """
        Get data for model prediction with specified lookback.
        
        Args:
            timeframe: Timeframe string
            lookback_bars: Number of bars to look back
            
        Returns:
            DataFrame with recent data for model
        """
        with self.cache_lock:
            if timeframe in self.data_cache:
                df = self.data_cache[timeframe].copy()
                return df.tail(lookback_bars)
            else:
                return pd.DataFrame()
    
    def is_cache_ready(self) -> bool:
        """Check if cache is ready for trading."""
        required_timeframes = ['1h', '4h', '1d']
        
        for tf in required_timeframes:
            if tf not in self.data_cache or len(self.data_cache[tf]) < 100:
                return False
        
        return True
    
    def get_cache_status(self) -> dict:
        """Get current cache status."""
        status = {}
        for timeframe in self.timeframes:
            if timeframe in self.data_cache:
                df = self.data_cache[timeframe]
                status[timeframe] = {
                    'count': len(df),
                    'latest_time': df['open_time'].iloc[-1] if len(df) > 0 else None,
                    'oldest_time': df['open_time'].iloc[0] if len(df) > 0 else None
                }
            else:
                status[timeframe] = {'count': 0, 'latest_time': None, 'oldest_time': None}
        
        return status
    
    def start_hourly_updates(self):
        """Start hourly data updates in background thread."""
        def update_worker():
            while True:
                try:
                    # Wait until next hour
                    now = datetime.now()
                    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    wait_seconds = (next_hour - now).total_seconds()
                    
                    self.logger.info(f"Waiting {wait_seconds/60:.1f} minutes until next hour for data update")
                    time.sleep(wait_seconds)
                    
                    # Update cache
                    self.update_cache()
                    
                except Exception as e:
                    self.logger.error(f"Error in hourly update worker: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        # Start background thread
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()
        self.logger.info("Started hourly data update worker")
    
    def force_update(self):
        """Force immediate cache update."""
        self.logger.info("Forcing immediate cache update...")
        self.update_cache()
    
    def cleanup_old_cache_files(self):
        """Clean up old cache files to save disk space."""
        try:
            for timeframe in self.timeframes:
                cache_file = f"{self.cache_dir}/{self.symbol}_{timeframe}.csv"
                if os.path.exists(cache_file):
                    # Keep only the most recent cache file
                    # You could implement more sophisticated cleanup here
                    pass
        except Exception as e:
            self.logger.error(f"Error cleaning up cache files: {e}")

class LiveDataManager:
    """High-level interface for live trading data management."""
    
    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        """Initialize live data manager."""
        self.config = config
        self.data_manager = DataManager(config, api_key, api_secret)
        self.logger = logging.getLogger('LiveDataManager')
        
        # Ensure cache directory exists
        os.makedirs(config['storage']['data_dir'], exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the data manager for live trading.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing live data manager...")
            
            # Check if cache is ready
            if not self.data_manager.is_cache_ready():
                self.logger.error("Cache not ready for trading")
                return False
            
            # Start hourly updates
            self.data_manager.start_hourly_updates()
            
            self.logger.info("Live data manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing live data manager: {e}")
            return False
    
    def get_model_data(self, coin: str, timeframe: str = '1h') -> pd.DataFrame:
        """
        Get data for model prediction.
        
        Args:
            coin: Coin symbol (e.g., 'ETHUSDC')
            timeframe: Timeframe string
            
        Returns:
            DataFrame with data for model
        """
        return self.data_manager.get_data_for_model(timeframe)
    
    def get_cache_status(self) -> dict:
        """Get current cache status."""
        return self.data_manager.get_cache_status()
    
    def force_update(self):
        """Force immediate data update."""
        self.data_manager.force_update()
    
    def is_ready(self) -> bool:
        """Check if data manager is ready for trading."""
        return self.data_manager.is_cache_ready()

def main():
    """Test the data manager."""
    import yaml
    
    # Load config
    with open('config_1h_trading.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data manager
    data_manager = LiveDataManager(config)
    
    # Initialize
    if data_manager.initialize():
        print("Data manager initialized successfully!")
        
        # Show cache status
        status = data_manager.get_cache_status()
        for tf, info in status.items():
            print(f"{tf}: {info['count']} candles, latest: {info['latest_time']}")
        
        # Keep running for testing
        try:
            while True:
                time.sleep(60)
                status = data_manager.get_cache_status()
                print(f"Cache status: {[(tf, info['count']) for tf, info in status.items()]}")
        except KeyboardInterrupt:
            print("Stopping data manager...")
    else:
        print("Failed to initialize data manager")

if __name__ == "__main__":
    main()
