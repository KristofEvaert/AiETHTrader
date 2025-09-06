#!/usr/bin/env python3
"""
Multi-Coin Data Manager for Live Trading
=======================================

This module handles efficient data retrieval and caching for multiple cryptocurrencies:
1. Individual data caches for each coin and timeframe
2. Hourly data updates for all coins
3. Automatic cache management to prevent memory bloat
4. Support for multiple timeframes (1h, 4h, 1d)

Strategy:
- Load 2+ years of historical data for all coins on startup
- Update cache every hour with new data for all coins
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

class MultiCoinDataManager:
    """Manages data retrieval and caching for multiple cryptocurrencies."""
    
    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        """
        Initialize the multi-coin data manager.
        
        Args:
            config: Configuration dictionary
            api_key: Binance API key (optional, uses testnet if not provided)
            api_secret: Binance API secret (optional, uses testnet if not provided)
        """
        self.config = config
        self.timeframes = config['timeframes']
        self.lookback_years = config.get('lookback_years', 2)
        self.cache_dir = config['storage']['data_dir']
        
        # Setup logging first
        self.setup_logging()
        
        # Get list of coins from artifacts directory
        self.coins = self.get_available_coins()
        
        # Initialize Binance client
        use_testnet = config.get('exchange', {}).get('use_testnet', True)
        
        if api_key and api_secret:
            if use_testnet:
                self.client = Client(api_key, api_secret, testnet=True)
            else:
                self.client = Client(api_key, api_secret)
            self.use_testnet = use_testnet
        else:
            # Use testnet for safety when no API keys provided
            self.client = Client(testnet=True)
            self.use_testnet = True
        
        # Data cache: {coin: {timeframe: DataFrame}}
        self.data_cache = {}
        self.cache_lock = threading.Lock()
        
        # Cache configuration
        self.max_cache_size = {
            '1h': 17520,  # 2 years of hourly data
            '4h': 4380,   # 2 years of 4-hour data
            '1d': 730     # 2 years of daily data
        }
        
        # Initialize cache
        self.initialize_cache()
    
    def get_available_coins(self) -> List[str]:
        """Get list of available coins from artifacts directory."""
        artifacts_dir = self.config['storage']['artifacts_dir']
        if os.path.exists(artifacts_dir):
            coins = [d for d in os.listdir(artifacts_dir) 
                    if os.path.isdir(os.path.join(artifacts_dir, d))]
            self.logger.info(f"Found {len(coins)} coins: {coins}")
            return coins
        else:
            self.logger.warning("Artifacts directory not found, using default coins")
            return ['ETHUSDC', 'ADAUSDC', 'BTCUSDC', 'DOGEUSDC', 'XRPUSDC']
    
    def setup_logging(self):
        """Setup logging for data manager."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multi_coin_data_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MultiCoinDataManager')
    
    def initialize_cache(self):
        """Initialize data cache with historical data for all coins."""
        self.logger.info("Initializing multi-coin data cache...")
        
        for coin in self.coins:
            self.logger.info(f"Loading data for {coin}")
            self.data_cache[coin] = {}
            
            for timeframe in self.timeframes:
                self.logger.info(f"  Loading {timeframe} data for {coin}")
                
                # Load existing data if available (new folder structure)
                cache_file = f"{self.cache_dir}/{coin}_{timeframe}.csv"
                data_file = f"{self.config['storage']['data_dir']}/{coin}/{timeframe}.csv"
                
                if os.path.exists(cache_file):
                    self.logger.info(f"    Loading existing {timeframe} data from cache")
                    df = pd.read_csv(cache_file)
                    df['open_time'] = pd.to_datetime(df['open_time'])
                    self.data_cache[coin][timeframe] = df
                elif os.path.exists(data_file):
                    self.logger.info(f"    Loading existing {timeframe} data from data folder")
                    df = pd.read_csv(data_file)
                    df['open_time'] = pd.to_datetime(df['open_time'])
                    self.data_cache[coin][timeframe] = df
                else:
                    # Download historical data
                    self.logger.info(f"    Downloading historical {timeframe} data")
                    df = self.download_historical_data(coin, timeframe)
                    self.data_cache[coin][timeframe] = df
                    self.save_cache_to_file(coin, timeframe)
        
        self.logger.info("Multi-coin data cache initialization complete")
        self.log_cache_status()
    
    def download_historical_data(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        Download historical data for a specific coin and timeframe.
        
        Args:
            coin: Coin symbol (e.g., 'ETHUSDC')
            timeframe: Timeframe string (1h, 4h, 1d)
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * self.lookback_years)
            
            self.logger.info(f"    Downloading {timeframe} data for {coin} from {start_date} to {end_date}")
            
            # Download data in chunks to avoid API limits
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                try:
                    # Download chunk (max 1000 candles per request)
                    klines = self.client.get_historical_klines(
                        coin,
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
                    self.logger.error(f"    Binance API error for {coin} {timeframe}: {e}")
                    time.sleep(1)
                    continue
                except Exception as e:
                    self.logger.error(f"    Error downloading {coin} {timeframe} data: {e}")
                    time.sleep(1)
                    continue
            
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                self.logger.info(f"    Downloaded {len(df)} {timeframe} candles for {coin}")
                return df
            else:
                self.logger.error(f"    No data downloaded for {coin} {timeframe}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"    Error downloading historical data for {coin} {timeframe}: {e}")
            return pd.DataFrame()
    
    def save_cache_to_file(self, coin: str, timeframe: str):
        """Save cache data to file."""
        with self.cache_lock:
            if coin in self.data_cache and timeframe in self.data_cache[coin]:
                cache_file = f"{self.cache_dir}/{coin}_{timeframe}.csv"
                self.data_cache[coin][timeframe].to_csv(cache_file, index=False)
                self.logger.info(f"    Saved {coin} {timeframe} cache to {cache_file}")
                
                # Also save to data folder for backup
                data_file = f"{self.config['storage']['data_dir']}/{coin}/{timeframe}.csv"
                os.makedirs(os.path.dirname(data_file), exist_ok=True)
                self.data_cache[coin][timeframe].to_csv(data_file, index=False)
    
    def get_latest_data(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        Get the latest data for a specific coin and timeframe.
        
        Args:
            coin: Coin symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame with latest data
        """
        with self.cache_lock:
            if coin in self.data_cache and timeframe in self.data_cache[coin]:
                return self.data_cache[coin][timeframe].copy()
            else:
                return pd.DataFrame()
    
    def update_cache(self):
        """Update cache with latest data from exchange for all coins."""
        self.logger.info("Updating multi-coin data cache...")
        
        for coin in self.coins:
            self.logger.info(f"Updating data for {coin}")
            
            for timeframe in self.timeframes:
                try:
                    # Check if this timeframe should be updated
                    if not self.should_update_timeframe(timeframe):
                        self.logger.info(f"    Skipping {timeframe} update for {coin} (not due yet)")
                        continue
                    
                    # Get only new candles
                    new_candles = self.get_new_candles_only(coin, timeframe)
                    
                    if not new_candles.empty:
                        with self.cache_lock:
                            if coin in self.data_cache and timeframe in self.data_cache[coin]:
                                # Merge with existing data
                                existing_df = self.data_cache[coin][timeframe]
                                
                                # Ensure timezone consistency
                                if existing_df['open_time'].dt.tz is None:
                                    existing_df['open_time'] = existing_df['open_time'].dt.tz_localize('UTC')
                                if new_candles['open_time'].dt.tz is None:
                                    new_candles['open_time'] = new_candles['open_time'].dt.tz_localize('UTC')
                                
                                # Remove duplicates and sort
                                combined_df = pd.concat([existing_df, new_candles], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                                
                                # Maintain cache size limit
                                max_size = self.max_cache_size.get(timeframe, 10000)
                                if len(combined_df) > max_size:
                                    combined_df = combined_df.tail(max_size)
                                
                                self.data_cache[coin][timeframe] = combined_df
                                self.logger.info(f"    Updated {coin} {timeframe} cache: {len(combined_df)} candles (+{len(new_candles)} new)")
                            else:
                                self.data_cache[coin][timeframe] = new_candles
                                self.logger.info(f"    Initialized {coin} {timeframe} cache: {len(new_candles)} candles")
                        
                        # Save to file
                        self.save_cache_to_file(coin, timeframe)
                    else:
                        self.logger.info(f"    No new {timeframe} data for {coin}")
                    
                except Exception as e:
                    self.logger.error(f"    Error updating {coin} {timeframe} cache: {e}")
                    # Retry once
                    try:
                        self.logger.info(f"    Retrying {coin} {timeframe} update...")
                        time.sleep(1)
                        new_candles = self.get_new_candles_only(coin, timeframe)
                        if not new_candles.empty:
                            with self.cache_lock:
                                if coin in self.data_cache and timeframe in self.data_cache[coin]:
                                    existing_df = self.data_cache[coin][timeframe]
                                    if existing_df['open_time'].dt.tz is None:
                                        existing_df['open_time'] = existing_df['open_time'].dt.tz_localize('UTC')
                                    if new_candles['open_time'].dt.tz is None:
                                        new_candles['open_time'] = new_candles['open_time'].dt.tz_localize('UTC')
                                    
                                    combined_df = pd.concat([existing_df, new_candles], ignore_index=True)
                                    combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                                    
                                    max_size = self.max_cache_size.get(timeframe, 10000)
                                    if len(combined_df) > max_size:
                                        combined_df = combined_df.tail(max_size)
                                    
                                    self.data_cache[coin][timeframe] = combined_df
                                    self.logger.info(f"    Retry successful: Updated {coin} {timeframe} cache: {len(combined_df)} candles (+{len(new_candles)} new)")
                                    self.save_cache_to_file(coin, timeframe)
                    except Exception as retry_e:
                        self.logger.error(f"    Retry failed for {coin} {timeframe}: {retry_e}")
        
        self.logger.info("Multi-coin data cache update complete")
        self.log_cache_status()
    
    def download_latest_data(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        Download the latest data for a specific coin and timeframe.
        
        Args:
            coin: Coin symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame with latest data
        """
        try:
            # Get latest klines
            klines = self.client.get_klines(
                symbol=coin,
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
            self.logger.error(f"    Error downloading latest {coin} {timeframe} data: {e}")
            return pd.DataFrame()
    
    def get_new_candles_only(self, coin: str, timeframe: str) -> pd.DataFrame:
        """
        Get only new candles that aren't already in the cache.
        
        Args:
            coin: Coin symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame with only new candles
        """
        try:
            # Get latest data from exchange
            latest_data = self.download_latest_data(coin, timeframe)
            
            if latest_data.empty:
                return pd.DataFrame()
            
            # Get existing data from cache
            with self.cache_lock:
                if coin in self.data_cache and timeframe in self.data_cache[coin]:
                    existing_df = self.data_cache[coin][timeframe]
                    
                    if existing_df.empty:
                        # No existing data, return all latest data
                        return latest_data
                    
                    # Get the latest timestamp from existing data as a string for comparison
                    latest_existing_time_str = existing_df['open_time'].max().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Convert latest_data timestamps to strings for comparison
                    latest_data_copy = latest_data.copy()
                    latest_data_copy['time_str'] = latest_data_copy['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Filter for only new candles using string comparison
                    new_candles = latest_data_copy[latest_data_copy['time_str'] > latest_existing_time_str]
                    
                    # Remove the temporary time_str column
                    if not new_candles.empty:
                        new_candles = new_candles.drop('time_str', axis=1)
                        self.logger.info(f"    Found {len(new_candles)} new {timeframe} candles for {coin}")
                        return new_candles
                    else:
                        self.logger.info(f"    No new {timeframe} candles for {coin}")
                        return pd.DataFrame()
                else:
                    # No existing cache, return all latest data
                    return latest_data
                    
        except Exception as e:
            self.logger.error(f"    Error getting new candles for {coin} {timeframe}: {e}")
            return pd.DataFrame()
    
    def should_update_timeframe(self, timeframe: str) -> bool:
        """
        Check if a timeframe should be updated based on current time.
        
        Args:
            timeframe: Timeframe string (1h, 4h, 1d)
            
        Returns:
            True if timeframe should be updated
        """
        now = datetime.now()
        
        if timeframe == '1h':
            # Update every hour
            return True
        elif timeframe == '4h':
            # Update every 4 hours (at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)
            return True
        elif timeframe == '1d':
            # Update daily at midnight
            return True
        else:
            return True
    
    def get_data_for_model(self, coin: str, timeframe: str, lookback_bars: int = 100) -> pd.DataFrame:
        """
        Get data for model prediction with specified lookback.
        
        Args:
            coin: Coin symbol
            timeframe: Timeframe string
            lookback_bars: Number of bars to look back
            
        Returns:
            DataFrame with recent data for model
        """
        with self.cache_lock:
            if coin in self.data_cache and timeframe in self.data_cache[coin]:
                df = self.data_cache[coin][timeframe].copy()
                return df.tail(lookback_bars)
            else:
                return pd.DataFrame()
    
    def is_cache_ready(self) -> bool:
        """Check if cache is ready for trading."""
        required_timeframes = ['1h', '4h', '1d']
        
        for coin in self.coins:
            for tf in required_timeframes:
                if (coin not in self.data_cache or 
                    tf not in self.data_cache[coin] or 
                    len(self.data_cache[coin][tf]) < 100):
                    return False
        
        return True
    
    def get_cache_status(self) -> dict:
        """Get current cache status for all coins."""
        status = {}
        for coin in self.coins:
            status[coin] = {}
            for timeframe in self.timeframes:
                if coin in self.data_cache and timeframe in self.data_cache[coin]:
                    df = self.data_cache[coin][timeframe]
                    status[coin][timeframe] = {
                        'count': len(df),
                        'latest_time': df['open_time'].iloc[-1] if len(df) > 0 else None,
                        'oldest_time': df['open_time'].iloc[0] if len(df) > 0 else None
                    }
                else:
                    status[coin][timeframe] = {'count': 0, 'latest_time': None, 'oldest_time': None}
        
        return status
    
    def log_cache_status(self):
        """Log current cache status."""
        status = self.get_cache_status()
        self.logger.info("Cache status:")
        for coin, timeframes in status.items():
            self.logger.info(f"  {coin}:")
            for tf, info in timeframes.items():
                self.logger.info(f"    {tf}: {info['count']} candles")
    
    def start_hourly_updates(self):
        """Start hourly data updates in background thread."""
        def update_worker():
            while True:
                try:
                    # Wait until next hour + 2 minutes (to ensure new bars are available)
                    now = datetime.now()
                    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    # Add 2 minutes delay to ensure new bars are available from Binance
                    next_update = next_hour + timedelta(minutes=2)
                    wait_seconds = (next_update - now).total_seconds()
                    
                    self.logger.info(f"Waiting {wait_seconds/60:.1f} minutes until next hour + 2min for data update")
                    time.sleep(wait_seconds)
                    
                    # Update cache for all coins
                    self.logger.info("Starting scheduled data update...")
                    self.update_cache()
                    
                except Exception as e:
                    self.logger.error(f"Error in hourly update worker: {e}")
                    time.sleep(60)  # Wait 1 minute before retrying
        
        # Start background thread
        update_thread = threading.Thread(target=update_worker, daemon=True)
        update_thread.start()
        self.logger.info("Started hourly multi-coin data update worker")
    
    def force_update(self):
        """Force immediate cache update for all coins."""
        self.logger.info("Forcing immediate multi-coin cache update...")
        self.update_cache()
    
    def force_update_specific_coins(self, coins: List[str]):
        """
        Force immediate cache update for specific coins.
        
        Args:
            coins: List of coin symbols to update
        """
        self.logger.info(f"Forcing immediate cache update for specific coins: {coins}")
        
        for coin in coins:
            if coin not in self.coins:
                self.logger.warning(f"Coin {coin} not found in available coins")
                continue
                
            self.logger.info(f"Force updating data for {coin}")
            
            for timeframe in self.timeframes:
                try:
                    # Get only new candles
                    new_candles = self.get_new_candles_only(coin, timeframe)
                    
                    if not new_candles.empty:
                        with self.cache_lock:
                            if coin in self.data_cache and timeframe in self.data_cache[coin]:
                                # Merge with existing data
                                existing_df = self.data_cache[coin][timeframe]
                                
                                # Ensure timezone consistency
                                if existing_df['open_time'].dt.tz is None:
                                    existing_df['open_time'] = existing_df['open_time'].dt.tz_localize('UTC')
                                if new_candles['open_time'].dt.tz is None:
                                    new_candles['open_time'] = new_candles['open_time'].dt.tz_localize('UTC')
                                
                                # Remove duplicates and sort
                                combined_df = pd.concat([existing_df, new_candles], ignore_index=True)
                                combined_df = combined_df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                                
                                # Maintain cache size limit
                                max_size = self.max_cache_size.get(timeframe, 10000)
                                if len(combined_df) > max_size:
                                    combined_df = combined_df.tail(max_size)
                                
                                self.data_cache[coin][timeframe] = combined_df
                                self.logger.info(f"    Force updated {coin} {timeframe} cache: {len(combined_df)} candles (+{len(new_candles)} new)")
                            else:
                                self.data_cache[coin][timeframe] = new_candles
                                self.logger.info(f"    Force initialized {coin} {timeframe} cache: {len(new_candles)} candles")
                        
                        # Save to file
                        self.save_cache_to_file(coin, timeframe)
                    else:
                        self.logger.info(f"    No new {timeframe} data for {coin}")
                    
                except Exception as e:
                    self.logger.error(f"    Error force updating {coin} {timeframe} cache: {e}")
        
        self.logger.info("Force update for specific coins complete")
    
    def cleanup_old_cache_files(self):
        """Clean up old cache files to save disk space."""
        try:
            for coin in self.coins:
                for timeframe in self.timeframes:
                    cache_file = f"{self.cache_dir}/{coin}_{timeframe}.csv"
                    if os.path.exists(cache_file):
                        # Keep only the most recent cache file
                        # You could implement more sophisticated cleanup here
                        pass
        except Exception as e:
            self.logger.error(f"Error cleaning up cache files: {e}")

class LiveMultiCoinDataManager:
    """High-level interface for live multi-coin trading data management."""
    
    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        """Initialize live multi-coin data manager."""
        self.config = config
        self.data_manager = MultiCoinDataManager(config, api_key, api_secret)
        self.logger = logging.getLogger('LiveMultiCoinDataManager')
        
        # Ensure cache directory exists
        os.makedirs(config['storage']['data_dir'], exist_ok=True)
    
    def initialize(self) -> bool:
        """
        Initialize the data manager for live trading.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing live multi-coin data manager...")
            
            # Check if cache is ready
            if not self.data_manager.is_cache_ready():
                self.logger.error("Cache not ready for trading")
                return False
            
            # Start hourly updates
            self.data_manager.start_hourly_updates()
            
            self.logger.info("Live multi-coin data manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing live multi-coin data manager: {e}")
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
        return self.data_manager.get_data_for_model(coin, timeframe)
    
    def get_cache_status(self) -> dict:
        """Get current cache status for all coins."""
        return self.data_manager.get_cache_status()
    
    def force_update(self):
        """Force immediate data update for all coins."""
        self.data_manager.force_update()
    
    def force_update_specific_coins(self, coins: List[str]):
        """Force immediate data update for specific coins."""
        self.data_manager.force_update_specific_coins(coins)
    
    def is_ready(self) -> bool:
        """Check if data manager is ready for trading."""
        return self.data_manager.is_cache_ready()
    
    def get_available_coins(self) -> List[str]:
        """Get list of available coins."""
        return self.data_manager.coins

def main():
    """Test the multi-coin data manager."""
    import yaml
    
    # Load config
    with open('config_live.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data manager
    data_manager = LiveMultiCoinDataManager(config)
    
    # Initialize
    if data_manager.initialize():
        print("Multi-coin data manager initialized successfully!")
        
        # Show cache status
        status = data_manager.get_cache_status()
        for coin, timeframes in status.items():
            print(f"\n{coin}:")
            for tf, info in timeframes.items():
                print(f"  {tf}: {info['count']} candles, latest: {info['latest_time']}")
        
        # Keep running for testing
        try:
            while True:
                time.sleep(60)
                status = data_manager.get_cache_status()
                print(f"\nCache status update:")
                for coin, timeframes in status.items():
                    total_candles = sum(info['count'] for info in timeframes.values())
                    print(f"  {coin}: {total_candles} total candles")
        except KeyboardInterrupt:
            print("Stopping multi-coin data manager...")
    else:
        print("Failed to initialize multi-coin data manager")

if __name__ == "__main__":
    main()
