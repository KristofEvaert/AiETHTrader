#!/usr/bin/env python3
"""
Train Separate Models for All Coins
===================================

This script trains individual AI models for each cryptocurrency
in the data folder, creating coin-specific trading models.

Each model will be optimized for its specific coin's price patterns,
volatility, and market behavior.
"""

import sys
import os
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ai_trader import TradingModel, FeatureEngineer, DataProcessor, ModelTrainer

class MultiCoinTrainer:
    """Trainer for multiple coins with individual models."""
    
    def __init__(self, config):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.results = {}
        
    def get_available_coins(self, data_dir):
        """Get list of available coins from data directory."""
        pattern = f"{data_dir}/*_1h.csv"
        files = glob.glob(pattern)
        
        coins = []
        for file in files:
            coin = os.path.basename(file).replace('_1h.csv', '')
            coins.append(coin)
        
        return sorted(coins)
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe."""
        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return df
    
    def create_multi_timeframe_features(self, df_1h, df_4h, df_1d):
        """Create features using multiple timeframes."""
        # Align timeframes
        df_1h['datetime'] = pd.to_datetime(df_1h['open_time'])
        df_4h['datetime'] = pd.to_datetime(df_4h['open_time'])
        df_1d['datetime'] = pd.to_datetime(df_1d['open_time'])
        
        # Add technical indicators to each timeframe
        df_1h = self.add_technical_indicators(df_1h)
        df_4h = self.add_technical_indicators(df_4h)
        df_1d = self.add_technical_indicators(df_1d)
        
        # Create 4h trend features
        df_4h['4h_trend'] = np.where(df_4h['close'] > df_4h['sma_20'], 1, 0)
        df_4h['4h_momentum'] = df_4h['close'].pct_change(1)
        
        # Create 1d trend features
        df_1d['1d_trend'] = np.where(df_1d['close'] > df_1d['sma_20'], 1, 0)
        df_1d['1d_momentum'] = df_1d['close'].pct_change(1)
        
        # Merge timeframes
        df_1h['hour'] = df_1h['datetime'].dt.hour
        df_1h['day_of_week'] = df_1h['datetime'].dt.dayofweek
        
        # Forward fill 4h and 1d data to align with 1h
        df_4h_aligned = df_4h.set_index('datetime').reindex(df_1h['datetime'], method='ffill')
        df_1d_aligned = df_1d.set_index('datetime').reindex(df_1h['datetime'], method='ffill')
        
        # Add 4h features
        for col in ['4h_trend', '4h_momentum', 'rsi', 'macd', 'bb_position']:
            if col in df_4h_aligned.columns:
                df_1h[f'4h_{col}'] = df_4h_aligned[col].values
        
        # Add 1d features
        for col in ['1d_trend', '1d_momentum', 'rsi', 'macd', 'bb_position']:
            if col in df_1d_aligned.columns:
                df_1h[f'1d_{col}'] = df_1d_aligned[col].values
        
        return df_1h
    
    def create_labels(self, df, min_gain_pct=3.0):
        """Create labels for buy/no-trade signals with minimum gain requirement."""
        # Calculate next bar's return
        df['next_return'] = df['close'].shift(-1) / df['open'].shift(-1) - 1
        
        # Create labels: 1 for buy (if gain >= min_gain_pct), 0 for no-trade
        df['label'] = np.where(df['next_return'] >= min_gain_pct / 100, 1, 0)
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training."""
        # Select feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_5', 'sma_10', 'sma_20', 'ema_5', 'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_width', 'bb_position', 'volume_ratio',
            'price_change_1', 'price_change_3', 'price_change_5',
            'volatility', 'hl_spread', 'hour', 'day_of_week',
            '4h_4h_trend', '4h_4h_momentum', '4h_rsi', '4h_macd', '4h_bb_position',
            '1d_1d_trend', '1d_1d_momentum', '1d_rsi', '1d_macd', '1d_bb_position'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0)
        y = df['label'].fillna(0)
        
        # Remove rows with NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def train_coin_model(self, coin, data_dir):
        """Train a model for a specific coin."""
        print(f"\n{'='*60}")
        print(f"TRAINING MODEL FOR {coin}")
        print(f"{'='*60}")
        
        try:
            # Load data for different timeframes
            df_1h = pd.read_csv(f"{data_dir}/{coin}_1h.csv")
            df_4h = pd.read_csv(f"{data_dir}/{coin}_4h.csv")
            df_1d = pd.read_csv(f"{data_dir}/{coin}_1d.csv")
            
            print(f"Loaded data: {len(df_1h)} 1h bars, {len(df_4h)} 4h bars, {len(df_1d)} 1d bars")
            
            # Create multi-timeframe features
            df_combined = self.create_multi_timeframe_features(df_1h, df_4h, df_1d)
            
            # Create labels
            df_combined = self.create_labels(df_combined, min_gain_pct=3.0)
            
            # Prepare features
            X, y = self.prepare_features(df_combined)
            
            print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
            print(f"Buy signals: {y.sum()}, No-trade signals: {(y == 0).sum()}")
            
            if y.sum() == 0:
                print(f"WARNING: No buy signals found for {coin}. Skipping training.")
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['train']['test_size'],
                random_state=self.config['train']['random_state'], stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.LongTensor(y_test.values)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'], shuffle=True)
            
            # Initialize model
            model = TradingModel(
                input_size=X_train_scaled.shape[1],
                hidden_size=self.config['train']['hidden_size'],
                dropout=self.config['train']['dropout']
            )
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config['train']['lr'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
            
            # Training loop
            model.train()
            best_loss = float('inf')
            training_history = {'loss': [], 'accuracy': []}
            
            for epoch in range(self.config['train']['epochs']):
                total_loss = 0
                correct = 0
                total = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                avg_loss = total_loss / len(train_loader)
                accuracy = 100 * correct / total
                
                training_history['loss'].append(avg_loss)
                training_history['accuracy'].append(accuracy)
                
                scheduler.step(avg_loss)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Save best model
                    os.makedirs(f"{self.config['storage']['artifacts_dir']}/{coin}", exist_ok=True)
                    torch.save(model.state_dict(), f"{self.config['storage']['artifacts_dir']}/{coin}/best_model.pth")
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # Load best model
            model.load_state_dict(torch.load(f"{self.config['storage']['artifacts_dir']}/{coin}/best_model.pth"))
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, test_predicted = torch.max(test_outputs, 1)
                test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
                print(f'Test Accuracy: {test_accuracy:.4f}')
                
                # Classification report
                print("\nClassification Report:")
                print(classification_report(y_test_tensor, test_predicted, target_names=['No-Trade', 'Buy']))
            
            # Save model and scaler
            torch.save(model.state_dict(), f"{self.config['storage']['artifacts_dir']}/{coin}/trading_model.pth")
            joblib.dump(scaler, f"{self.config['storage']['artifacts_dir']}/{coin}/scaler.pkl")
            
            # Save training history
            history_df = pd.DataFrame(training_history)
            history_df.to_csv(f"{self.config['storage']['artifacts_dir']}/{coin}/training_history.csv", index=False)
            
            print(f"Model saved to {self.config['storage']['artifacts_dir']}/{coin}/")
            
            return {
                'coin': coin,
                'model': model,
                'scaler': scaler,
                'test_accuracy': test_accuracy,
                'buy_signals': y.sum(),
                'total_bars': len(y),
                'buy_signal_rate': y.sum() / len(y)
            }
            
        except Exception as e:
            print(f"Error training {coin}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all_coins(self, data_dir):
        """Train models for all available coins."""
        coins = self.get_available_coins(data_dir)
        print(f"Found coins to train: {coins}")
        
        results = []
        
        for coin in coins:
            result = self.train_coin_model(coin, data_dir)
            if result:
                results.append(result)
                self.results[coin] = result
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of all training results."""
        print(f"\n{'='*80}")
        print("MULTI-COIN TRAINING SUMMARY")
        print(f"{'='*80}")
        
        if not results:
            print("No models were successfully trained!")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'Coin': result['coin'],
                'Test Accuracy': f"{result['test_accuracy']:.1%}",
                'Buy Signals': result['buy_signals'],
                'Total Bars': result['total_bars'],
                'Buy Signal Rate': f"{result['buy_signal_rate']:.2%}",
                'Model Status': 'Trained'
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Overall statistics
        avg_accuracy = np.mean([r['test_accuracy'] for r in results])
        total_buy_signals = sum(r['buy_signals'] for r in results)
        total_bars = sum(r['total_bars'] for r in results)
        
        print(f"\nOVERALL STATISTICS:")
        print(f"Models Successfully Trained: {len(results)}")
        print(f"Average Test Accuracy: {avg_accuracy:.1%}")
        print(f"Total Buy Signals Across All Coins: {total_buy_signals}")
        print(f"Total Bars Processed: {total_bars}")
        
        # Best and worst performers
        best_coin = max(results, key=lambda x: x['test_accuracy'])
        worst_coin = min(results, key=lambda x: x['test_accuracy'])
        
        print(f"\nBEST PERFORMER: {best_coin['coin']} ({best_coin['test_accuracy']:.1%} accuracy)")
        print(f"WORST PERFORMER: {worst_coin['coin']} ({worst_coin['test_accuracy']:.1%} accuracy)")
        
        return summary_df

def main():
    """Main function to train models for all coins."""
    print("Multi-Coin Model Training")
    print("=" * 60)
    print("Training individual AI models for each cryptocurrency")
    print("Each model will be optimized for its specific coin's patterns")
    print("=" * 60)
    
    # Load configuration
    with open('config_1h_trading.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create artifacts directory
    os.makedirs(config['storage']['artifacts_dir'], exist_ok=True)
    
    # Initialize trainer
    trainer = MultiCoinTrainer(config)
    
    # Train models for all coins
    results = trainer.train_all_coins(config['storage']['data_dir'])
    
    # Create summary report
    summary_df = trainer.create_summary_report(results)
    
    # Save summary
    summary_df.to_csv('multi_coin_training_summary.csv', index=False)
    print(f"\nTraining summary saved to: multi_coin_training_summary.csv")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Models saved in: {config['storage']['artifacts_dir']}/")
    print("Each coin has its own folder with:")
    print("- trading_model.pth (PyTorch model)")
    print("- scaler.pkl (feature scaler)")
    print("- training_history.csv (training progress)")

if __name__ == "__main__":
    main()
