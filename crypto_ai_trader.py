#!/usr/bin/env python3
"""
AiETHTrader - AI-Powered Crypto Trading Bot
==========================================

This module implements an AI trading bot that uses PyTorch neural networks
to predict price movements and execute trades based on 1-hour charts with
4-hour and daily trend analysis.

Key Features:
- 1-hour trading signals with 4h/1d trend confirmation
- Buy and No-Trade signals only (no sell signals)
- 3% minimum gain requirement for trade validation
- Multi-timeframe technical analysis
"""

import os
import sys
import argparse
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
import warnings
warnings.filterwarnings('ignore')

class TradingModel(nn.Module):
    """PyTorch neural network for trading signal prediction."""
    
    def __init__(self, input_size, hidden_size=128, dropout=0.2):
        super(TradingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Buy (1) or No-Trade (0)
        )
    
    def forward(self, x):
        return self.network(x)

class FeatureEngineer:
    """Feature engineering for multi-timeframe analysis."""
    
    def __init__(self):
        self.scalers = {}
    
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

class DataProcessor:
    """Data processing and preparation for training."""
    
    def __init__(self, config):
        self.config = config
        self.feature_engineer = FeatureEngineer()
    
    def load_data(self):
        """Load and process data from CSV files."""
        symbol = self.config['symbol']
        data_dir = self.config['storage']['data_dir']
        
        # Load data for different timeframes
        df_1h = pd.read_csv(f"{data_dir}/{symbol}_1h.csv")
        df_4h = pd.read_csv(f"{data_dir}/{symbol}_4h.csv")
        df_1d = pd.read_csv(f"{data_dir}/{symbol}_1d.csv")
        
        print(f"Loaded data: {len(df_1h)} 1h bars, {len(df_4h)} 4h bars, {len(df_1d)} 1d bars")
        
        # Create multi-timeframe features
        df_combined = self.feature_engineer.create_multi_timeframe_features(df_1h, df_4h, df_1d)
        
        # Create labels
        df_combined = self.feature_engineer.create_labels(df_combined, min_gain_pct=3.0)
        
        return df_combined
    
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
        
        print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        print(f"Buy signals: {y.sum()}, No-trade signals: {(y == 0).sum()}")
        
        return X, y

class ModelTrainer:
    """Model training and evaluation."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X, y):
        """Train the model."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['train']['test_size'],
            random_state=self.config['train']['random_state'], stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test.values)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.config['train']['batch_size'], shuffle=True)
        
        # Initialize model
        self.model = TradingModel(
            input_size=X_train_scaled.shape[1],
            hidden_size=self.config['train']['hidden_size'],
            dropout=self.config['train']['dropout']
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])
        
        # Training loop
        self.model.train()
        for epoch in range(self.config['train']['epochs']):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            if (epoch + 1) % 5 == 0:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
            print(f'Test Accuracy: {test_accuracy:.4f}')
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test_tensor, test_predicted, target_names=['No-Trade', 'Buy']))
        
        return X_test, y_test, test_predicted
    
    def save_model(self, artifacts_dir):
        """Save the trained model and scaler."""
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), f"{artifacts_dir}/trading_model.pth")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, f"{artifacts_dir}/scaler.pkl")
        
        print(f"Model saved to {artifacts_dir}/")

class Backtester:
    """Backtesting engine for strategy validation."""
    
    def __init__(self, config):
        self.config = config
    
    def backtest(self, df, model, scaler, threshold=0.5):
        """Run backtest on the strategy."""
        # Prepare features
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
        
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0)
        
        # Scale features
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            buy_prob = probabilities[:, 1].numpy()
        
        # Generate signals
        signals = np.where(buy_prob >= threshold, 1, 0)
        
        # Calculate returns
        df['signal'] = signals
        df['return'] = df['close'].shift(-1) / df['open'].shift(-1) - 1
        df['strategy_return'] = df['signal'] * df['return']
        
        # Calculate metrics
        total_trades = signals.sum()
        winning_trades = (df['strategy_return'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = df['strategy_return'].sum()
        avg_return_per_trade = df['strategy_return'].mean()
        
        print(f"\nBacktest Results:")
        print(f"Total trades: {total_trades}")
        print(f"Winning trades: {winning_trades}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Total return: {total_return:.2%}")
        print(f"Average return per trade: {avg_return_per_trade:.2%}")
        
        return df

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AiETHTrader - AI Crypto Trading Bot')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("AiETHTrader - AI Crypto Trading Bot")
    print("=" * 50)
    
    if args.train:
        print("Starting model training...")
        
        # Initialize components
        data_processor = DataProcessor(config)
        trainer = ModelTrainer(config)
        backtester = Backtester(config)
        
        # Load and process data
        df = data_processor.load_data()
        X, y = data_processor.prepare_features(df)
        
        # Train model
        X_test, y_test, predictions = trainer.train(X, y)
        
        # Save model
        trainer.save_model(config['storage']['artifacts_dir'])
        
        # Run backtest
        backtest_df = backtester.backtest(df, trainer.model, trainer.scaler, 
                                        threshold=config['backtest']['threshold'])
        
        print("Training completed!")
    
    elif args.live:
        print("Live trading not implemented yet.")
        print("Please train the model first with --train")
    
    else:
        print("Please specify --train or --live")

if __name__ == "__main__":
    main()
