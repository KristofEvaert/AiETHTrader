#!/usr/bin/env python3
"""
1-Hour Trading Model Training Script
===================================

This script trains a specialized AI model for 1-hour trading with the following features:
- Uses 1h bars as primary timeframe
- Incorporates 4h and 1d trends for signal confirmation
- Only generates BUY and NO-TRADE signals (no sell signals)
- Requires minimum 3% gain for trade validation
- Optimized for hourly trading cycles

Usage:
    python train_1h_model.py
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
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_ai_trader import TradingModel, FeatureEngineer, DataProcessor, ModelTrainer, Backtester

class EnhancedModelTrainer(ModelTrainer):
    """Enhanced trainer with additional analysis and visualization."""
    
    def __init__(self, config):
        super().__init__(config)
        self.training_history = {'loss': [], 'accuracy': []}
    
    def train(self, X, y):
        """Enhanced training with history tracking."""
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
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
            
            avg_loss = total_loss / len(train_loader)
            accuracy = 100 * correct / total
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(accuracy)
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save best model
                torch.save(self.model.state_dict(), f"{self.config['storage']['artifacts_dir']}/best_model.pth")
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config["train"]["epochs"]}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Load best model
        self.model.load_state_dict(torch.load(f"{self.config['storage']['artifacts_dir']}/best_model.pth"))
        
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
            
            # Confusion matrix
            cm = confusion_matrix(y_test_tensor, test_predicted)
            print("\nConfusion Matrix:")
            print(cm)
        
        return X_test, y_test, test_predicted
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.config['storage']['artifacts_dir']}/training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

def analyze_data_distribution(df):
    """Analyze the distribution of buy signals and market conditions."""
    print("\n" + "="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    # Basic statistics
    total_bars = len(df)
    buy_signals = df['label'].sum()
    no_trade_signals = (df['label'] == 0).sum()
    
    print(f"Total 1h bars: {total_bars:,}")
    print(f"Buy signals: {buy_signals:,} ({buy_signals/total_bars:.2%})")
    print(f"No-trade signals: {no_trade_signals:,} ({no_trade_signals/total_bars:.2%})")
    
    # Market condition analysis
    df['market_trend'] = np.where(df['close'] > df['sma_20'], 'Bullish', 'Bearish')
    df['volatility_level'] = pd.cut(df['volatility'], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Buy signal distribution by market conditions
    print("\nBuy Signal Distribution by Market Conditions:")
    trend_analysis = df.groupby(['market_trend', 'volatility_level'])['label'].agg(['count', 'sum', 'mean']).round(3)
    print(trend_analysis)
    
    # Time-based analysis
    df['hour'] = pd.to_datetime(df['open_time']).dt.hour
    hourly_analysis = df.groupby('hour')['label'].agg(['count', 'sum', 'mean']).round(3)
    print("\nBuy Signal Distribution by Hour:")
    print(hourly_analysis)
    
    return df

def main():
    """Main training function."""
    print("1-Hour Trading Model Training")
    print("=" * 50)
    print("Features:")
    print("- 1h primary timeframe with 4h/1d trend analysis")
    print("- Buy and No-Trade signals only (no sell signals)")
    print("- 3% minimum gain requirement")
    print("- Optimized for hourly trading cycles")
    print("=" * 50)
    
    # Load configuration
    with open('config_1h_trading.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create artifacts directory
    os.makedirs(config['storage']['artifacts_dir'], exist_ok=True)
    
    # Initialize components
    data_processor = DataProcessor(config)
    trainer = EnhancedModelTrainer(config)
    backtester = Backtester(config)
    
    try:
        # Load and process data
        print("\nLoading and processing data...")
        df = data_processor.load_data()
        
        # Analyze data distribution
        df = analyze_data_distribution(df)
        
        # Prepare features
        print("\nPreparing features...")
        X, y = data_processor.prepare_features(df)
        
        # Train model
        print("\nTraining model...")
        X_test, y_test, predictions = trainer.train(X, y)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save model
        print("\nSaving model...")
        trainer.save_model(config['storage']['artifacts_dir'])
        
        # Run backtest
        print("\nRunning backtest...")
        backtest_df = backtester.backtest(df, trainer.model, trainer.scaler, 
                                        threshold=config['backtest']['threshold'])
        
        # Save backtest results
        backtest_df.to_csv(f"{config['storage']['artifacts_dir']}/backtest_results.csv", index=False)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model saved to: {config['storage']['artifacts_dir']}/")
        print(f"Backtest results saved to: {config['storage']['artifacts_dir']}/backtest_results.csv")
        print(f"Training history plot saved to: {config['storage']['artifacts_dir']}/training_history.png")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
