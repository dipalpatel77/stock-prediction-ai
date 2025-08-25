#!/usr/bin/env python3
"""
Test Prediction Display
Simple test to verify prediction display functionality
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def test_prediction_display():
    """Test the prediction display functionality."""
    print("🔮 Testing Prediction Display")
    print("=" * 50)
    
    # Check if we have trained models
    ticker = "AAPL"
    model_files = []
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.startswith(ticker)]
    
    print(f"📁 Found {len(model_files)} trained models for {ticker}:")
    for model in model_files:
        print(f"   • {model}")
    
    # Check if we have enhanced data
    data_file = f"data/{ticker}_partA_partC_enhanced.csv"
    if os.path.exists(data_file):
        print(f"\n✅ Enhanced data file found: {data_file}")
        
        # Load and display sample data
        df = pd.read_csv(data_file)
        print(f"📊 Data shape: {df.shape}")
        print(f"📊 Columns: {len(df.columns)}")
        
        # Get current price
        if len(df.columns) > 1 and df.columns[1] == 'Price':
            current_price = float(df.iloc[-1]['Close'])
        else:
            current_price = float(df.iloc[-1]['Close'])
        
        print(f"📈 Current Price: ${current_price:.2f}")
        
        # Simulate predictions
        print("\n🎯 Simulated Predictions:")
        print("=" * 30)
        
        # Ensemble predictions
        ensemble_predictions = [
            ('Random Forest', current_price * 1.02),
            ('Gradient Boost', current_price * 0.98)
        ]
        
        print("🤖 Ensemble Model Predictions:")
        for model_name, pred in ensemble_predictions:
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • {model_name}: ${pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Enhanced predictions
        enhanced_predictions = [
            current_price * 1.01,
            current_price * 1.03,
            current_price * 0.99
        ]
        
        print(f"\n🚀 Enhanced Model Predictions (3 days):")
        for i, pred in enumerate(enhanced_predictions, 1):
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • Day {i}: ${pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Calculate average prediction
        all_predictions = [pred for _, pred in ensemble_predictions] + enhanced_predictions
        avg_prediction = sum(all_predictions) / len(all_predictions)
        avg_change = avg_prediction - current_price
        avg_change_pct = (avg_change / current_price) * 100
        avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
        
        print(f"\n📊 Average Prediction: ${avg_prediction:.2f} ({avg_direction} {avg_change_pct:+.2f}%)")
        
        # Trading recommendation
        if avg_change_pct > 2:
            recommendation = "🟢 BUY - Strong upward momentum expected"
        elif avg_change_pct > 0.5:
            recommendation = "🟡 BUY - Moderate upward potential"
        elif avg_change_pct < -2:
            recommendation = "🔴 SELL - Strong downward pressure expected"
        elif avg_change_pct < -0.5:
            recommendation = "🟠 SELL - Moderate downward potential"
        else:
            recommendation = "⚪ HOLD - Stable price movement expected"
        
        print(f"\n💡 Trading Recommendation: {recommendation}")
        
        print("\n✅ Prediction display test completed successfully!")
        
    else:
        print(f"❌ Enhanced data file not found: {data_file}")

if __name__ == "__main__":
    test_prediction_display()
