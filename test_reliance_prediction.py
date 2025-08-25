#!/usr/bin/env python3
"""
Test RELIANCE Prediction
Simple test to verify RELIANCE stock prediction functionality
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

def test_reliance_prediction():
    """Test RELIANCE stock prediction."""
    print("🎯 Testing RELIANCE Stock Prediction")
    print("=" * 50)
    
    ticker = "RELIANCE"
    
    # Check if data exists
    data_file = f"data/{ticker}_partA_partC_enhanced.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Load data
    print(f"📊 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"📈 Data shape: {df.shape}")
    
    # Get current price
    current_price = float(df.iloc[-1]['Close'])
    print(f"💰 Current Price: ₹{current_price:.2f}")
    
    # Check for models
    model_files = []
    if os.path.exists('models'):
        model_files = [f for f in os.listdir('models') if f.startswith(ticker)]
    
    print(f"📁 Found {len(model_files)} models:")
    for model in model_files:
        print(f"   • {model}")
    
    # Test ensemble predictions
    print("\n🤖 Testing Ensemble Predictions...")
    
    # Prepare features
    essential_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    basic_indicators = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD']
    
    available_essential = [col for col in essential_cols if col in df.columns]
    available_indicators = [col for col in basic_indicators if col in df.columns]
    
    if not available_essential:
        print("   ❌ No essential columns available")
        return
    
    feature_cols = available_essential + available_indicators
    X = df[feature_cols].dropna()
    
    if len(X) == 0:
        print("   ❌ No valid data for predictions")
        return
    
    # Get last data point
    last_data = X.iloc[-1:].values
    print(f"   📊 Using {len(feature_cols)} features for prediction")
    
    # Test each model
    predictions = []
    
    # Random Forest
    rf_model_path = f"models/{ticker}_random_forest_model.pkl"
    if os.path.exists(rf_model_path):
        try:
            rf_model = joblib.load(rf_model_path)
            rf_pred = rf_model.predict(last_data)[0]
            predictions.append(('Random Forest', rf_pred))
            print(f"   ✅ Random Forest: ₹{rf_pred:.2f}")
        except Exception as e:
            print(f"   ❌ Random Forest error: {e}")
    
    # Gradient Boost
    gb_model_path = f"models/{ticker}_gradient_boost_model.pkl"
    if os.path.exists(gb_model_path):
        try:
            gb_model = joblib.load(gb_model_path)
            gb_pred = gb_model.predict(last_data)[0]
            predictions.append(('Gradient Boost', gb_pred))
            print(f"   ✅ Gradient Boost: ₹{gb_pred:.2f}")
        except Exception as e:
            print(f"   ❌ Gradient Boost error: {e}")
    
    # SVR
    svr_model_path = f"models/{ticker}_svr_model.pkl"
    if os.path.exists(svr_model_path):
        try:
            svr_model = joblib.load(svr_model_path)
            svr_pred = svr_model.predict(last_data)[0]
            predictions.append(('SVR', svr_pred))
            print(f"   ✅ SVR: ₹{svr_pred:.2f}")
        except Exception as e:
            print(f"   ❌ SVR error: {e}")
    
    # Display results
    if predictions:
        print(f"\n🎯 PREDICTION RESULTS")
        print("=" * 50)
        print(f"📊 Stock: {ticker}")
        print(f"💰 Current Price: ₹{current_price:.2f}")
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for model_name, pred in predictions:
            change = pred - current_price
            change_pct = (change / current_price) * 100
            direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"   • {model_name}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Calculate average
        avg_prediction = sum([pred for _, pred in predictions]) / len(predictions)
        avg_change = avg_prediction - current_price
        avg_change_pct = (avg_change / current_price) * 100
        avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
        
        print(f"\n📊 Average Prediction: ₹{avg_prediction:.2f} ({avg_direction} {avg_change_pct:+.2f}%)")
        
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
        
        print(f"\n✅ RELIANCE prediction test completed successfully!")
    else:
        print("❌ No predictions generated")

if __name__ == "__main__":
    test_reliance_prediction()
