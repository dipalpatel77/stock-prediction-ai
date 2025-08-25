#!/usr/bin/env python3
"""
Run RELIANCE Prediction
Direct prediction using existing models and data
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

def main():
    """Run RELIANCE prediction."""
    print("🎯 RELIANCE Stock Price Prediction")
    print("=" * 60)
    
    ticker = "RELIANCE"
    
    # Load data
    data_file = f"data/{ticker}_partA_partC_enhanced.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    df = pd.read_csv(data_file)
    current_price = float(df.iloc[-1]['Close'])
    
    print(f"📊 Stock: {ticker}")
    print(f"💰 Current Price: ₹{current_price:.2f}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate predictions
    predictions = []
    
    # Prepare features
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'SMA_10', 'SMA_20', 'RSI_14', 'MACD']
    available_cols = [col for col in feature_cols if col in df.columns]
    
    if len(available_cols) < 5:
        print("❌ Insufficient features for prediction")
        return
    
    X = df[available_cols].dropna()
    if len(X) == 0:
        print("❌ No valid data for prediction")
        return
    
    last_data = X.iloc[-1:].values
    
    # Load and run models
    models_to_check = [
        ('Random Forest', f'models/{ticker}_random_forest_model.pkl'),
        ('Gradient Boost', f'models/{ticker}_gradient_boost_model.pkl'),
        ('SVR', f'models/{ticker}_svr_model.pkl')
    ]
    
    for model_name, model_path in models_to_check:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                pred = model.predict(last_data)[0]
                predictions.append((model_name, pred))
                print(f"✅ {model_name}: ₹{pred:.2f}")
            except Exception as e:
                print(f"❌ {model_name} error: {e}")
    
    if not predictions:
        print("❌ No predictions generated")
        return
    
    # Display results
    print(f"\n🎯 PREDICTION RESULTS")
    print("=" * 60)
    
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
    print(f"\n✅ RELIANCE prediction completed successfully!")

if __name__ == "__main__":
    main()
