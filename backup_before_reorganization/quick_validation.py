#!/usr/bin/env python3
"""
Quick Prediction Validation
Simple script to quickly validate prediction accuracy
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys

def quick_validate_predictions(ticker):
    """Quick validation of predictions."""
    print(f"🔍 QUICK VALIDATION FOR {ticker}")
    print("=" * 50)
    
    # 1. Check if prediction file exists
    prediction_file = f"data/{ticker}_timeframe_predictions.csv"
    if not os.path.exists(prediction_file):
        print(f"❌ No prediction file found: {prediction_file}")
        return
    
    # Load predictions
    df_predictions = pd.read_csv(prediction_file)
    print(f"✅ Found prediction file with {len(df_predictions)} predictions")
    
    # 2. Get current actual price
    try:
        print("\n📊 FETCHING CURRENT PRICE...")
        stock = yf.Ticker(ticker)
        current_data = stock.history(period="1d")
        
        if current_data.empty:
            print("❌ Could not fetch current price")
            return
        
        current_price = current_data['Close'].iloc[-1]
        print(f"✅ Current Price: ${current_price:.2f}")
        
    except Exception as e:
        print(f"❌ Error fetching current price: {e}")
        return
    
    # 3. Compare with predictions
    print("\n📈 PREDICTION vs ACTUAL COMPARISON:")
    print("-" * 50)
    
    # Short-term predictions
    short_term = df_predictions[df_predictions['Timeframe'] == 'Short-Term']
    if not short_term.empty:
        print("🔸 SHORT-TERM PREDICTIONS:")
        for _, row in short_term.iterrows():
            predicted = row['Predicted_Price']
            error = abs(predicted - current_price)
            error_percent = (error / current_price) * 100
            accuracy = max(0, 1 - error_percent/100)
            
            print(f"  {row['Period']}: ${predicted:.2f} "
                  f"(Error: {error_percent:.1f}%, Accuracy: {accuracy*100:.1f}%)")
    
    # Medium-term predictions
    medium_term = df_predictions[df_predictions['Timeframe'] == 'Medium-Term']
    if not medium_term.empty:
        print("\n🔸 MEDIUM-TERM PREDICTIONS:")
        for _, row in medium_term.iterrows():
            predicted = row['Predicted_Price']
            error = abs(predicted - current_price)
            error_percent = (error / current_price) * 100
            accuracy = max(0, 1 - error_percent/100)
            
            print(f"  {row['Period']}: ${predicted:.2f} "
                  f"(Error: {error_percent:.1f}%, Accuracy: {accuracy*100:.1f}%)")
    
    # Long-term predictions
    long_term = df_predictions[df_predictions['Timeframe'] == 'Long-Term']
    if not long_term.empty:
        print("\n🔸 LONG-TERM PREDICTIONS:")
        for _, row in long_term.iterrows():
            predicted = row['Predicted_Price']
            error = abs(predicted - current_price)
            error_percent = (error / current_price) * 100
            accuracy = max(0, 1 - error_percent/100)
            
            print(f"  {row['Period']}: ${predicted:.2f} "
                  f"(Error: {error_percent:.1f}%, Accuracy: {accuracy*100:.1f}%)")
    
    # 4. Historical accuracy check
    print("\n📊 HISTORICAL ACCURACY CHECK:")
    print("-" * 50)
    
    try:
        # Get historical data for the last 30 days
        historical_data = stock.history(period="30d")
        if not historical_data.empty:
            # Calculate daily returns
            daily_returns = historical_data['Close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = daily_returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate trend
            trend = daily_returns.mean() * 252  # Annualized
            
            print(f"  Volatility (30-day): {volatility*100:.1f}%")
            print(f"  Trend (30-day): {trend*100:.1f}%")
            
            # Price range
            price_range = historical_data['Close'].max() - historical_data['Close'].min()
            price_range_percent = (price_range / historical_data['Close'].mean()) * 100
            
            print(f"  Price Range (30-day): {price_range_percent:.1f}%")
            
    except Exception as e:
        print(f"  ❌ Historical analysis failed: {e}")
    
    # 5. Validation recommendations
    print("\n💡 VALIDATION RECOMMENDATIONS:")
    print("-" * 50)
    print("  • Check predictions daily to track accuracy over time")
    print("  • Compare with actual price movements")
    print("  • Monitor prediction errors and adjust models if needed")
    print("  • Use multiple timeframes for better validation")
    print("  • Consider market volatility when interpreting results")
    
    print("\n✅ Quick validation completed!")

def validate_multiple_stocks(tickers):
    """Validate predictions for multiple stocks."""
    for ticker in tickers:
        quick_validate_predictions(ticker)
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        quick_validate_predictions(ticker)
    else:
        # Validate recent predictions
        print("🔍 VALIDATING RECENT PREDICTIONS")
        print("=" * 50)
        
        # Find all prediction files
        data_dir = "data"
        prediction_files = [f for f in os.listdir(data_dir) if f.endswith('_timeframe_predictions.csv')]
        
        if not prediction_files:
            print("❌ No prediction files found in data/ directory")
            print("💡 Run the unified analysis pipeline first to generate predictions")
        else:
            print(f"✅ Found {len(prediction_files)} prediction files")
            
            # Extract tickers
            tickers = [f.replace('_timeframe_predictions.csv', '') for f in prediction_files]
            
            print(f"📊 Validating predictions for: {', '.join(tickers)}")
            print()
            
            # Validate each ticker
            for ticker in tickers[:3]:  # Limit to first 3 for quick validation
                quick_validate_predictions(ticker)
                print("\n" + "="*60 + "\n")
