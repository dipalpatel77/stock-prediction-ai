#!/usr/bin/env python3
"""
Display RELIANCE Prediction Results
Show the existing prediction results in a formatted way
"""

import pandas as pd
from datetime import datetime

def display_reliance_results():
    """Display RELIANCE prediction results."""
    print("🎯 RELIANCE Stock Price Prediction Results")
    print("=" * 60)
    
    # Load existing predictions
    predictions_file = "data/RELIANCE_latest_predictions.csv"
    
    try:
        df = pd.read_csv(predictions_file)
        if df.empty:
            print("❌ No prediction data found")
            return
        
        # Get the latest prediction
        latest = df.iloc[-1]
        
        # Extract data
        current_price = float(latest['Current_Price'])
        rf_pred = float(latest['Random Forest_Prediction'])
        gb_pred = float(latest['Gradient Boost_Prediction'])
        svr_pred = float(latest['SVR_Prediction'])
        
        # Enhanced predictions
        enhanced_preds = []
        for i in range(1, 6):
            col_name = f'Enhanced_Day_{i}'
            if col_name in latest:
                enhanced_preds.append(float(latest[col_name]))
        
        print(f"📊 Stock: RELIANCE")
        print(f"💰 CURRENT PRICE: ₹{current_price:.2f}")
        print(f"📅 Analysis Date: {latest['Date']}")
        print("-" * 60)
        print()
        
        # Display ensemble predictions
        print("🤖 Ensemble Model Predictions:")
        
        # Random Forest
        rf_change = rf_pred - current_price
        rf_change_pct = (rf_change / current_price) * 100
        rf_direction = "📈" if rf_change > 0 else "📉" if rf_change < 0 else "➡️"
        print(f"   • Random Forest: ₹{rf_pred:.2f} ({rf_direction} {rf_change_pct:+.2f}%)")
        
        # Gradient Boost
        gb_change = gb_pred - current_price
        gb_change_pct = (gb_change / current_price) * 100
        gb_direction = "📈" if gb_change > 0 else "📉" if gb_change < 0 else "➡️"
        print(f"   • Gradient Boost: ₹{gb_pred:.2f} ({gb_direction} {gb_change_pct:+.2f}%)")
        
        # SVR
        svr_change = svr_pred - current_price
        svr_change_pct = (svr_change / current_price) * 100
        svr_direction = "📈" if svr_change > 0 else "📉" if svr_change < 0 else "➡️"
        print(f"   • SVR: ₹{svr_pred:.2f} ({svr_direction} {svr_change_pct:+.2f}%)")
        
        # Display enhanced predictions
        if enhanced_preds:
            print(f"\n🚀 Enhanced Model Predictions (5 days):")
            for i, pred in enumerate(enhanced_preds, 1):
                change = pred - current_price
                change_pct = (change / current_price) * 100
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   • Day {i}: ₹{pred:.2f} ({direction} {change_pct:+.2f}%)")
        
        # Calculate average prediction
        all_predictions = [rf_pred, gb_pred, svr_pred] + enhanced_preds
        avg_prediction = sum(all_predictions) / len(all_predictions)
        avg_change = avg_prediction - current_price
        avg_change_pct = (avg_change / current_price) * 100
        avg_direction = "📈" if avg_change > 0 else "📉" if avg_change < 0 else "➡️"
        
        print(f"\n📊 Average Prediction: ₹{avg_prediction:.2f} ({avg_direction} {avg_change_pct:+.2f}%)")
        
        # Price summary
        print(f"\n📋 PRICE SUMMARY:")
        print(f"   Current Price: ₹{current_price:.2f}")
        print(f"   Predicted Price: ₹{avg_prediction:.2f}")
        print(f"   Expected Change: {avg_change_pct:+.2f}%")
        
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
        
        print(f"\n✅ RELIANCE prediction analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error reading prediction data: {e}")

if __name__ == "__main__":
    display_reliance_results()
