#!/usr/bin/env python3
"""
Very simple test for the stock prediction system
"""

import pandas as pd
import os

def main():
    print("🧪 Simple System Test")
    print("=" * 30)
    
    # Test 1: Check if data exists
    data_file = "data/RELIANCE_partA_partC_enhanced.csv"
    if os.path.exists(data_file):
        print(f"✅ Data file found: {data_file}")
        try:
            df = pd.read_csv(data_file)
            print(f"📊 Data shape: {df.shape}")
            print(f"📅 Columns: {list(df.columns[:5])}...")
        except Exception as e:
            print(f"❌ Error reading data: {e}")
    else:
        print(f"❌ Data file not found: {data_file}")
    
    # Test 2: Try to import the main module
    try:
        import run_stock_prediction
        print("✅ Main module imported successfully")
    except Exception as e:
        print(f"❌ Error importing main module: {e}")
    
    # Test 3: Try to create prediction engine
    try:
        from run_stock_prediction import UnifiedPredictionEngine
        engine = UnifiedPredictionEngine('RELIANCE', mode='simple')
        print("✅ Prediction engine created successfully")
    except Exception as e:
        print(f"❌ Error creating prediction engine: {e}")

if __name__ == "__main__":
    main()
