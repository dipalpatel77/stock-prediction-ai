import pandas as pd
import os

# Test enhanced indicators function
ticker = "AAPL"
data_file = f"data/{ticker}_partA_preprocessed.csv"

print("=== Debug Enhanced Indicators ===")
print(f"File exists: {os.path.exists(data_file)}")

if os.path.exists(data_file):
    # Read the preprocessed data
    df = pd.read_csv(data_file)
    print(f"Raw data shape: {df.shape}")
    print(f"Raw data columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    # Try to fix the data structure
    print("\n=== Fixing data structure ===")
    
    # Handle the specific format where the second column contains dates
    if len(df.columns) > 1 and df.columns[1] == 'Price':
        print("Detected Price column format")
        df['Date'] = df['Price']
        df = df.drop('Price', axis=1)
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    elif len(df.columns) > 1 and df.columns[1] == 'Date':
        print("Detected Date column format")
        df['Date'] = df['Date']
        df = df.drop('Date', axis=1)
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    elif len(df.columns) > 1 and df.columns[1] == 'Price' and df.columns[2] == 'Adj Close':
        print("Detected AAPL format")
        df['Date'] = df['Price']
        df = df.drop('Price', axis=1)
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
    else:
        print("Using standard format")
        df.set_index(df.columns[0], inplace=True)
        df.index = pd.to_datetime(df.index)
    
    print(f"Fixed data shape: {df.shape}")
    print(f"Fixed data columns: {df.columns.tolist()}")
    print(f"Index type: {type(df.index)}")
    print(f"First few rows of fixed data:")
    print(df.head())
    
    # Test the enhanced indicators function
    print("\n=== Testing Enhanced Indicators ===")
    try:
        from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
        
        technical_analyzer = OptimizedTechnicalIndicators()
        df_enhanced = technical_analyzer.add_all_indicators(df)
        
        print(f"Enhanced data shape: {df_enhanced.shape}")
        print(f"Enhanced data columns: {df_enhanced.columns.tolist()}")
        print("✅ Enhanced indicators working!")
        
    except Exception as e:
        print(f"❌ Error in enhanced indicators: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Preprocessed data file not found!")
