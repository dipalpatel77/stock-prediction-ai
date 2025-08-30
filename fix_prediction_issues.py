#!/usr/bin/env python3
"""
Fix Prediction Issues Script
Addresses common issues with the AI Stock Predictor
"""

import os
import sys
from pathlib import Path

def fix_environment_variables():
    """Fix Angel One environment variables."""
    print("🔧 Fixing environment variables...")
    
    # Set Angel One environment variables
    os.environ['ANGEL_ONE_API_KEY'] = '3PMAARNa'
    os.environ['ANGEL_ONE_CLIENT_CODE'] = 'D54448'
    os.environ['ANGEL_ONE_CLIENT_PIN'] = '2251'
    os.environ['ANGEL_ONE_TOTP_SECRET'] = 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
    
    print("✅ Environment variables set")

def fix_symbol_format(ticker):
    """Fix ticker symbol format for Indian stocks."""
    print(f"🔧 Fixing symbol format for {ticker}...")
    
    # Common Indian stock mappings
    indian_stocks = {
        'TCS': 'TCS.NS',
        'RELIANCE': 'RELIANCE.NS',
        'HDFC': 'HDFC.NS',
        'INFY': 'INFY.NS',
        'ICICIBANK': 'ICICIBANK.NS',
        'HINDUNILVR': 'HINDUNILVR.NS',
        'ITC': 'ITC.NS',
        'SBIN': 'SBIN.NS',
        'BHARTIARTL': 'BHARTIARTL.NS',
        'KOTAKBANK': 'KOTAKBANK.NS',
        'AXISBANK': 'AXISBANK.NS',
        'ASIANPAINT': 'ASIANPAINT.NS',
        'MARUTI': 'MARUTI.NS',
        'SUNPHARMA': 'SUNPHARMA.NS',
        'TATAMOTORS': 'TATAMOTORS.NS',
        'WIPRO': 'WIPRO.NS',
        'ULTRACEMCO': 'ULTRACEMCO.NS',
        'TITAN': 'TITAN.NS',
        'NESTLEIND': 'NESTLEIND.NS',
        'POWERGRID': 'POWERGRID.NS',
        'TECHM': 'TECHM.NS',
        'BAJFINANCE': 'BAJFINANCE.NS',
        'NTPC': 'NTPC.NS',
        'HCLTECH': 'HCLTECH.NS',
        'JSWSTEEL': 'JSWSTEEL.NS',
        'ONGC': 'ONGC.NS',
        'TATASTEEL': 'TATASTEEL.NS',
        'ADANIENT': 'ADANIENT.NS',
        'ADANIPORTS': 'ADANIPORTS.NS',
        'BAJAJFINSV': 'BAJAJFINSV.NS',
        'BRITANNIA': 'BRITANNIA.NS',
        'CIPLA': 'CIPLA.NS',
        'COALINDIA': 'COALINDIA.NS',
        'DIVISLAB': 'DIVISLAB.NS',
        'DRREDDY': 'DRREDDY.NS',
        'EICHERMOT': 'EICHERMOT.NS',
        'GRASIM': 'GRASIM.NS',
        'HDFCLIFE': 'HDFCLIFE.NS',
        'HEROMOTOCO': 'HEROMOTOCO.NS',
        'HINDALCO': 'HINDALCO.NS',
        'LT': 'LT.NS',
        'M&M': 'M&M.NS',
        'SHREECEM': 'SHREECEM.NS',
        'TATACONSUM': 'TATACONSUM.NS',
        'UPL': 'UPL.NS',
        'VEDL': 'VEDL.NS',
        'WIPRO': 'WIPRO.NS',
        'ZEEL': 'ZEEL.NS',
        'SWIGGY': 'SWIGGY.NS',  # New age companies
        'ZOMATO': 'ZOMATO.NS',
        'PAYTM': 'PAYTM.NS',
        'NYKAA': 'NYKAA.NS',
        'DELHIVERY': 'DELHIVERY.NS'
    }
    
    # Check if it's an Indian stock without suffix
    if ticker in indian_stocks:
        corrected_ticker = indian_stocks[ticker]
        print(f"   ✅ Corrected {ticker} → {corrected_ticker}")
        return corrected_ticker
    
    # Check if it already has a suffix
    if '.' in ticker:
        print(f"   ✅ {ticker} already has correct format")
        return ticker
    
    # For non-Indian stocks, return as is
    print(f"   ✅ {ticker} is not an Indian stock")
    return ticker

def test_angel_one_connection():
    """Test Angel One connection."""
    print("🔧 Testing Angel One connection...")
    
    try:
        from core.angel_one_data_downloader import AngelOneDataDownloader
        
        # Create downloader
        downloader = AngelOneDataDownloader()
        
        # Test authentication
        auth_result = downloader.authenticate()
        
        if auth_result:
            print("✅ Angel One authentication successful")
            return True
        else:
            print("❌ Angel One authentication failed")
            return False
            
    except Exception as e:
        print(f"❌ Angel One connection error: {e}")
        return False

def test_data_service():
    """Test DataService functionality."""
    print("🔧 Testing DataService...")
    
    try:
        from core.data_service import DataService
        
        # Create DataService
        data_service = DataService()
        
        # Test with a known working stock
        test_ticker = "RELIANCE.NS"
        print(f"   Testing with {test_ticker}...")
        
        # Try to load data
        data = data_service.load_stock_data(test_ticker, period='1mo')
        
        if data is not None and not data.empty:
            print(f"   ✅ Successfully loaded {len(data)} records for {test_ticker}")
            print(f"   📊 Data shape: {data.shape}")
            print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
            return True
        else:
            print(f"   ❌ Failed to load data for {test_ticker}")
            return False
            
    except Exception as e:
        print(f"   ❌ DataService error: {e}")
        return False

def test_indian_stock_mapper():
    """Test Indian Stock Mapper."""
    print("🔧 Testing Indian Stock Mapper...")
    
    try:
        from data_downloaders.indian_stock_mapper import load_angel_master, get_symbol_info
        
        # Load Angel master data
        angel_master = load_angel_master()
        
        if angel_master is not None and not angel_master.empty:
            print(f"   ✅ Successfully loaded Angel master data ({len(angel_master)} records)")
            
            # Test symbol lookup
            test_symbols = ['TCS', 'RELIANCE', 'HDFC', 'INFY']
            for symbol in test_symbols:
                symbol_info = get_symbol_info(symbol, angel_master)
                if symbol_info:
                    print(f"   ✅ Found {symbol}: {symbol_info['exchange']} - Token: {symbol_info['token']}")
                else:
                    print(f"   ⚠️ {symbol} not found in Angel master")
            
            return True
        else:
            print("   ❌ Failed to load Angel master data")
            return False
            
    except Exception as e:
        print(f"   ❌ Indian Stock Mapper error: {e}")
        return False

def create_test_script():
    """Create a test script for troubleshooting."""
    print("🔧 Creating test script...")
    
    test_script = '''#!/usr/bin/env python3
"""
Quick Test Script for AI Stock Predictor
"""

import os
import sys

# Set environment variables
os.environ['ANGEL_ONE_API_KEY'] = '3PMAARNa'
os.environ['ANGEL_ONE_CLIENT_CODE'] = 'D54448'
os.environ['ANGEL_ONE_CLIENT_PIN'] = '2251'
os.environ['ANGEL_ONE_TOTP_SECRET'] = 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'

def test_basic_functionality():
    """Test basic functionality."""
    print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        from core.data_service import DataService
        from core.angel_one_data_downloader import AngelOneDataDownloader
        from data_downloaders.indian_stock_mapper import load_angel_master
        
        print("✅ All imports successful")
        
        # Test DataService
        data_service = DataService()
        print("✅ DataService created")
        
        # Test Angel One
        angel_downloader = AngelOneDataDownloader()
        print("✅ AngelOneDataDownloader created")
        
        # Test Indian Stock Mapper
        angel_master = load_angel_master()
        print(f"✅ Indian Stock Mapper loaded ({len(angel_master)} records)")
        
        # Test data download
        test_ticker = "RELIANCE.NS"
        data = data_service.load_stock_data(test_ticker, period='1mo')
        
        if data is not None and not data.empty:
            print(f"✅ Data download successful for {test_ticker}")
            print(f"   Records: {len(data)}")
            print(f"   Features: {data.shape[1]}")
            return True
        else:
            print(f"❌ Data download failed for {test_ticker}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\\n🎉 All tests passed!")
    else:
        print("\\n❌ Some tests failed!")
'''
    
    with open('quick_test.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created quick_test.py")

def main():
    """Main function."""
    print("🚀 AI Stock Predictor - Issue Fix Script")
    print("=" * 50)
    
    # Fix environment variables
    fix_environment_variables()
    
    # Test Angel One connection
    angel_success = test_angel_one_connection()
    
    # Test DataService
    data_success = test_data_service()
    
    # Test Indian Stock Mapper
    mapper_success = test_indian_stock_mapper()
    
    # Create test script
    create_test_script()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 FIX SUMMARY")
    print("=" * 50)
    
    print(f"Angel One Connection: {'✅' if angel_success else '❌'}")
    print(f"DataService: {'✅' if data_success else '❌'}")
    print(f"Indian Stock Mapper: {'✅' if mapper_success else '❌'}")
    
    if angel_success and data_success and mapper_success:
        print("\n🎉 All components working!")
        print("🚀 You can now run stock analysis")
    else:
        print("\n⚠️ Some components need attention")
        print("🔧 Run 'python quick_test.py' for detailed testing")
    
    print("\n📋 RECOMMENDATIONS:")
    print("1. Use correct symbol format (e.g., 'TCS.NS' not 'TCS')")
    print("2. Ensure environment variables are set")
    print("3. Run 'python quick_test.py' for troubleshooting")
    print("4. Check internet connection for data downloads")

if __name__ == "__main__":
    main()
