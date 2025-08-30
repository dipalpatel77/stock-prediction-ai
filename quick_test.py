#!/usr/bin/env python3
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
    print("Testing basic functionality...")
    
    try:
        # Test imports
        from core.data_service import DataService
        from core.angel_one_data_downloader import AngelOneDataDownloader
        from data_downloaders.indian_stock_mapper import load_angel_master
        
        print("OK: All imports successful")
        
        # Test DataService
        data_service = DataService()
        print("OK: DataService created")
        
        # Test Angel One
        angel_downloader = AngelOneDataDownloader()
        print("OK: AngelOneDataDownloader created")
        
        # Test Indian Stock Mapper
        angel_master = load_angel_master()
        print(f"OK: Indian Stock Mapper loaded ({len(angel_master)} records)")
        
        # Test data download
        test_ticker = "RELIANCE.NS"
        data = data_service.load_stock_data(test_ticker, period='1mo')
        
        if data is not None and not data.empty:
            print(f"OK: Data download successful for {test_ticker}")
            print(f"   Records: {len(data)}")
            print(f"   Features: {data.shape[1]}")
            return True
        else:
            print(f"ERROR: Data download failed for {test_ticker}")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        return False

def test_symbol_format():
    """Test symbol format handling."""
    print("\nTesting symbol format handling...")
    
    try:
        from core.data_service import DataService
        data_service = DataService()
        
        test_symbols = ['TCS', 'TCS.NS', 'RELIANCE', 'RELIANCE.NS', 'AAPL']
        
        for symbol in test_symbols:
            is_indian = data_service._is_indian_stock(symbol)
            print(f"   {symbol}: {'Indian' if is_indian else 'Non-Indian'}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Symbol format test failed: {e}")
        return False

if __name__ == "__main__":
    print("AI Stock Predictor - Quick Test")
    print("=" * 40)
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    # Test symbol format
    symbol_success = test_symbol_format()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    if basic_success and symbol_success:
        print("SUCCESS: All tests passed!")
        print("The system is working correctly.")
    else:
        print("ERROR: Some tests failed!")
        print("Check the error messages above.")
    
    print("\nRECOMMENDATIONS:")
    print("1. Use correct symbol format (e.g., 'TCS.NS' not 'TCS')")
    print("2. Ensure environment variables are set")
    print("3. Check internet connection for data downloads")
