#!/usr/bin/env python3
"""
Test PNB lookup in Angel One database
"""

import sys
import os

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

def test_pnb_lookup():
    """Test PNB stock lookup in Angel One database"""
    print("🔍 Testing PNB lookup in Angel One database")
    print("=" * 50)
    
    try:
        # Test Angel One data downloader
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        downloader = AngelOneDataDownloader()
        print("✅ Angel One Data Downloader initialized")
        
        # Test symbol lookup for PNB
        print("\n🔍 Looking up PNB in Angel One database...")
        token = downloader.get_symbol_token("PNB", "NSE")
        print(f"✅ PNB Token: {token}")
        
        # Test with BSE as well
        print("\n🔍 Looking up PNB in BSE...")
        token_bse = downloader.get_symbol_token("PNB", "BSE")
        print(f"✅ PNB BSE Token: {token_bse}")
        
        # Test other Indian stocks
        test_stocks = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "SBIN", "WIPRO", "BHARTIARTL"]
        
        print("\n🔍 Testing other Indian stocks...")
        for stock in test_stocks:
            try:
                token = downloader.get_symbol_token(stock, "NSE")
                print(f"✅ {stock}: {token}")
            except Exception as e:
                print(f"❌ {stock}: {e}")
        
        print("\n🎉 Angel One stock lookup is working!")
        return True
        
    except Exception as e:
        print(f"❌ Angel One lookup error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pnb_lookup()
    sys.exit(0 if success else 1)
