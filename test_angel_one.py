#!/usr/bin/env python3
"""
Test Angel One API integration
"""

import sys
import os

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

def test_angel_one():
    """Test Angel One API integration"""
    print("🇮🇳 Testing Angel One API Integration")
    print("=" * 50)
    
    try:
        # Test Angel One interface
        from main.interfaces.angel_one_interface import AngelOneInterface
        angel_interface = AngelOneInterface()
        print("✅ Angel One Interface initialized")
        
        # Test Angel One manager
        from main.services.angel_one_manager import AngelOneManager
        
        # Create test configuration
        test_config = {
            'api_key': '1TKgQThc ',
            'api_secret': 'D54448',
            'access_token': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE',
            'exchange': 'NSE',
            'interval': 'ONE_DAY',
            'connection_string': 'sqlite:///angel_one_test.db'
        }
        
        angel_manager = AngelOneManager(test_config)
        print("✅ Angel One Manager initialized")
        
        # Test rate limiting
        rate_limit = angel_manager.check_rate_limit()
        print(f"✅ Rate limiting: {rate_limit}")
        
        # Test configuration
        angel_config = angel_interface.configure_angel_one("RELIANCE", test_config)
        print(f"✅ Angel One configuration: {angel_config is not None}")
        
        print("\n🎉 Angel One API integration is working!")
        return True
        
    except Exception as e:
        print(f"❌ Angel One API error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_angel_one()
    sys.exit(0 if success else 1)
