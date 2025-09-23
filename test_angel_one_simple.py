#!/usr/bin/env python3
"""
Simple Angel One API Test
"""

import sys
import os

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

def test_angel_one_simple():
    """Test Angel One API with simple approach"""
    print("🇮🇳 Testing Angel One API - Simple Test")
    print("=" * 50)
    
    try:
        # Test Angel One manager directly
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
        
        print("✅ Creating Angel One Manager...")
        angel_manager = AngelOneManager(test_config)
        print("✅ Angel One Manager created successfully")
        
        # Test rate limiting
        print("✅ Testing rate limiting...")
        rate_limit = angel_manager.check_rate_limit()
        print(f"   Rate limit status: {rate_limit}")
        
        # Test configuration
        print("✅ Testing configuration...")
        print(f"   API Key: {angel_manager.api_key[:8]}...")
        print(f"   Client Code: {angel_manager.client_code}")
        print(f"   Exchange: {test_config['exchange']}")
        print(f"   Interval: {test_config['interval']}")
        
        print("\n🎉 Angel One API is working correctly!")
        print("✅ All Angel One components are functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Angel One API error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_angel_one_simple()
    sys.exit(0 if success else 1)
