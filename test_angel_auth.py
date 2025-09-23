#!/usr/bin/env python3
"""
Test Angel One Authentication
"""

import sys
import os
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_angel_authentication():
    """Test Angel One authentication step by step"""
    print("🔐 Testing Angel One Authentication")
    print("=" * 40)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        from src.utils.angel_one_data_downloader import AngelOneDataDownloader
        
        # Initialize downloader
        print("📊 Initializing Angel One Data Downloader...")
        downloader = AngelOneDataDownloader()
        
        # Check credentials
        print("\n🔑 Checking credentials...")
        print(f"   API Key: {downloader.api_key}")
        print(f"   Client Code: {downloader.client_code}")
        print(f"   Client PIN: {downloader.client_pin}")
        print(f"   TOTP Secret: {downloader.totp_secret[:10]}...")
        
        # Test TOTP generation
        print("\n🔐 Testing TOTP generation...")
        try:
            import pyotp
            totp_code = pyotp.TOTP(downloader.totp_secret).now()
            print(f"   ✅ TOTP generated: {totp_code}")
        except Exception as e:
            print(f"   ❌ TOTP generation failed: {e}")
            return False
        
        # Test authentication
        print("\n🔐 Testing authentication...")
        auth_success = downloader.authenticate()
        
        if auth_success:
            print("✅ Authentication successful!")
            print(f"   JWT Token: {downloader.jwt_token[:20]}...")
            print(f"   Refresh Token: {downloader.refresh_token[:20]}...")
            return True
        else:
            print("❌ Authentication failed!")
            return False
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_manual_authentication():
    """Test manual authentication with detailed logging"""
    print("\n🔍 Testing Manual Authentication with Detailed Logging")
    print("=" * 55)
    
    try:
        import requests
        import pyotp
        
        # Credentials
        api_key = "1TKgQThc "
        client_code = "D54448"
        client_pin = "2251"
        totp_secret = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"
        
        # Generate TOTP
        totp = pyotp.TOTP(totp_secret)
        totp_code = totp.now()
        print(f"🔐 Generated TOTP: {totp_code}")
        
        # Prepare login payload
        login_payload = {
            "clientcode": client_code,
            "password": client_pin,
            "totp": totp_code
        }
        
        print(f"📤 Login payload: {login_payload}")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": "127.0.0.1",
            "X-ClientPublicIP": "127.0.0.1",
            "X-MACAddress": "XX:XX:XX:XX:XX:XX",
            "X-PrivateKey": api_key
        }
        
        print(f"📤 Headers: {headers}")
        
        # Make request
        login_url = "https://apiconnect.angelone.in/rest/auth/angelbroking/user/v1/loginByPassword"
        
        print(f"🌐 Making request to: {login_url}")
        
        response = requests.post(
            login_url,
            json=login_payload,
            headers=headers,
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 Response data: {data}")
            
            if data.get("status") == True:
                jwt_token = data["data"]["jwtToken"]
                refresh_token = data["data"]["refreshToken"]
                print(f"✅ Authentication successful!")
                print(f"   JWT Token: {jwt_token[:20]}...")
                print(f"   Refresh Token: {refresh_token[:20]}...")
                return True
            else:
                error_code = data.get("errorCode", "Unknown")
                error_msg = data.get("message", "Unknown error")
                print(f"❌ Authentication failed: {error_msg} (Code: {error_code})")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Manual authentication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    try:
        print("🧪 Testing Angel One Authentication")
        print("=" * 50)
        
        # Test standard authentication
        auth_success = test_angel_authentication()
        
        # Test manual authentication with detailed logging
        manual_success = test_manual_authentication()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        print(f"Standard Authentication: {'✅ PASS' if auth_success else '❌ FAIL'}")
        print(f"Manual Authentication: {'✅ PASS' if manual_success else '❌ FAIL'}")
        
        overall_success = auth_success or manual_success
        print(f"\n🎯 Overall Result: {'✅ AUTHENTICATION WORKS' if overall_success else '❌ AUTHENTICATION FAILED'}")
        
        if not overall_success:
            print("\n🔍 Troubleshooting Steps:")
            print("1. Check if TOTP secret is correct")
            print("2. Verify API credentials")
            print("3. Check network connectivity")
            print("4. Verify Angel One API status")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
