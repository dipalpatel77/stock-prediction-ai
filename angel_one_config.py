#!/usr/bin/env python3
"""
Angel One API Configuration Helper
Helps users set up their Angel One API credentials
"""

import os
from dotenv import load_dotenv

def setup_angel_one_credentials():
    """
    Interactive setup for Angel One API credentials.
    Creates a .env file with the required credentials.
    """
    print("🔧 Angel One API Configuration Setup")
    print("=" * 50)
    print("To use Angel One API, you need to:")
    print("1. Create an account on Angel One")
    print("2. Generate API credentials")
    print("3. Get your API key and auth token")
    print()
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("📁 .env file already exists. Current settings:")
        load_dotenv()
        api_key = os.getenv('ANGEL_ONE_API_KEY', 'Not set')
        auth_token = os.getenv('ANGEL_ONE_AUTH_TOKEN', 'Not set')
        print(f"   API Key: {'*' * len(api_key) if api_key != 'Not set' else 'Not set'}")
        print(f"   Auth Token: {'*' * len(auth_token) if auth_token != 'Not set' else 'Not set'}")
        
        update = input("\nDo you want to update these credentials? (y/n): ").lower()
        if update != 'y':
            print("✅ Using existing credentials")
            return
    
    print("Please enter your Angel One API credentials:")
    print()
    
    # Get credentials from user
    api_key = input("Enter your API Key: ").strip()
    auth_token = input("Enter your Auth Token: ").strip()
    client_ip = input("Enter your Client IP (default: 127.0.0.1): ").strip() or "127.0.0.1"
    mac_address = input("Enter your MAC Address (default: 00:00:00:00:00:00): ").strip() or "00:00:00:00:00:00"
    
    # Create .env file
    env_content = f"""# Angel One API Credentials
ANGEL_ONE_API_KEY={api_key}
ANGEL_ONE_AUTH_TOKEN={auth_token}
CLIENT_IP={client_ip}
MAC_ADDRESS={mac_address}

# Other Configuration
# Add any other environment variables here
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("\n✅ .env file created successfully!")
        print("📁 Credentials saved to .env file")
        print("\n⚠️  Important: Keep your .env file secure and never commit it to version control!")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False
    
    return True

def test_angel_one_connection():
    """
    Test the Angel One API connection with current credentials.
    """
    try:
        from angel_one_data_downloader import AngelOneDataDownloader
        
        # Load environment variables
        load_dotenv()
        
        # Get credentials
        api_key = os.getenv('ANGEL_ONE_API_KEY')
        auth_token = os.getenv('ANGEL_ONE_AUTH_TOKEN')
        client_ip = os.getenv('CLIENT_IP', '127.0.0.1')
        mac_address = os.getenv('MAC_ADDRESS', '00:00:00:00:00:00')
        
        if not api_key or not auth_token:
            print("❌ API credentials not found. Please run setup_angel_one_credentials() first.")
            return False
        
        print("🧪 Testing Angel One API connection...")
        
        # Initialize downloader
        downloader = AngelOneDataDownloader(
            api_key=api_key,
            auth_token=auth_token,
            client_ip=client_ip,
            mac_address=mac_address
        )
        
        # Test with a small data request
        print("📊 Testing with RELIANCE stock (last 7 days)...")
        df = downloader.download_historical_data(
            symbol="RELIANCE",
            from_date="2024-08-18",
            to_date="2024-08-25",
            interval="ONE_DAY"
        )
        
        if df is not None and len(df) > 0:
            print("✅ Connection successful!")
            print(f"📈 Downloaded {len(df)} records")
            print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
            return True
        else:
            print("❌ Connection failed. Please check your credentials.")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def get_supported_stocks():
    """
    Display list of supported stock symbols.
    """
    try:
        from angel_one_data_downloader import AngelOneDataDownloader
        downloader = AngelOneDataDownloader()
        
        print("📋 Supported Stock Symbols:")
        print("=" * 50)
        
        # Get all supported stocks
        stocks = list(downloader.stock_tokens.keys())
        
        # Display in columns
        for i in range(0, len(stocks), 4):
            row = stocks[i:i+4]
            print("   ".join(f"{stock:15}" for stock in row))
        
        print(f"\nTotal: {len(stocks)} stocks supported")
        
    except Exception as e:
        print(f"❌ Error getting supported stocks: {e}")

def main():
    """Main function for configuration setup."""
    print("🚀 Angel One API Configuration")
    print("=" * 40)
    print("1. Setup API Credentials")
    print("2. Test Connection")
    print("3. Show Supported Stocks")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            setup_angel_one_credentials()
        elif choice == "2":
            test_angel_one_connection()
        elif choice == "3":
            get_supported_stocks()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")
        
        print()

if __name__ == "__main__":
    main()
