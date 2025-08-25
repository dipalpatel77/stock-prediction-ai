#!/usr/bin/env python3
"""
Angel One Historical Data Downloader
Downloads historical stock data for Indian stocks using Angel One API
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class AngelOneDataDownloader:
    """
    Downloads historical data from Angel One API for Indian stocks
    """
    
    def __init__(self, api_key=None, auth_token=None, client_ip="127.0.0.1", mac_address="00:00:00:00:00:00"):
        self.base_url = "https://apiconnect.angelone.in"
        self.api_key = api_key
        self.auth_token = auth_token
        self.client_ip = client_ip
        self.mac_address = mac_address
        
        # Common headers for all requests
        self.headers = {
            'X-PrivateKey': self.api_key,
            'Accept': 'application/json',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': self.client_ip,
            'X-ClientPublicIP': self.client_ip,
            'X-MACAddress': self.mac_address,
            'X-UserType': 'USER',
            'Authorization': f'Bearer {self.auth_token}',
            'Content-Type': 'application/json'
        }
        
        # Stock token mappings (common Indian stocks)
        self.stock_tokens = {
            'RELIANCE': '2885',
            'TCS': '11536',
            'HDFCBANK': '341',
            'INFY': '1594',
            'ICICIBANK': '4963',
            'HINDUNILVR': '1394',
            'ITC': '1660',
            'SBIN': '3045',
            'BHARTIARTL': '822',
            'KOTAKBANK': '1922',
            'AXISBANK': '590',
            'ASIANPAINT': '694',
            'MARUTI': '10999',
            'HCLTECH': '7229',
            'SUNPHARMA': '3351',
            'TATAMOTORS': '3456',
            'WIPRO': '3787',
            'ULTRACEMCO': '11543',
            'TITAN': '3506',
            'BAJFINANCE': '811',
            'NESTLEIND': '2303',
            'POWERGRID': '14977',
            'BAJAJFINSV': '812',
            'NTPC': '11630',
            'HINDALCO': '1363',
            'JSWSTEEL': '11723',
            'ONGC': '11618',
            'TECHM': '3563',
            'ADANIENT': '25',
            'TATACONSUM': '3432',
            'BRITANNIA': '317',
            'SHREECEM': '3103',
            'DIVISLAB': '378',
            'EICHERMOT': '505',
            'GRASIM': '10099',
            'CIPLA': '2664',
            'HEROMOTOCO': '5251',
            'DRREDDY': '1023',
            'COALINDIA': '2031',
            'ADANIPORTS': '3862',
            'M&M': '189',
            'TATASTEEL': '3435',
            'BPCL': '526',
            'VEDL': '3718',
            'LT': '6401',
            'SBILIFE': '3045',
            'HDFC': '1330',
            'INDUSINDBK': '5258',
            'APOLLOHOSP': '157',
            'BAJAJ-AUTO': '166',
            'TATAPOWER': '3437',
            'UPL': '11287',
            'PERSISTENT': '11584',
            'COLPAL': '2031',
            'MARICO': '10999',
            'DABUR': '257',
            'GODREJCP': '10073',
            'BERGEPAINT': '438',
            'PIDILITIND': '11584',
            'HAVELLS': '1363',
            'SIEMENS': '3123',
            'ABBOTINDIA': '1',
            'BIOCON': '288',
            'MCDOWELL-N': '189',
            'VEDANTACO': '3718',
            'DLF': '377',
            'GODREJPROP': '10073',
            'PEL': '11584',
            'TORNTPHARM': '11584',
            'ALKEM': '11584',
            'DEEPAKNTR': '11584',
            'LUPIN': '11584',
            'CADILAHC': '11584',
            'AUROPHARMA': '11584',
            'DIVISLAB': '378',
            'SUNTV': '3351',
            'ZEEL': '11584',
            'PFC': '11584',
            'RECLTD': '11584',
            'BANKBARODA': '11584',
            'CANBK': '11584',
            'UNIONBANK': '11584',
            'PNB': '11584',
            'IDFCFIRSTB': '11584',
            'FEDERALBNK': '11584',
            'KARNATAKA': '11584',
            'J&KBANK': '11584',
            'SOUTHBANK': '11584',
            'UCOBANK': '11584',
            'CENTRALBK': '11584',
            'INDIANB': '11584',
            'MAHABANK': '11584',
            'BANKINDIA': '11584',
            'IOB': '11584',
            'ALLAHABAD': '11584',
            'ANDHRABANK': '11584',
            'VIJAYABANK': '11584',
            'DENABANK': '11584',
            'ORIENTBANK': '11584',
            'SYNDICATE': '11584',
            'CORPBANK': '11584',
            'UNITEDBNK': '11584',
            'DENA': '11584',
            'ANDHRABANK': '11584',
            'VIJAYABANK': '11584',
            'DENABANK': '11584',
            'ORIENTBANK': '11584',
            'SYNDICATE': '11584',
            'CORPBANK': '11584',
            'UNITEDBNK': '11584',
            'DENA': '11584',
            'ANDHRABANK': '11584',
            'VIJAYABANK': '11584',
            'DENABANK': '11584',
            'ORIENTBANK': '11584',
            'SYNDICATE': '11584',
            'CORPBANK': '11584',
            'UNITEDBNK': '11584',
            'DENA': '11584'
        }
    
    def get_stock_token(self, symbol):
        """Get stock token for a given symbol."""
        symbol_upper = symbol.upper()
        if symbol_upper in self.stock_tokens:
            return self.stock_tokens[symbol_upper]
        else:
            print(f"⚠️ Token not found for {symbol}. Please add it to stock_tokens dictionary.")
            return None
    
    def download_historical_data(self, symbol, from_date, to_date, interval="ONE_DAY", exchange="NSE"):
        """
        Download historical data for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE', 'TCS')
            from_date (str): Start date in 'YYYY-MM-DD' format
            to_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval (ONE_MINUTE, FIVE_MINUTE, ONE_HOUR, ONE_DAY)
            exchange (str): Exchange (NSE, BSE)
        
        Returns:
            pandas.DataFrame: Historical data with columns [Date, Open, High, Low, Close, Volume]
        """
        try:
            # Get stock token
            symbol_token = self.get_stock_token(symbol)
            if not symbol_token:
                return None
            
            # Prepare request payload
            payload = {
                "exchange": exchange,
                "symboltoken": symbol_token,
                "interval": interval,
                "fromdate": f"{from_date} 09:15",
                "todate": f"{to_date} 15:30"
            }
            
            print(f"📊 Downloading {interval} data for {symbol} from {from_date} to {to_date}...")
            
            # Make API request
            url = f"{self.base_url}/rest/secure/angelbroking/historical/v1/getCandleData"
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status') and data.get('data'):
                    # Convert to DataFrame
                    df = pd.DataFrame(data['data'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    
                    # Convert date format
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    
                    # Convert numeric columns
                    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    print(f"✅ Downloaded {len(df)} records for {symbol}")
                    return df
                else:
                    print(f"❌ API Error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading data for {symbol}: {e}")
            return None
    
    def download_multiple_days(self, symbol, days=365, interval="ONE_DAY", exchange="NSE"):
        """
        Download data for multiple days by splitting into chunks.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to download
            interval (str): Data interval
            exchange (str): Exchange
        
        Returns:
            pandas.DataFrame: Combined historical data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get max days per request for the interval
            max_days_map = {
                'ONE_MINUTE': 30,
                'THREE_MINUTE': 60,
                'FIVE_MINUTE': 100,
                'TEN_MINUTE': 100,
                'FIFTEEN_MINUTE': 200,
                'THIRTY_MINUTE': 200,
                'ONE_HOUR': 400,
                'ONE_DAY': 2000
            }
            
            max_days = max_days_map.get(interval, 2000)
            
            # Split into chunks
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=max_days), end_date)
                
                df_chunk = self.download_historical_data(
                    symbol=symbol,
                    from_date=current_start.strftime('%Y-%m-%d'),
                    to_date=current_end.strftime('%Y-%m-%d'),
                    interval=interval,
                    exchange=exchange
                )
                
                if df_chunk is not None and len(df_chunk) > 0:
                    all_data.append(df_chunk)
                
                current_start = current_end
                time.sleep(1)  # Rate limiting
            
            if all_data:
                # Combine all chunks
                combined_df = pd.concat(all_data, axis=0)
                combined_df = combined_df.sort_index()
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                print(f"✅ Total {len(combined_df)} records downloaded for {symbol}")
                return combined_df
            else:
                print(f"❌ No data downloaded for {symbol}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading multiple days for {symbol}: {e}")
            return None
    
    def save_data(self, df, symbol, interval="ONE_DAY"):
        """Save downloaded data to CSV file."""
        try:
            if df is not None and len(df) > 0:
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                # Save to CSV
                filename = f"data/{symbol}_{interval.lower()}_data.csv"
                df.to_csv(filename)
                print(f"💾 Data saved to {filename}")
                return filename
            else:
                print("❌ No data to save")
                return None
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return None
    
    def get_latest_data(self, symbol, days=30, interval="ONE_DAY"):
        """
        Get latest data for a symbol.
        
        Args:
            symbol (str): Stock symbol
            days (int): Number of days to download
            interval (str): Data interval
        
        Returns:
            pandas.DataFrame: Latest data
        """
        try:
            df = self.download_multiple_days(symbol, days, interval)
            if df is not None:
                self.save_data(df, symbol, interval)
            return df
        except Exception as e:
            print(f"❌ Error getting latest data for {symbol}: {e}")
            return None

def main():
    """Example usage of the Angel One Data Downloader."""
    print("🚀 Angel One Historical Data Downloader")
    print("=" * 50)
    
    # Initialize downloader (you need to provide your API credentials)
    downloader = AngelOneDataDownloader(
        api_key="YOUR_API_KEY",
        auth_token="YOUR_AUTH_TOKEN",
        client_ip="127.0.0.1",
        mac_address="00:00:00:00:00:00"
    )
    
    # Example: Download data for RELIANCE
    symbol = "RELIANCE"
    days = 365
    
    print(f"📊 Downloading {days} days of data for {symbol}...")
    
    df = downloader.get_latest_data(symbol, days, "ONE_DAY")
    
    if df is not None:
        print(f"✅ Successfully downloaded data for {symbol}")
        print(f"📈 Data shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print("\n📊 Sample data:")
        print(df.head())
    else:
        print(f"❌ Failed to download data for {symbol}")

if __name__ == "__main__":
    main()
