#!/usr/bin/env python3
"""
Debug script to find the correct ITC token and exchange
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.angel_one_service import AngelOneService
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ITCTokenFinder:
    def __init__(self):
        self.angel_service = AngelOneService({
            'api_key': '1TKgQThc',
            'api_secret': 'D54448', 
            'client_code': 'D54448',
            'client_pin': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        })
        
    def test_token_exchange(self, token, exchange):
        """Test a specific token and exchange combination"""
        try:
            # Authenticate first
            if not self.angel_service.authenticate():
                logger.error("Authentication failed")
                return False
                
            # Test with a recent date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d 09:15')
            to_date = end_date.strftime('%Y-%m-%d 15:30')
            
            logger.info(f"Testing ITC token {token} on {exchange} exchange...")
            logger.info(f"Date range: {from_date} to {to_date}")
            
            # Make API request
            data = self.angel_service.get_historical_data(
                symbol_token=token,
                exchange=exchange,
                interval='ONE_DAY',
                from_date=from_date,
                to_date=to_date
            )
            
            if data and len(data) > 0:
                # Check if we got reasonable data
                latest_price = data['Close'].iloc[-1]
                logger.info(f"✅ SUCCESS: Found {len(data)} records, latest price: ₹{latest_price:.2f}")
                
                # Check if price is reasonable for ITC (should be around ₹400-500)
                if 300 <= latest_price <= 600:
                    logger.info(f"✅ PRICE LOOKS CORRECT: ₹{latest_price:.2f} is reasonable for ITC")
                    return True
                else:
                    logger.warning(f"⚠️ PRICE SUSPICIOUS: ₹{latest_price:.2f} doesn't look like ITC price")
                    return False
            else:
                logger.warning(f"❌ No data received for token {token} on {exchange}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing {token} on {exchange}: {e}")
            return False
    
    def find_correct_token(self):
        """Find the correct ITC token and exchange"""
        logger.info("🔍 Searching for correct ITC token and exchange...")
        
        # Potential ITC tokens to test
        potential_tokens = [
            '4241',  # Current token
            '4240',  # Common variation
            '4242',  # Another variation
            '4243',  # Another variation
            '4244',  # Another variation
            '4245',  # Another variation
            '4246',  # Another variation
            '4247',  # Another variation
            '4248',  # Another variation
            '4249',  # Another variation
            '4250',  # Another variation
        ]
        
        # Exchanges to test
        exchanges = ['NSE', 'BSE', 'NFO', 'CDS', 'MCX']
        
        successful_combinations = []
        
        for token in potential_tokens:
            for exchange in exchanges:
                try:
                    if self.test_token_exchange(token, exchange):
                        successful_combinations.append((token, exchange))
                        logger.info(f"🎉 FOUND WORKING COMBINATION: Token {token} on {exchange}")
                except Exception as e:
                    logger.error(f"Error testing {token} on {exchange}: {e}")
                    continue
        
        if successful_combinations:
            logger.info(f"\n✅ Found {len(successful_combinations)} working combinations:")
            for token, exchange in successful_combinations:
                logger.info(f"   Token: {token}, Exchange: {exchange}")
        else:
            logger.error("❌ No working combinations found")
            
        return successful_combinations

def main():
    finder = ITCTokenFinder()
    combinations = finder.find_correct_token()
    
    if combinations:
        print(f"\n🎯 RECOMMENDED ITC CONFIGURATION:")
        print(f"   Symbol Token: {combinations[0][0]}")
        print(f"   Exchange: {combinations[0][1]}")
        print(f"\n📝 Update angel_one_service.py with:")
        print(f"   'ITC': '{combinations[0][0]}',")
        print(f"   And set exchange to '{combinations[0][1]}' in get_historical_data method")
    else:
        print("\n❌ No working ITC configuration found")

if __name__ == "__main__":
    main()
