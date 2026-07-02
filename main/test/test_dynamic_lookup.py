#!/usr/bin/env python3
"""
Test script to check dynamic lookup and data processor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.services.angel_one_service import AngelOneService
from main.pipeline.data_processor import DataProcessor

def test_dynamic_lookup():
    """Test dynamic lookup functionality"""
    print("🧪 Testing Dynamic Lookup System")
    print("=" * 50)
    
    # Test dynamic lookup
    config = {
        'api_key': '1TKgQThc', 
        'api_secret': 'D54448', 
        'access_token': '2251', 
        'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
    }
    
    service = AngelOneService(config)
    token, exchange = service.get_dynamic_token_and_exchange('ADANIPORTS')
    print(f'ADANIPORTS: Token={token}, Exchange={exchange}')
    
    # Test data processor
    processor = DataProcessor('ADANIPORTS')
    print(f'Is Indian stock: {processor._is_indian_stock("ADANIPORTS")}')
    print(f'Angel manager available: {processor.angel_manager is not None}')
    
    if processor.angel_manager:
        print(f'Angel manager type: {type(processor.angel_manager)}')
        if hasattr(processor.angel_manager, 'angel_service'):
            print(f'Angel service available: {processor.angel_manager.angel_service is not None}')

if __name__ == "__main__":
    test_dynamic_lookup()
