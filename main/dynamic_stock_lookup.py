#!/usr/bin/env python3
"""
Dynamic Stock Lookup System for Angel One
Finds any stock symbol in the angel_one_symbols.json and returns the equity token
"""

import json
import re
from typing import Dict, List, Optional, Tuple

class DynamicStockLookup:
    """Dynamic stock lookup system for Angel One API"""
    
    def __init__(self, symbols_file: str = 'angel_one_symbols.json'):
        """Initialize with symbols file"""
        self.symbols_file = symbols_file
        self.symbols_data = None
        self._load_symbols()
    
    def _load_symbols(self):
        """Load symbols from JSON file"""
        try:
            with open(self.symbols_file, 'r', encoding='utf-8') as f:
                self.symbols_data = json.load(f)
            print(f"✅ Loaded {len(self.symbols_data):,} symbols from {self.symbols_file}")
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            self.symbols_data = []
    
    def clean_stock_name(self, stock_name: str) -> str:
        """Clean and normalize stock name for searching"""
        # Remove common suffixes and clean the name
        stock_name = stock_name.upper().strip()
        
        # Remove common suffixes
        suffixes_to_remove = ['.NS', '.BSE', '.NSE', '.BO', '.IN']
        for suffix in suffixes_to_remove:
            if stock_name.endswith(suffix):
                stock_name = stock_name[:-len(suffix)]
        
        return stock_name
    
    def find_stock_equity(self, stock_name: str) -> Optional[Dict]:
        """
        Find equity token for any stock name
        
        Args:
            stock_name: Stock name to search for (e.g., 'RELIANCE', 'TCS', 'INFY')
            
        Returns:
            Dict with token, symbol, name, exchange info or None if not found
        """
        if not self.symbols_data:
            return None
        
        cleaned_name = self.clean_stock_name(stock_name)
        print(f"🔍 Searching for: '{cleaned_name}'")
        
        # Search for exact matches first
        exact_matches = []
        partial_matches = []
        
        for symbol in self.symbols_data:
            symbol_name = symbol.get('name', '').upper()
            symbol_symbol = symbol.get('symbol', '').upper()
            instrument_type = symbol.get('instrumenttype', '').upper()
            exchange = symbol.get('exch_seg', '')
            
            # Look for equity instruments
            if instrument_type in ['EQ', 'EQUITY', '']:
                # Check for exact name match
                if cleaned_name == symbol_name or cleaned_name == symbol_symbol:
                    exact_matches.append(symbol)
                # Check for partial match
                elif cleaned_name in symbol_name or cleaned_name in symbol_symbol:
                    partial_matches.append(symbol)
        
        # Return best match
        if exact_matches:
            best_match = exact_matches[0]
            print(f"✅ Found exact match: {best_match.get('symbol')} - {best_match.get('name')}")
            return best_match
        elif partial_matches:
            best_match = partial_matches[0]
            print(f"✅ Found partial match: {best_match.get('symbol')} - {best_match.get('name')}")
            return best_match
        else:
            print(f"❌ No equity found for: {cleaned_name}")
            return None
    
    def find_stock_with_suffix(self, stock_name: str) -> Optional[Dict]:
        """
        Find stock with common suffixes like -EQ, -BE, etc.
        
        Args:
            stock_name: Stock name to search for
            
        Returns:
            Dict with token info or None if not found
        """
        if not self.symbols_data:
            return None
        
        cleaned_name = self.clean_stock_name(stock_name)
        
        # Try different suffix combinations
        suffixes_to_try = ['-EQ', '-BE', '-CE', '-PE', '']
        
        for suffix in suffixes_to_try:
            search_symbol = f"{cleaned_name}{suffix}"
            print(f"🔍 Trying: {search_symbol}")
            
            for symbol in self.symbols_data:
                symbol_symbol = symbol.get('symbol', '').upper()
                instrument_type = symbol.get('instrumenttype', '').upper()
                
                if symbol_symbol == search_symbol and instrument_type in ['EQ', 'EQUITY', '']:
                    print(f"✅ Found with suffix: {symbol.get('symbol')} - {symbol.get('name')}")
                    return symbol
        
        return None
    
    def get_stock_info(self, stock_name: str) -> Optional[Dict]:
        """
        Get comprehensive stock information
        
        Args:
            stock_name: Stock name to search for
            
        Returns:
            Dict with complete stock information
        """
        # Try direct equity search first
        result = self.find_stock_equity(stock_name)
        if result:
            return result
        
        # Try with suffix search
        result = self.find_stock_with_suffix(stock_name)
        if result:
            return result
        
        # Try fuzzy search
        result = self.fuzzy_search(stock_name)
        if result:
            return result
        
        return None
    
    def fuzzy_search(self, stock_name: str) -> Optional[Dict]:
        """Fuzzy search for stock name"""
        if not self.symbols_data:
            return None
        
        cleaned_name = self.clean_stock_name(stock_name)
        
        # Find all possible matches
        matches = []
        for symbol in self.symbols_data:
            symbol_name = symbol.get('name', '').upper()
            symbol_symbol = symbol.get('symbol', '').upper()
            instrument_type = symbol.get('instrumenttype', '').upper()
            
            if instrument_type in ['EQ', 'EQUITY', '']:
                # Check if any part of the name matches
                if (cleaned_name in symbol_name or 
                    cleaned_name in symbol_symbol or
                    symbol_name in cleaned_name):
                    matches.append(symbol)
        
        if matches:
            # Return the first match
            best_match = matches[0]
            print(f"✅ Found fuzzy match: {best_match.get('symbol')} - {best_match.get('name')}")
            return best_match
        
        return None
    
    def get_token_and_exchange(self, stock_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get token and exchange for a stock
        
        Args:
            stock_name: Stock name to search for
            
        Returns:
            Tuple of (token, exchange) or (None, None) if not found
        """
        stock_info = self.get_stock_info(stock_name)
        
        if stock_info:
            token = stock_info.get('token')
            exchange = stock_info.get('exch_seg')
            symbol = stock_info.get('symbol')
            name = stock_info.get('name')
            
            print(f"📊 Stock Info:")
            print(f"   Symbol: {symbol}")
            print(f"   Name: {name}")
            print(f"   Token: {token}")
            print(f"   Exchange: {exchange}")
            
            return token, exchange
        
        return None, None

def test_dynamic_lookup():
    """Test the dynamic lookup system"""
    print("🧪 Testing Dynamic Stock Lookup System")
    print("=" * 50)
    
    lookup = DynamicStockLookup()
    
    # Test cases
    test_stocks = [
        'RELIANCE',
        'TCS', 
        'INFY',
        'HDFC',
        'NTPC',
        'ONGC',
        'GIPCL',
        'WIPRO',
        'BHARTIARTL'
    ]
    
    for stock in test_stocks:
        print(f"\n🔍 Testing: {stock}")
        print("-" * 30)
        
        token, exchange = lookup.get_token_and_exchange(stock)
        
        if token and exchange:
            print(f"✅ SUCCESS: {stock} -> Token: {token}, Exchange: {exchange}")
        else:
            print(f"❌ FAILED: {stock} not found")
    
    return lookup

if __name__ == "__main__":
    test_dynamic_lookup()
