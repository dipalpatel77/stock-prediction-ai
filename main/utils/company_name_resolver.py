#!/usr/bin/env python3
"""
Company Name Resolver
Resolves stock symbols to company names from the Angel One symbols JSON file
"""

import json
import os
from typing import Optional, Dict, List

class CompanyNameResolver:
    """Resolves stock symbols to company names"""
    
    def __init__(self, symbols_file_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'angel_one_symbols.json')):
        """
        Initialize the company name resolver
        
        Args:
            symbols_file_path: Path to the Angel One symbols JSON file
        """
        self.symbols_file_path = symbols_file_path
        self.symbols_data = None
        self.symbol_to_name_map = {}
        self._load_symbols()
    
    def _load_symbols(self):
        """Load symbols from JSON file"""
        try:
            if os.path.exists(self.symbols_file_path):
                with open(self.symbols_file_path, 'r', encoding='utf-8') as f:
                    self.symbols_data = json.load(f)
                
                # Create symbol to name mapping
                for entry in self.symbols_data:
                    if 'symbol' in entry and 'name' in entry:
                        symbol = entry['symbol']
                        name = entry['name']
                        self.symbol_to_name_map[symbol] = name
                
                print(f"✅ Loaded {len(self.symbol_to_name_map)} symbols from {self.symbols_file_path}")
            else:
                print(f"⚠️ Symbols file not found: {self.symbols_file_path}")
                
        except Exception as e:
            print(f"❌ Error loading symbols: {e}")
            self.symbols_data = []
    
    def get_company_name(self, symbol: str) -> Optional[str]:
        """
        Get company name for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'TCS', 'RELIANCE')
            
        Returns:
            Company name if found, None otherwise
        """
        # Direct lookup
        if symbol in self.symbol_to_name_map:
            return self.symbol_to_name_map[symbol]
        
        # Try case-insensitive lookup
        symbol_lower = symbol.lower()
        for sym, name in self.symbol_to_name_map.items():
            if sym.lower() == symbol_lower:
                return name
        
        # Try partial match
        for sym, name in self.symbol_to_name_map.items():
            if symbol_lower in sym.lower() or sym.lower() in symbol_lower:
                return name
        
        return None
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get detailed information for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with symbol information
        """
        company_name = self.get_company_name(symbol)
        
        return {
            'symbol': symbol,
            'company_name': company_name,
            'display_name': f"{company_name} ({symbol})" if company_name else symbol,
            'found': company_name is not None
        }
    
    def search_symbols(self, query: str) -> List[Dict]:
        """
        Search for symbols matching a query
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols with company names
        """
        results = []
        query_lower = query.lower()
        
        for symbol, name in self.symbol_to_name_map.items():
            if (query_lower in symbol.lower() or 
                query_lower in name.lower()):
                results.append({
                    'symbol': symbol,
                    'company_name': name,
                    'display_name': f"{name} ({symbol})"
                })
        
        return results[:10]  # Limit to 10 results
    
    def get_popular_stocks(self) -> List[Dict]:
        """
        Get list of popular Indian stocks
        
        Returns:
            List of popular stocks with company names
        """
        popular_symbols = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ITC', 'KOTAKBANK', 'BHARTIARTL', 'ASIANPAINT', 'MARUTI',
            'TITAN', 'SUNPHARMA', 'NESTLEIND', 'ULTRACEMCO', 'POWERGRID',
            'NTPC', 'ONGC', 'COALINDIA', 'TECHM', 'WIPRO'
        ]
        
        results = []
        for symbol in popular_symbols:
            company_name = self.get_company_name(symbol)
            if company_name:
                results.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'display_name': f"{company_name} ({symbol})"
                })
        
        return results

# Global instance
company_resolver = CompanyNameResolver()

def get_company_name(symbol: str) -> Optional[str]:
    """Get company name for a symbol"""
    return company_resolver.get_company_name(symbol)

def get_symbol_info(symbol: str) -> Dict:
    """Get symbol information"""
    return company_resolver.get_symbol_info(symbol)

def get_display_name(symbol: str) -> str:
    """Get display name for a symbol"""
    info = get_symbol_info(symbol)
    return info['display_name']
