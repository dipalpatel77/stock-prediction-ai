#!/usr/bin/env python3
"""
Script to find ONGC symbol token from Angel One symbol master list
"""

import json
import pandas as pd
from typing import List, Dict, Any

def load_symbol_data(file_path: str) -> List[Dict[str, Any]]:
    """Load symbol data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} symbols from {file_path}")
        return data
    except Exception as e:
        print(f"❌ Error loading symbol data: {e}")
        return []

def search_ongc_symbols(symbols: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Search for ONGC related symbols"""
    ongc_symbols = []
    
    for symbol in symbols:
        name = symbol.get('name', '').upper()
        symbol_name = symbol.get('symbol', '').upper()
        
        # Search for ONGC in name or symbol
        if 'ONGC' in name or 'ONGC' in symbol_name:
            ongc_symbols.append(symbol)
    
    return ongc_symbols

def display_ongc_results(ongc_symbols: List[Dict[str, Any]]):
    """Display ONGC symbol results"""
    if not ongc_symbols:
        print("❌ No ONGC symbols found")
        return
    
    print(f"\n🔍 Found {len(ongc_symbols)} ONGC related symbols:")
    print("=" * 80)
    
    for i, symbol in enumerate(ongc_symbols, 1):
        print(f"\n{i}. Symbol Details:")
        print(f"   Name: {symbol.get('name', 'N/A')}")
        print(f"   Symbol: {symbol.get('symbol', 'N/A')}")
        print(f"   Token: {symbol.get('token', 'N/A')}")
        print(f"   Exchange: {symbol.get('exch_seg', 'N/A')}")
        print(f"   Instrument Type: {symbol.get('instrumenttype', 'N/A')}")
        print(f"   Lot Size: {symbol.get('lotsize', 'N/A')}")
        print(f"   Tick Size: {symbol.get('tick_size', 'N/A')}")
        print(f"   Expiry: {symbol.get('expiry', 'N/A')}")
        print(f"   Strike: {symbol.get('strike', 'N/A')}")

def find_best_ongc_token(ongc_symbols: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the best ONGC token for equity trading"""
    if not ongc_symbols:
        return None
    
    # Filter for equity instruments
    equity_symbols = [
        s for s in ongc_symbols 
        if s.get('instrumenttype', '').upper() in ['EQ', 'EQUITY', ''] 
        and s.get('exch_seg', '').upper() in ['NSE', 'BSE']
    ]
    
    if equity_symbols:
        print(f"\n🎯 Best ONGC token for equity trading:")
        best = equity_symbols[0]
        print(f"   Token: {best.get('token')}")
        print(f"   Symbol: {best.get('symbol')}")
        print(f"   Exchange: {best.get('exch_seg')}")
        print(f"   Name: {best.get('name')}")
        return best
    
    # If no equity found, return the first one
    print(f"\n⚠️ No equity instruments found, using first available:")
    best = ongc_symbols[0]
    print(f"   Token: {best.get('token')}")
    print(f"   Symbol: {best.get('symbol')}")
    print(f"   Exchange: {best.get('exch_seg')}")
    print(f"   Name: {best.get('name')}")
    return best

def main():
    """Main function"""
    print("🔍 Angel One Symbol Token Finder for ONGC")
    print("=" * 50)
    
    # Load symbol data
    symbols = load_symbol_data('angel_one_symbols.json')
    if not symbols:
        return
    
    # Search for ONGC symbols
    ongc_symbols = search_ongc_symbols(symbols)
    
    # Display results
    display_ongc_results(ongc_symbols)
    
    # Find best token
    best_token = find_best_ongc_token(ongc_symbols)
    
    if best_token:
        print(f"\n✅ Recommended ONGC Configuration:")
        print(f"   Symbol Token: {best_token.get('token')}")
        print(f"   Exchange: {best_token.get('exch_seg')}")
        print(f"   Symbol Name: {best_token.get('symbol')}")
        
        # Generate code snippet for updating the system
        print(f"\n💻 Code to update Angel One service:")
        print(f"   'ONGC': '{best_token.get('token')}'  # {best_token.get('name')}")
    
    print(f"\n📊 Summary:")
    print(f"   Total symbols in master list: {len(symbols)}")
    print(f"   ONGC related symbols found: {len(ongc_symbols)}")

if __name__ == "__main__":
    main()
