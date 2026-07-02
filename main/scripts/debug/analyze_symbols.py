#!/usr/bin/env python3
"""
Script to analyze the angel_one_symbols.json file structure
"""

import json
import os

def analyze_symbols_file():
    """Analyze the angel_one_symbols.json file"""
    try:
        # Check file size
        file_size = os.path.getsize('angel_one_symbols.json')
        print(f"📊 File Analysis:")
        print(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        # Load JSON data
        with open('angel_one_symbols.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Total symbols: {len(data):,}")
        
        # Analyze structure
        if data:
            sample = data[0]
            print(f"\n📋 Sample Record Structure:")
            for key, value in sample.items():
                print(f"   {key}: {value}")
        
        # Analyze data types
        print(f"\n📈 Data Distribution:")
        
        # Count by exchange
        exchanges = {}
        for symbol in data:
            exch = symbol.get('exch_seg', 'Unknown')
            exchanges[exch] = exchanges.get(exch, 0) + 1
        
        for exch, count in sorted(exchanges.items()):
            print(f"   {exch}: {count:,} symbols")
        
        # Count by instrument type
        instrument_types = {}
        for symbol in data:
            inst_type = symbol.get('instrumenttype', 'Unknown')
            instrument_types[inst_type] = instrument_types.get(inst_type, 0) + 1
        
        print(f"\n🔧 Instrument Types:")
        for inst_type, count in sorted(instrument_types.items()):
            if count > 100:  # Only show significant counts
                print(f"   {inst_type}: {count:,} symbols")
        
        # Find some specific examples
        print(f"\n🔍 Sample Symbols:")
        equity_symbols = [s for s in data if s.get('instrumenttype', '').upper() in ['EQ', 'EQUITY', ''] and s.get('exch_seg') == 'NSE'][:5]
        for symbol in equity_symbols:
            print(f"   {symbol.get('symbol')} ({symbol.get('name')}) - Token: {symbol.get('token')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Angel One Symbols File Analyzer")
    print("=" * 50)
    analyze_symbols_file()
