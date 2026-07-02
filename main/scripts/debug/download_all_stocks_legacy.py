#!/usr/bin/env python3
"""
Download All Stocks Script
Downloads all equity data using Angel One API and stores in database
Self-deletes after completion
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_equity_symbols() -> List[Dict]:
    """Load equity symbols from Angel One JSON"""
    try:
        with open('angel_one_symbols.json', 'r', encoding='utf-8') as f:
            symbols_data = json.load(f)
        
        # Filter for equity symbols only
        equity_symbols = [
            symbol for symbol in symbols_data 
            if symbol.get('instrumenttype') == 'EQ'
        ]
        
        logger.info(f"Found {len(equity_symbols)} equity symbols")
        return equity_symbols
        
    except Exception as e:
        logger.error(f"Error loading symbols: {e}")
        return []

def download_stock_data(symbol: str, interval: str) -> bool:
    """Download and store data for a single stock-interval combination"""
    try:
        from main.services.angel_one_service import AngelOneService
        from main.services.database_manager import DatabaseManager
        
        # Download data
        angel_service = AngelOneService()
        data = angel_service.get_historical_data(
            symbol=symbol,
            interval=interval,
            days=365
        )
        
        if data is not None and not data.empty:
            # Store in database
            config = {
                'database_url': 'sqlite:///default.db',
                'max_connections': 5,
                'min_connections': 1,
                'connection_timeout': 30
            }
            
            db_manager = DatabaseManager(config)
            success = db_manager.store_stock_data(
                ticker=symbol,
                data=data,
                source='angel_one',
                interval=interval
            )
            
            if success:
                logger.info(f"✅ {symbol} ({interval}): {len(data)} records stored")
                return True
            else:
                logger.error(f"❌ {symbol} ({interval}): Database storage failed")
                return False
        else:
            logger.warning(f"⚠️ {symbol} ({interval}): No data received")
            return False
            
    except Exception as e:
        logger.error(f"❌ {symbol} ({interval}): {e}")
        return False

def main():
    """Main function"""
    print("🚀 Download All Equity Data")
    print("="*40)
    print("This script will download ALL equity data using Angel One API")
    print("for ALL intervals and store in database.")
    print("The script will self-delete after completion.")
    print("="*40)
    
    # Get configuration
    max_stocks_input = input("\nMax stocks to download (press Enter for all): ")
    max_stocks = None
    if max_stocks_input.strip():
        try:
            max_stocks = int(max_stocks_input)
        except ValueError:
            print("⚠️ Invalid number, using all stocks")
    
    # Confirmation
    response = input(f"\nProceed with download? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled.")
        return
    
    # Load symbols
    symbols = load_equity_symbols()
    if not symbols:
        print("❌ No symbols found!")
        return
    
    # Apply limit
    if max_stocks:
        symbols = symbols[:max_stocks]
        print(f"📊 Limited to {max_stocks} stocks")
    
    # Define intervals
    intervals = [
        'ONE_MINUTE',
        'FIVE_MINUTE', 
        'FIFTEEN_MINUTE',
        'THIRTY_MINUTE',
        'ONE_HOUR',
        'ONE_DAY'
    ]
    
    total_combinations = len(symbols) * len(intervals)
    print(f"📊 Total stocks: {len(symbols)}")
    print(f"📊 Total intervals: {len(intervals)}")
    print(f"📊 Total combinations: {total_combinations}")
    
    # Start download
    start_time = datetime.now()
    processed = 0
    successful = 0
    failed = 0
    
    for i, symbol_data in enumerate(symbols):
        symbol = symbol_data.get('name', '')
        company_name = symbol_data.get('companyname', '')
        
        if not symbol:
            continue
        
        print(f"\n📈 Processing {symbol} ({company_name}) - {i+1}/{len(symbols)}")
        
        for interval in intervals:
            try:
                if download_stock_data(symbol, interval):
                    successful += 1
                else:
                    failed += 1
                
                processed += 1
                
                # Progress update
                progress = (processed / total_combinations) * 100
                if processed % 10 == 0:
                    print(f"📊 Progress: {progress:.1f}% ({processed}/{total_combinations})")
                
                # Small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error processing {symbol} ({interval}): {e}")
                failed += 1
                processed += 1
    
    # Final summary
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    print("\n" + "="*50)
    print("🎉 DOWNLOAD COMPLETED!")
    print("="*50)
    print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Total combinations processed: {processed}")
    print(f"✅ Successful downloads: {successful}")
    print(f"❌ Failed downloads: {failed}")
    
    # Show database summary
    try:
        import sqlite3
        conn = sqlite3.connect('default.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM angel_one_data")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM angel_one_data")
        total_stocks = cursor.fetchone()[0]
        
        print(f"\n📊 Database Summary:")
        print(f"   Total records: {total_records:,}")
        print(f"   Total stocks: {total_stocks}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error showing database summary: {e}")
    
    print("\n🎉 Download completed successfully!")
    
    # Self-delete
    try:
        script_path = os.path.abspath(__file__)
        if os.path.exists(script_path):
            print(f"\n🗑️ Self-deleting script: {script_path}")
            os.remove(script_path)
            print("✅ Script deleted successfully!")
    except Exception as e:
        print(f"⚠️ Could not delete script: {e}")

if __name__ == "__main__":
    main()

