#!/usr/bin/env python3
"""
Equity Data Downloader
Downloads all equity data using Angel One API for all intervals
Stores data in database and self-deletes after completion
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EquityDataDownloader:
    """Downloads all equity data using Angel One API"""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.intervals = [
            'ONE_MINUTE',
            'FIVE_MINUTE', 
            'FIFTEEN_MINUTE',
            'THIRTY_MINUTE',
            'ONE_HOUR',
            'ONE_DAY'
        ]
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'total_records': 0
        }
    
    def load_equity_symbols(self) -> List[Dict]:
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
    
    def download_single_combination(self, symbol: str, interval: str) -> Dict:
        """Download data for a single symbol-interval combination"""
        result = {
            'symbol': symbol,
            'interval': interval,
            'success': False,
            'records': 0,
            'error': None
        }
        
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
                    result['success'] = True
                    result['records'] = len(data)
                    logger.info(f"✅ {symbol} ({interval}): {len(data)} records")
                else:
                    result['error'] = "Database storage failed"
                    logger.error(f"❌ {symbol} ({interval}): Storage failed")
            else:
                result['error'] = "No data received"
                logger.warning(f"⚠️ {symbol} ({interval}): No data")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ {symbol} ({interval}): {e}")
        
        return result
    
    def download_all_data(self, max_stocks: int = None):
        """Download all equity data with parallel processing"""
        logger.info("🚀 Starting equity data download...")
        
        # Load symbols
        symbols = self.load_equity_symbols()
        if not symbols:
            logger.error("❌ No symbols found!")
            return
        
        # Apply limit
        if max_stocks:
            symbols = symbols[:max_stocks]
            logger.info(f"📊 Limited to {max_stocks} stocks")
        
        # Create task list
        tasks = []
        for symbol_data in symbols:
            symbol = symbol_data.get('name', '')
            if symbol:
                for interval in self.intervals:
                    tasks.append((symbol, interval))
        
        total_tasks = len(tasks)
        logger.info(f"📊 Total tasks: {total_tasks}")
        logger.info(f"📊 Using {self.max_workers} parallel workers")
        
        start_time = datetime.now()
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.download_single_combination, symbol, interval): (symbol, interval)
                for symbol, interval in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                symbol, interval = future_to_task[future]
                try:
                    result = future.result()
                    
                    # Update stats
                    self.stats['processed'] += 1
                    if result['success']:
                        self.stats['successful'] += 1
                        self.stats['total_records'] += result['records']
                    else:
                        self.stats['failed'] += 1
                    
                    # Progress update
                    progress = (self.stats['processed'] / total_tasks) * 100
                    if self.stats['processed'] % 20 == 0:
                        logger.info(f"📊 Progress: {progress:.1f}% ({self.stats['processed']}/{total_tasks})")
                        
                except Exception as e:
                    logger.error(f"❌ Task failed: {e}")
                    self.stats['failed'] += 1
                    self.stats['processed'] += 1
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Print summary
        self.print_summary(execution_time)
    
    def print_summary(self, execution_time: float):
        """Print final summary"""
        logger.info("\n" + "="*60)
        logger.info("🎉 DOWNLOAD COMPLETED!")
        logger.info("="*60)
        logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        logger.info(f"📊 Total tasks processed: {self.stats['processed']}")
        logger.info(f"✅ Successful downloads: {self.stats['successful']}")
        logger.info(f"❌ Failed downloads: {self.stats['failed']}")
        logger.info(f"📈 Total records downloaded: {self.stats['total_records']:,}")
        
        if execution_time > 0:
            records_per_second = self.stats['total_records'] / execution_time
            logger.info(f"🚀 Download speed: {records_per_second:.1f} records/second")
        
        # Show database summary
        self.show_database_summary()
    
    def show_database_summary(self):
        """Show database summary"""
        try:
            import sqlite3
            
            conn = sqlite3.connect('default.db')
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM angel_one_data")
            total_records = cursor.fetchone()[0]
            
            # Get stocks count
            cursor.execute("SELECT COUNT(DISTINCT ticker) FROM angel_one_data")
            total_stocks = cursor.fetchone()[0]
            
            # Get records by interval
            cursor.execute("""
                SELECT interval_type, COUNT(*) as records 
                FROM angel_one_data 
                GROUP BY interval_type 
                ORDER BY records DESC
            """)
            interval_stats = cursor.fetchall()
            
            logger.info(f"\n📊 Database Summary:")
            logger.info(f"   Total records: {total_records:,}")
            logger.info(f"   Total stocks: {total_stocks}")
            logger.info(f"   Records by interval:")
            for interval, records in interval_stats:
                logger.info(f"     {interval}: {records:,} records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error showing database summary: {e}")

def main():
    """Main function"""
    print("🚀 Equity Data Downloader")
    print("="*40)
    print("Downloads ALL equity data using Angel One API")
    print("for ALL intervals and stores in database.")
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
    
    workers_input = input("Number of parallel workers (press Enter for 3): ")
    max_workers = 3
    if workers_input.strip():
        try:
            max_workers = int(workers_input)
        except ValueError:
            print("⚠️ Invalid number, using 3 workers")
    
    # Confirmation
    response = input(f"\nProceed with download? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled.")
        return
    
    try:
        # Initialize downloader
        downloader = EquityDataDownloader(max_workers=max_workers)
        
        # Start download
        downloader.download_all_data(max_stocks)
        
        print("\n🎉 Download completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Self-delete the script
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

