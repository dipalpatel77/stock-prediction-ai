#!/usr/bin/env python3
"""
Download All Equity Data Script
Downloads all equity data using Angel One API for all intervals and stores in database
Self-deletes after completion
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_all_equity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AllEquityDataDownloader:
    """Downloads all equity data using Angel One API"""
    
    def __init__(self):
        self.intervals = [
            'ONE_MINUTE',
            'FIVE_MINUTE', 
            'FIFTEEN_MINUTE',
            'THIRTY_MINUTE',
            'ONE_HOUR',
            'ONE_DAY'
        ]
        self.downloaded_stocks = []
        self.failed_stocks = []
        self.total_records = 0
        
    def load_angel_one_symbols(self) -> List[Dict]:
        """Load Angel One symbols from JSON file"""
        try:
            with open('angel_one_symbols.json', 'r', encoding='utf-8') as f:
                symbols_data = json.load(f)
            
            # Filter for equity symbols only
            equity_symbols = []
            for symbol in symbols_data:
                if symbol.get('instrumenttype') == 'EQ':  # Equity only
                    equity_symbols.append(symbol)
            
            logger.info(f"Found {len(equity_symbols)} equity symbols")
            return equity_symbols
            
        except Exception as e:
            logger.error(f"Error loading Angel One symbols: {e}")
            return []
    
    def download_stock_data(self, symbol: str, interval: str, days: int = 365) -> pd.DataFrame:
        """Download stock data for a specific symbol and interval"""
        try:
            from main.services.angel_one_service import AngelOneService
            
            # Initialize Angel One service
            angel_service = AngelOneService()
            
            # Download data
            data = angel_service.get_historical_data(
                symbol=symbol,
                interval=interval,
                days=days
            )
            
            if data is not None and not data.empty:
                logger.info(f"Downloaded {len(data)} records for {symbol} ({interval})")
                return data
            else:
                logger.warning(f"No data received for {symbol} ({interval})")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading {symbol} ({interval}): {e}")
            return pd.DataFrame()
    
    def store_data_in_database(self, symbol: str, data: pd.DataFrame, interval: str) -> bool:
        """Store data in database"""
        try:
            from main.services.database_manager import DatabaseManager
            
            # Initialize database manager
            config = {
                'database_url': 'sqlite:///default.db',
                'max_connections': 10,
                'min_connections': 2,
                'connection_timeout': 30,
                'query_timeout': 60,
                'enable_query_cache': True
            }
            
            db_manager = DatabaseManager(config)
            
            # Store data
            success = db_manager.store_stock_data(
                ticker=symbol,
                data=data,
                source='angel_one',
                interval=interval
            )
            
            if success:
                logger.info(f"Successfully stored {len(data)} records for {symbol} ({interval})")
                return True
            else:
                logger.error(f"Failed to store data for {symbol} ({interval})")
                return False
                
        except Exception as e:
            logger.error(f"Error storing {symbol} ({interval}): {e}")
            return False
    
    def download_all_equity_data(self, max_stocks: int = None):
        """Download all equity data for all intervals"""
        logger.info("🚀 Starting comprehensive equity data download...")
        
        # Load symbols
        symbols = self.load_angel_one_symbols()
        if not symbols:
            logger.error("No symbols found!")
            return
        
        # Limit stocks if specified
        if max_stocks:
            symbols = symbols[:max_stocks]
            logger.info(f"Limited to {max_stocks} stocks for testing")
        
        total_stocks = len(symbols)
        total_combinations = total_stocks * len(self.intervals)
        
        logger.info(f"📊 Total stocks: {total_stocks}")
        logger.info(f"📊 Total intervals: {len(self.intervals)}")
        logger.info(f"📊 Total combinations: {total_combinations}")
        
        start_time = datetime.now()
        processed = 0
        successful = 0
        
        for i, symbol_data in enumerate(symbols):
            symbol = symbol_data.get('name', '')
            company_name = symbol_data.get('companyname', '')
            
            if not symbol:
                continue
                
            logger.info(f"\n📈 Processing {symbol} ({company_name}) - {i+1}/{total_stocks}")
            
            stock_success = True
            
            for interval in self.intervals:
                try:
                    logger.info(f"  🔄 Downloading {symbol} ({interval})...")
                    
                    # Download data
                    data = self.download_stock_data(symbol, interval)
                    
                    if not data.empty:
                        # Store in database
                        if self.store_data_in_database(symbol, data, interval):
                            successful += 1
                            self.total_records += len(data)
                        else:
                            stock_success = False
                    else:
                        logger.warning(f"  ⚠️ No data for {symbol} ({interval})")
                        stock_success = False
                    
                    processed += 1
                    
                    # Progress update
                    progress = (processed / total_combinations) * 100
                    logger.info(f"  📊 Progress: {progress:.1f}% ({processed}/{total_combinations})")
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"  ❌ Error processing {symbol} ({interval}): {e}")
                    stock_success = False
                    processed += 1
            
            if stock_success:
                self.downloaded_stocks.append(symbol)
                logger.info(f"  ✅ Successfully processed {symbol}")
            else:
                self.failed_stocks.append(symbol)
                logger.warning(f"  ⚠️ Some issues with {symbol}")
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Final summary
        self.print_summary(execution_time, successful, processed)
    
    def print_summary(self, execution_time: float, successful: int, processed: int):
        """Print final summary"""
        logger.info("\n" + "="*60)
        logger.info("🎉 DOWNLOAD COMPLETED!")
        logger.info("="*60)
        logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        logger.info(f"📊 Total combinations processed: {processed}")
        logger.info(f"✅ Successful downloads: {successful}")
        logger.info(f"❌ Failed downloads: {processed - successful}")
        logger.info(f"📈 Total records downloaded: {self.total_records:,}")
        logger.info(f"🏢 Stocks successfully processed: {len(self.downloaded_stocks)}")
        logger.info(f"⚠️ Stocks with issues: {len(self.failed_stocks)}")
        
        if self.downloaded_stocks:
            logger.info(f"\n✅ Successfully processed stocks:")
            for stock in self.downloaded_stocks[:10]:  # Show first 10
                logger.info(f"   - {stock}")
            if len(self.downloaded_stocks) > 10:
                logger.info(f"   ... and {len(self.downloaded_stocks) - 10} more")
        
        if self.failed_stocks:
            logger.info(f"\n⚠️ Stocks with issues:")
            for stock in self.failed_stocks[:10]:  # Show first 10
                logger.info(f"   - {stock}")
            if len(self.failed_stocks) > 10:
                logger.info(f"   ... and {len(self.failed_stocks) - 10} more")
        
        # Database summary
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
            
            # Get records by ticker
            cursor.execute("""
                SELECT ticker, COUNT(*) as records 
                FROM angel_one_data 
                GROUP BY ticker 
                ORDER BY records DESC 
                LIMIT 10
            """)
            top_stocks = cursor.fetchall()
            
            logger.info(f"\n📊 Database Summary:")
            logger.info(f"   Total records: {total_records:,}")
            logger.info(f"   Total stocks: {total_stocks}")
            logger.info(f"   Top stocks by records:")
            for ticker, records in top_stocks:
                logger.info(f"     {ticker}: {records:,} records")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error showing database summary: {e}")

def main():
    """Main function"""
    print("🚀 Angel One Equity Data Downloader")
    print("="*50)
    print("This script will download ALL equity data using Angel One API")
    print("for ALL intervals and store it in the database.")
    print("The script will self-delete after completion.")
    print("="*50)
    
    # Ask for confirmation
    response = input("\nDo you want to proceed? This may take several hours. (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled.")
        return
    
    # Ask for max stocks limit (for testing)
    max_stocks_input = input("\nEnter max number of stocks to download (press Enter for all): ")
    max_stocks = None
    if max_stocks_input.strip():
        try:
            max_stocks = int(max_stocks_input)
            print(f"📊 Limited to {max_stocks} stocks")
        except ValueError:
            print("⚠️ Invalid number, downloading all stocks")
    
    try:
        # Initialize downloader
        downloader = AllEquityDataDownloader()
        
        # Start download
        downloader.download_all_equity_data(max_stocks)
        
        print("\n🎉 Download completed successfully!")
        print("📊 Check the database and logs for details.")
        
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

