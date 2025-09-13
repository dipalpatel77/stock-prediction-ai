#!/usr/bin/env python3
"""
Database Migration Tool
Migrate existing CSV data to database storage
"""

import os
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
import argparse
import sys
from typing import List, Dict, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database_service import DatabaseService
from config.database_config import get_database_config, load_config_from_env

class DataMigrator:
    """Tool for migrating CSV data to database storage."""
    
    def __init__(self, db_config_preset: str = "local"):
        self.db_config = get_database_config(db_config_preset)
        self.db_service = DatabaseService(
            db_type=self.db_config.db_type,
            connection_string=self.db_config.connection_string
        )
        self.data_dir = Path("data")
        self.migration_log = []
    
    def migrate_all_data(self, dry_run: bool = False) -> Dict:
        """Migrate all CSV data files to database."""
        print("🔄 Starting data migration to database...")
        print(f"Database type: {self.db_config.db_type}")
        print(f"Connection: {self.db_config.connection_string}")
        
        if dry_run:
            print("🔍 DRY RUN MODE - No data will be written")
        
        # Find all CSV files
        csv_files = self._find_csv_files()
        print(f"Found {len(csv_files)} CSV files to migrate")
        
        migration_results = {
            'total_files': len(csv_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_records': 0,
            'errors': []
        }
        
        for csv_file in csv_files:
            try:
                result = self._migrate_single_file(csv_file, dry_run)
                migration_results['total_records'] += result['records']
                
                if result['success']:
                    migration_results['successful'] += 1
                elif result['skipped']:
                    migration_results['skipped'] += 1
                else:
                    migration_results['failed'] += 1
                    migration_results['errors'].append(result['error'])
                
            except Exception as e:
                migration_results['failed'] += 1
                migration_results['errors'].append(f"{csv_file}: {str(e)}")
                print(f"❌ Error migrating {csv_file}: {e}")
        
        # Print summary
        self._print_migration_summary(migration_results)
        
        return migration_results
    
    def _find_csv_files(self) -> List[Path]:
        """Find CSV files that contain stock price data."""
        csv_files = []
        
        # Look for raw data files (main stock price data)
        raw_data_files = list(self.data_dir.glob("*_raw_data.csv"))
        csv_files.extend(raw_data_files)
        
        # Look for timeframe data files (processed stock price data)
        timeframe_files = list(self.data_dir.glob("*_*_term_data.csv"))
        csv_files.extend(timeframe_files)
        
        # Look for enhanced data files (processed stock price data)
        enhanced_files = list(self.data_dir.glob("*_partA_partC_enhanced.csv"))
        csv_files.extend(enhanced_files)
        
        # Look for preprocessed data files (processed stock price data)
        preprocessed_files = list(self.data_dir.glob("*_partA_preprocessed.csv"))
        csv_files.extend(preprocessed_files)
        
        # Look for yfinance data files (stock price data)
        yfinance_files = list(self.data_dir.glob("*_yfinance_data.csv"))
        csv_files.extend(yfinance_files)
        
        # Remove duplicates
        csv_files = list(set(csv_files))
        
        return sorted(csv_files)
    
    def _is_stock_price_data(self, file_path: Path) -> bool:
        """Check if a CSV file contains stock price data."""
        try:
            # Read first few rows to check structure
            df = pd.read_csv(file_path, nrows=5)
            
            # Check if it has the required columns for stock price data
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            has_required_columns = all(col in df.columns for col in required_columns)
            
            # Check if Date column exists and can be parsed
            has_valid_date = 'Date' in df.columns
            
            return has_required_columns and has_valid_date
            
        except Exception as e:
            print(f"⚠️ Error checking file {file_path.name}: {e}")
            return False
    
    def _migrate_single_file(self, csv_file: Path, dry_run: bool = False) -> Dict:
        """Migrate a single CSV file to database."""
        try:
            print(f"📁 Processing: {csv_file.name}")
            
            # Check if this file contains stock price data
            if not self._is_stock_price_data(csv_file):
                print(f"⚠️ Skipping {csv_file.name} - not stock price data")
                return {
                    'success': False,
                    'records': 0,
                    'error': 'Not stock price data'
                }
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            if df.empty:
                return {
                    'success': False,
                    'skipped': True,
                    'records': 0,
                    'error': 'Empty file'
                }
            
            # Extract ticker from filename
            ticker = self._extract_ticker_from_filename(csv_file.name)
            
            if not ticker:
                return {
                    'success': False,
                    'skipped': True,
                    'records': 0,
                    'error': 'Could not extract ticker from filename'
                }
            
            # Check if data already exists
            if not dry_run:
                existing_info = self.db_service.get_data_info(ticker)
                if existing_info.get('exists', False):
                    print(f"⚠️ Data already exists for {ticker}, skipping...")
                    return {
                        'success': False,
                        'skipped': True,
                        'records': len(df),
                        'error': 'Data already exists'
                    }
            
            # Prepare data for storage
            if not dry_run:
                success = self.db_service.store_stock_data(
                    ticker=ticker,
                    data=df,
                    data_source="migration"
                )
                
                if success:
                    print(f"✅ Migrated {len(df)} records for {ticker}")
                    return {
                        'success': True,
                        'skipped': False,
                        'records': len(df),
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'skipped': False,
                        'records': len(df),
                        'error': 'Database storage failed'
                    }
            else:
                print(f"🔍 Would migrate {len(df)} records for {ticker}")
                return {
                    'success': True,
                    'skipped': False,
                    'records': len(df),
                    'error': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'skipped': False,
                'records': 0,
                'error': str(e)
            }
    
    def _extract_ticker_from_filename(self, filename: str) -> str:
        """Extract ticker symbol from filename."""
        # Remove file extension
        name = filename.replace('.csv', '')
        
        # Handle different filename patterns
        if '_raw_data' in name:
            ticker = name.replace('_raw_data', '')
        elif '_short_term_data' in name:
            ticker = name.replace('_short_term_data', '')
        elif '_mid_term_data' in name:
            ticker = name.replace('_mid_term_data', '')
        elif '_long_term_data' in name:
            ticker = name.replace('_long_term_data', '')
        else:
            # Try to extract ticker from other patterns
            parts = name.split('_')
            if len(parts) > 0:
                ticker = parts[0]
            else:
                ticker = name
        
        return ticker.upper()
    
    def _print_migration_summary(self, results: Dict):
        """Print migration summary."""
        print("\n" + "="*60)
        print("📊 MIGRATION SUMMARY")
        print("="*60)
        print(f"Total files processed: {results['total_files']}")
        print(f"✅ Successful: {results['successful']}")
        print(f"⚠️ Skipped: {results['skipped']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"📈 Total records: {results['total_records']}")
        
        if results['errors']:
            print(f"\n❌ Errors encountered:")
            for error in results['errors'][:10]:  # Show first 10 errors
                print(f"  • {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")
    
    def verify_migration(self) -> Dict:
        """Verify that migration was successful."""
        print("\n🔍 Verifying migration...")
        
        # Get database stats
        db_stats = self.db_service.get_database_stats()
        print(f"Database stats: {db_stats}")
        
        # Check a few sample tickers
        sample_tickers = ["AAPL", "MSFT", "GOOGL", "TCS.NS", "RELIANCE.NS"]
        verification_results = {
            'total_tickers_checked': len(sample_tickers),
            'tickers_found': 0,
            'tickers_missing': 0,
            'sample_data': {}
        }
        
        for ticker in sample_tickers:
            info = self.db_service.get_data_info(ticker)
            if info.get('exists', False):
                verification_results['tickers_found'] += 1
                verification_results['sample_data'][ticker] = {
                    'records': info['records'],
                    'first_date': info['first_date'],
                    'last_date': info['last_date']
                }
                print(f"✅ {ticker}: {info['records']} records")
            else:
                verification_results['tickers_missing'] += 1
                print(f"❌ {ticker}: Not found")
        
        return verification_results
    
    def cleanup_csv_files(self, backup: bool = True) -> int:
        """Clean up CSV files after successful migration."""
        if not backup:
            print("⚠️ WARNING: This will permanently delete CSV files!")
            response = input("Are you sure? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return 0
        
        print("🧹 Cleaning up CSV files...")
        
        csv_files = self._find_csv_files()
        cleaned_count = 0
        
        for csv_file in csv_files:
            try:
                if backup:
                    # Move to backup directory
                    backup_dir = Path("data/backups/csv_files")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_file = backup_dir / csv_file.name
                    csv_file.rename(backup_file)
                    print(f"📦 Backed up: {csv_file.name}")
                else:
                    # Delete file
                    csv_file.unlink()
                    print(f"🗑️ Deleted: {csv_file.name}")
                
                cleaned_count += 1
                
            except Exception as e:
                print(f"❌ Error cleaning up {csv_file.name}: {e}")
        
        print(f"✅ Cleaned up {cleaned_count} CSV files")
        return cleaned_count

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Migrate CSV data to database")
    parser.add_argument("--preset", default="local", 
                       choices=["development", "production", "cloud", "local"],
                       help="Database configuration preset")
    parser.add_argument("--dry-run", action="store_true",
                       help="Perform a dry run without writing data")
    parser.add_argument("--verify", action="store_true",
                       help="Verify migration after completion")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up CSV files after migration")
    parser.add_argument("--backup", action="store_true", default=True,
                       help="Backup CSV files before cleanup")
    
    args = parser.parse_args()
    
    try:
        # Initialize migrator
        migrator = DataMigrator(args.preset)
        
        # Perform migration
        results = migrator.migrate_all_data(dry_run=args.dry_run)
        
        # Verify if requested
        if args.verify and not args.dry_run:
            verification = migrator.verify_migration()
        
        # Cleanup if requested
        if args.cleanup and not args.dry_run and results['successful'] > 0:
            migrator.cleanup_csv_files(backup=args.backup)
        
        print("\n✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
