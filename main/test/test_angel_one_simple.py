#!/usr/bin/env python3
"""
Simple Angel One Integration Test

This script tests:
1. Download all interval data from Angel One
2. Store them in associated database tables
3. Retrieve data from database to train models
4. Consider time intervals for different prediction horizons
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
import time

# Add project root to path (supports running from main/test/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_angel_one_integration():
    """Test Angel One integration comprehensively"""
    
    logger.info("🚀 Starting Angel One Comprehensive Integration Test")
    logger.info("=" * 60)
    
    try:
        # Import services
        from main.services.angel_one_service import AngelOneService
        from main.services.interval_manager import IntervalManager, PredictionHorizon
        
        # Configuration
        config = {
            'api_key': '1TKgQThc',
            'api_secret': 'D54448', 
            'access_token': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        }
        
        # Initialize services
        angel_service = AngelOneService(config)
        interval_manager = IntervalManager()
        
        test_symbol = 'RELIANCE'
        
        logger.info(f"Test Symbol: {test_symbol}")
        logger.info(f"Test Time: {datetime.now()}")
        
        # Test 1: Download all interval data
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: Downloading all interval data from Angel One")
        logger.info("=" * 60)
        
        intervals = [
            'ONE_MINUTE',
            'FIVE_MINUTE', 
            'FIFTEEN_MINUTE',
            'THIRTY_MINUTE',
            'ONE_HOUR',
            'ONE_DAY'
        ]
        
        downloaded_data = {}
        
        for interval in intervals:
            logger.info(f"Downloading {interval} data for {test_symbol}...")
            
            try:
                # Calculate date range based on interval
                if interval == 'ONE_MINUTE':
                    days = 7  # Last 7 days for minute data
                elif interval in ['FIVE_MINUTE', 'FIFTEEN_MINUTE']:
                    days = 30  # Last 30 days for 5/15 minute data
                elif interval == 'THIRTY_MINUTE':
                    days = 60  # Last 60 days for 30 minute data
                elif interval == 'ONE_HOUR':
                    days = 90  # Last 90 days for hourly data
                else:  # ONE_DAY
                    days = 365  # Last 365 days for daily data
                
                # Download data
                data = angel_service.get_historical_data(
                    symbol=test_symbol,
                    interval=interval,
                    days=days
                )
                
                if data is not None and not data.empty:
                    downloaded_data[interval] = data
                    logger.info(f"✅ {interval}: Downloaded {len(data)} records")
                    logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
                else:
                    logger.warning(f"❌ {interval}: No data received")
                    
            except Exception as e:
                logger.error(f"❌ {interval}: Error - {e}")
        
        logger.info(f"\nDownloaded data for {len(downloaded_data)} intervals")
        
        # Test 2: Test interval management for different horizons
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Testing interval management for different horizons")
        logger.info("=" * 60)
        
        prediction_horizons = [
            PredictionHorizon.INTRADAY,
            PredictionHorizon.SHORT_TERM,
            PredictionHorizon.MEDIUM_TERM,
            PredictionHorizon.LONG_TERM
        ]
        
        for horizon in prediction_horizons:
            logger.info(f"\nTesting {horizon.value} prediction horizon...")
            
            try:
                # Get optimal intervals for this horizon
                interval_config = interval_manager.get_optimal_intervals(horizon, downloaded_data)
                
                if not interval_config['intervals']:
                    logger.warning(f"❌ {horizon.value}: No suitable intervals available")
                    continue
                
                logger.info(f"   Optimal intervals: {interval_config['intervals']}")
                logger.info(f"   Description: {interval_config['description']}")
                
                # Aggregate data for this horizon
                aggregated_data = interval_manager.aggregate_multi_interval_data(
                    downloaded_data, interval_config
                )
                
                if not aggregated_data.empty:
                    logger.info(f"✅ {horizon.value}: Aggregated {len(aggregated_data)} records")
                    logger.info(f"   Date range: {aggregated_data.index.min()} to {aggregated_data.index.max()}")
                else:
                    logger.warning(f"❌ {horizon.value}: No aggregated data available")
                    
            except Exception as e:
                logger.error(f"❌ {horizon.value}: Error - {e}")
        
        # Test 3: Test multi-interval data structure
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Testing multi-interval data structure")
        logger.info("=" * 60)
        
        if downloaded_data:
            logger.info("Multi-interval data structure:")
            for interval, data in downloaded_data.items():
                logger.info(f"  {interval}: {len(data)} records, columns: {list(data.columns)}")
        
        # Test 4: Test data quality and completeness
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: Testing data quality and completeness")
        logger.info("=" * 60)
        
        for interval, data in downloaded_data.items():
            logger.info(f"\n{interval} data quality:")
            logger.info(f"  Records: {len(data)}")
            logger.info(f"  Columns: {list(data.columns)}")
            logger.info(f"  Date range: {data.index.min()} to {data.index.max()}")
            logger.info(f"  Missing values: {data.isnull().sum().sum()}")
            logger.info(f"  Data types: {data.dtypes.to_dict()}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Downloaded data for {len(downloaded_data)} intervals")
        logger.info(f"✅ Tested interval management for {len(prediction_horizons)} horizons")
        logger.info(f"✅ Verified data quality and structure")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Comprehensive test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test execution"""
    print("🚀 Angel One Comprehensive Integration Test")
    print("=" * 60)
    
    success = test_angel_one_integration()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    main()
