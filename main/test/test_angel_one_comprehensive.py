#!/usr/bin/env python3
"""
Comprehensive Angel One Integration Test

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

from main.services.angel_one_service import AngelOneService
from main.services.database_manager import DatabaseManager
from main.services.interval_manager import IntervalManager, PredictionHorizon
from main.pipeline.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AngelOneComprehensiveTest:
    """Comprehensive test for Angel One integration"""
    
    def __init__(self):
        """Initialize test components"""
        self.config = {
            'api_key': '1TKgQThc',
            'api_secret': 'D54448', 
            'access_token': '2251',
            'totp_secret': 'NP4SAXOKMTJQZ4KZP2TBTYXRCE'
        }
        
        # Initialize services
        self.angel_service = AngelOneService(self.config)
        self.db_manager = DatabaseManager()
        self.interval_manager = IntervalManager()
        self.model_trainer = ModelTrainer()
        
        # Test symbol
        self.test_symbol = 'RELIANCE'
        
        logger.info("Angel One Comprehensive Test initialized")
    
    def test_1_download_all_intervals(self):
        """Test 1: Download all interval data from Angel One"""
        logger.info("=" * 60)
        logger.info("TEST 1: Downloading all interval data from Angel One")
        logger.info("=" * 60)
        
        # Define all intervals to test
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
            logger.info(f"Downloading {interval} data for {self.test_symbol}...")
            
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
                data = self.angel_service.get_historical_data(
                    symbol=self.test_symbol,
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
        
        logger.info(f"Downloaded data for {len(downloaded_data)} intervals")
        return downloaded_data
    
    def test_2_store_in_database(self, downloaded_data):
        """Test 2: Store data in associated database tables"""
        logger.info("=" * 60)
        logger.info("TEST 2: Storing data in database tables")
        logger.info("=" * 60)
        
        stored_tables = {}
        
        for interval, data in downloaded_data.items():
            logger.info(f"Storing {interval} data in database...")
            
            try:
                # Create table name based on interval
                table_name = f"angel_one_{interval.lower()}_{self.test_symbol.lower()}"
                
                # Store data in database
                success = self.db_manager.store_historical_data(
                    data=data,
                    symbol=self.test_symbol,
                    interval=interval,
                    table_name=table_name
                )
                
                if success:
                    stored_tables[interval] = table_name
                    logger.info(f"✅ {interval}: Stored in table '{table_name}'")
                else:
                    logger.warning(f"❌ {interval}: Failed to store in database")
                    
            except Exception as e:
                logger.error(f"❌ {interval}: Database error - {e}")
        
        logger.info(f"Stored data in {len(stored_tables)} database tables")
        return stored_tables
    
    def test_3_retrieve_from_database(self, stored_tables):
        """Test 3: Retrieve data from database"""
        logger.info("=" * 60)
        logger.info("TEST 3: Retrieving data from database")
        logger.info("=" * 60)
        
        retrieved_data = {}
        
        for interval, table_name in stored_tables.items():
            logger.info(f"Retrieving {interval} data from database...")
            
            try:
                # Retrieve data from database
                data = self.db_manager.get_historical_data(
                    symbol=self.test_symbol,
                    interval=interval,
                    table_name=table_name
                )
                
                if data is not None and not data.empty:
                    retrieved_data[interval] = data
                    logger.info(f"✅ {interval}: Retrieved {len(data)} records from database")
                    logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
                else:
                    logger.warning(f"❌ {interval}: No data retrieved from database")
                    
            except Exception as e:
                logger.error(f"❌ {interval}: Database retrieval error - {e}")
        
        logger.info(f"Retrieved data for {len(retrieved_data)} intervals from database")
        return retrieved_data
    
    def test_4_time_interval_training(self, retrieved_data):
        """Test 4: Consider time intervals for different prediction horizons"""
        logger.info("=" * 60)
        logger.info("TEST 4: Time interval-based training for different horizons")
        logger.info("=" * 60)
        
        prediction_horizons = [
            PredictionHorizon.INTRADAY,
            PredictionHorizon.SHORT_TERM,
            PredictionHorizon.MEDIUM_TERM,
            PredictionHorizon.LONG_TERM
        ]
        
        training_results = {}
        
        for horizon in prediction_horizons:
            logger.info(f"Training models for {horizon.value} prediction...")
            
            try:
                # Get optimal intervals for this horizon
                interval_config = self.interval_manager.get_optimal_intervals(horizon, retrieved_data)
                
                if not interval_config['intervals']:
                    logger.warning(f"❌ {horizon.value}: No suitable intervals available")
                    continue
                
                logger.info(f"   Using intervals: {interval_config['intervals']}")
                logger.info(f"   Description: {interval_config['description']}")
                
                # Aggregate data for this horizon
                aggregated_data = self.interval_manager.aggregate_multi_interval_data(
                    retrieved_data, interval_config
                )
                
                if aggregated_data.empty:
                    logger.warning(f"❌ {horizon.value}: No aggregated data available")
                    continue
                
                logger.info(f"   Aggregated data: {len(aggregated_data)} records")
                
                # Train models for this horizon
                training_result = self.model_trainer._train_horizon_models(
                    horizon=horizon,
                    data=aggregated_data,
                    symbol=self.test_symbol
                )
                
                if training_result:
                    training_results[horizon.value] = training_result
                    logger.info(f"✅ {horizon.value}: Models trained successfully")
                else:
                    logger.warning(f"❌ {horizon.value}: Model training failed")
                    
            except Exception as e:
                logger.error(f"❌ {horizon.value}: Training error - {e}")
        
        logger.info(f"Training completed for {len(training_results)} horizons")
        return training_results
    
    def run_comprehensive_test(self):
        """Run all tests in sequence"""
        logger.info("🚀 Starting Angel One Comprehensive Test")
        logger.info(f"Test Symbol: {self.test_symbol}")
        logger.info(f"Test Time: {datetime.now()}")
        
        start_time = time.time()
        
        try:
            # Test 1: Download all intervals
            downloaded_data = self.test_1_download_all_intervals()
            
            if not downloaded_data:
                logger.error("❌ No data downloaded. Cannot proceed with remaining tests.")
                return False
            
            # Test 2: Store in database
            stored_tables = self.test_2_store_in_database(downloaded_data)
            
            if not stored_tables:
                logger.error("❌ No data stored in database. Cannot proceed with remaining tests.")
                return False
            
            # Test 3: Retrieve from database
            retrieved_data = self.test_3_retrieve_from_database(stored_tables)
            
            if not retrieved_data:
                logger.error("❌ No data retrieved from database. Cannot proceed with training.")
                return False
            
            # Test 4: Time interval training
            training_results = self.test_4_time_interval_training(retrieved_data)
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE TEST SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Downloaded data for {len(downloaded_data)} intervals")
            logger.info(f"✅ Stored data in {len(stored_tables)} database tables")
            logger.info(f"✅ Retrieved data for {len(retrieved_data)} intervals")
            logger.info(f"✅ Trained models for {len(training_results)} horizons")
            logger.info(f"⏱️  Total test duration: {duration:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Comprehensive test failed: {e}")
            return False

def main():
    """Main test execution"""
    print("🚀 Angel One Comprehensive Integration Test")
    print("=" * 60)
    
    # Create and run test
    test = AngelOneComprehensiveTest()
    success = test.run_comprehensive_test()
    
    if success:
        print("\n✅ All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check logs for details.")
    
    return success

if __name__ == "__main__":
    main()
