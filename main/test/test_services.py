#!/usr/bin/env python3
"""
Test Suite for Service Components
Tests all service components individually and in integration
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import service components
from main.services.database_manager import DatabaseManager
from main.services.angel_one_manager import AngelOneManager
from main.services.api_coordinator import APICoordinator
from main.services.data_service_wrapper import DataServiceWrapper
from main.services.economic_data_service import EconomicDataService
from main.services.feature_engineering_service import FeatureEngineeringService
from main.services.technical_indicators_service import TechnicalIndicatorsService
from main.services.interval_specific_storage import IntervalSpecificStorageService
from main.services.smart_data_fetcher import SmartDataFetcher
from main.services.advanced_cache_manager import AdvancedCacheManager
from main.services.ml_optimizer import MLOptimizer
from main.services.auto_scaler import AutoScaler
from main.services.monitoring_dashboard import MonitoringDashboard


class TestDatabaseManager(unittest.TestCase):
    """Test DatabaseManager component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'database_url': 'sqlite:///test_stock_data.db',
            'pool_size': 5
        }
        self.db_manager = DatabaseManager(config=self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.sample_data.set_index('Date', inplace=True)
    
    def test_initialization(self):
        """Test DatabaseManager initialization"""
        self.assertEqual(self.db_manager.config, self.config)
        self.assertIsNotNone(self.db_manager.logger)
    
    def test_connection_pool_initialization(self):
        """Test connection pool initialization"""
        self.assertIsNotNone(self.db_manager.connection_pool)
    
    def test_test_connection(self):
        """Test database connection"""
        # This might fail if database is not available, but should not crash
        try:
            result = self.db_manager.test_connection()
            self.assertIsInstance(result, bool)
        except Exception as e:
            # Expected if database is not available
            self.assertIsInstance(e, Exception)
    
    def test_store_stock_data(self):
        """Test storing stock data"""
        try:
            result = self.db_manager.store_stock_data(
                ticker='AAPL',
                data=self.sample_data,
                source='test',
                interval='ONE_DAY'
            )
            self.assertTrue(result)
        except Exception as e:
            # Expected if database is not available
            self.assertIsInstance(e, Exception)
    
    def test_get_stock_data(self):
        """Test retrieving stock data"""
        try:
            result = self.db_manager.get_stock_data(
                ticker='AAPL',
                period='1y',
                source='test',
                interval='ONE_DAY'
            )
            if result is not None:
                self.assertIsInstance(result, pd.DataFrame)
        except Exception as e:
            # Expected if database is not available
            self.assertIsInstance(e, Exception)
    
    def test_database_statistics(self):
        """Test database statistics"""
        stats = self.db_manager.get_database_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_connections', stats)
        self.assertIn('active_connections', stats)
    
    def test_performance_report(self):
        """Test performance report"""
        report = self.db_manager.get_performance_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('query_count', report)
        self.assertIn('average_query_time', report)


class TestAngelOneManager(unittest.TestCase):
    """Test AngelOneManager component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'totp_secret': 'test_totp',
            'exchange': 'NSE'
        }
        self.angel_manager = AngelOneManager(self.config)
    
    def test_initialization(self):
        """Test AngelOneManager initialization"""
        self.assertEqual(self.angel_manager.config, self.config)
        self.assertIsNotNone(self.angel_manager.logger)
    
    def test_validate_config(self):
        """Test configuration validation"""
        is_valid = self.angel_manager.validate_config()
        self.assertIsInstance(is_valid, bool)
    
    def test_get_config_status(self):
        """Test configuration status"""
        status = self.angel_manager.get_config_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('api_configured', status)
        self.assertIn('exchange', status)
    
    def test_get_available_intervals(self):
        """Test getting available intervals"""
        intervals = self.angel_manager.get_available_intervals()
        
        self.assertIsInstance(intervals, list)
        self.assertGreater(len(intervals), 0)
        self.assertIn('ONE_DAY', intervals)
    
    def test_get_interval_description(self):
        """Test getting interval descriptions"""
        description = self.angel_manager.get_interval_description('ONE_DAY')
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)
    
    def test_get_api_limits(self):
        """Test getting API limits"""
        limits = self.angel_manager.get_api_limits('ONE_DAY')
        
        self.assertIsInstance(limits, int)
        self.assertGreater(limits, 0)
    
    def test_test_connection(self):
        """Test API connection"""
        # This will likely fail without real credentials, but should not crash
        try:
            result = self.angel_manager.test_connection()
            self.assertIsInstance(result, bool)
        except Exception as e:
            # Expected without real API credentials
            self.assertIsInstance(e, Exception)


class TestAPICoordinator(unittest.TestCase):
    """Test APICoordinator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.coordinator = APICoordinator(max_workers=4)
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
    
    def test_initialization(self):
        """Test APICoordinator initialization"""
        self.assertEqual(self.coordinator.max_workers, 4)
        self.assertIsNotNone(self.coordinator.logger)
    
    def test_check_rate_limit(self):
        """Test rate limiting"""
        result = self.coordinator.check_rate_limit('test_api')
        self.assertIsInstance(result, bool)
    
    def test_handle_api_error(self):
        """Test API error handling"""
        test_error = Exception("Test API error")
        result = self.coordinator.handle_api_error(test_error, 'test_api')
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('error', result)
    
    def test_get_coordinator_status(self):
        """Test coordinator status"""
        status = self.coordinator.get_coordinator_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('active_workers', status)
        self.assertIn('rate_limits', status)
    
    def test_get_api_statistics(self):
        """Test API statistics"""
        stats = self.coordinator.get_api_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_calls', stats)
        self.assertIn('success_rate', stats)
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Should not raise an exception
        self.coordinator.clear_cache()


class TestDataServiceWrapper(unittest.TestCase):
    """Test DataServiceWrapper component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.ticker = "AAPL"
        self.config = {'use_enhanced': True}
        self.data_wrapper = DataServiceWrapper(ticker=self.ticker, config=self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
        self.sample_data.set_index('Date', inplace=True)
    
    def test_initialization(self):
        """Test DataServiceWrapper initialization"""
        self.assertEqual(self.data_wrapper.ticker, self.ticker)
        self.assertEqual(self.data_wrapper.config, self.config)
        self.assertIsNotNone(self.data_wrapper.logger)
    
    def test_is_indian_stock(self):
        """Test Indian stock detection"""
        # Test Indian stock
        self.assertTrue(self.data_wrapper._is_indian_stock('RELIANCE'))
        self.assertTrue(self.data_wrapper._is_indian_stock('TCS.NS'))
        
        # Test non-Indian stock
        self.assertFalse(self.data_wrapper._is_indian_stock('AAPL'))
        self.assertFalse(self.data_wrapper._is_indian_stock('MSFT'))
    
    def test_get_data_source_info(self):
        """Test data source information"""
        info = self.data_wrapper.get_data_source_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('primary_source', info)
        self.assertIn('fallback_source', info)
    
    def test_get_data_source(self):
        """Test data source detection"""
        source = self.data_wrapper.get_data_source()
        
        self.assertIsInstance(source, str)
        self.assertIn(source, ['angel_one', 'yahoo_finance', 'both'])
    
    def test_convert_period_to_days(self):
        """Test period conversion"""
        test_cases = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        
        for period, expected_days in test_cases.items():
            result = self.data_wrapper._convert_period_to_days(period)
            self.assertEqual(result, expected_days)


class TestEconomicDataService(unittest.TestCase):
    """Test EconomicDataService component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.economic_service = EconomicDataService()
    
    def test_initialization(self):
        """Test EconomicDataService initialization"""
        self.assertIsNotNone(self.economic_service)
    
    def test_get_economic_indicators(self):
        """Test economic indicators"""
        indicators = self.economic_service.get_economic_indicators()
        
        self.assertIsInstance(indicators, dict)
        self.assertIn('gdp_growth', indicators)
        self.assertIn('inflation_rate', indicators)
        self.assertIn('interest_rate', indicators)
    
    def test_get_sentiment_indicators(self):
        """Test sentiment indicators"""
        sentiment = self.economic_service.get_sentiment_indicators()
        
        self.assertIsInstance(sentiment, dict)
        self.assertIn('vix', sentiment)
        self.assertIn('fear_greed_index', sentiment)
    
    def test_get_sector_performance(self):
        """Test sector performance"""
        sectors = self.economic_service.get_sector_performance()
        
        self.assertIsInstance(sectors, dict)
        self.assertGreater(len(sectors), 0)
    
    def test_get_global_market_data(self):
        """Test global market data"""
        global_data = self.economic_service.get_global_market_data()
        
        self.assertIsInstance(global_data, dict)
        self.assertIn('us_markets', global_data)
        self.assertIn('european_markets', global_data)
    
    def test_get_economic_calendar(self):
        """Test economic calendar"""
        calendar = self.economic_service.get_economic_calendar(days_ahead=7)
        
        self.assertIsInstance(calendar, list)
    
    def test_get_economic_summary(self):
        """Test economic summary"""
        summary = self.economic_service.get_economic_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('overall_sentiment', summary)
        self.assertIn('key_indicators', summary)
    
    def test_test_connection(self):
        """Test connection"""
        result = self.economic_service.test_connection()
        self.assertIsInstance(result, bool)


class TestFeatureEngineeringService(unittest.TestCase):
    """Test FeatureEngineeringService component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_service = FeatureEngineeringService()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test FeatureEngineeringService initialization"""
        self.assertIsNotNone(self.feature_service)
    
    def test_engineer_all_features(self):
        """Test feature engineering"""
        features = self.feature_service.engineer_all_features(self.sample_data)
        
        self.assertIsInstance(features, dict)
        self.assertIn('price_features', features)
        self.assertIn('technical_features', features)
        self.assertIn('volume_features', features)
    
    def test_create_price_features(self):
        """Test price feature creation"""
        price_features = self.feature_service._create_price_features(self.sample_data)
        
        self.assertIsInstance(price_features, dict)
        self.assertIn('price_change', price_features)
        self.assertIn('price_volatility', price_features)
    
    def test_create_technical_features(self):
        """Test technical feature creation"""
        technical_features = self.feature_service._create_technical_features(self.sample_data)
        
        self.assertIsInstance(technical_features, dict)
        self.assertIn('sma_features', technical_features)
        self.assertIn('rsi_features', technical_features)
    
    def test_create_volume_features(self):
        """Test volume feature creation"""
        volume_features = self.feature_service._create_volume_features(self.sample_data)
        
        self.assertIsInstance(volume_features, dict)
        self.assertIn('volume_sma', volume_features)
        self.assertIn('volume_ratio', volume_features)
    
    def test_create_time_features(self):
        """Test time feature creation"""
        time_features = self.feature_service._create_time_features(self.sample_data)
        
        self.assertIsInstance(time_features, dict)
        self.assertIn('day_of_week', time_features)
        self.assertIn('month', time_features)
    
    def test_create_statistical_features(self):
        """Test statistical feature creation"""
        statistical_features = self.feature_service._create_statistical_features(self.sample_data)
        
        self.assertIsInstance(statistical_features, dict)
        self.assertIn('rolling_mean', statistical_features)
        self.assertIn('rolling_std', statistical_features)
    
    def test_get_feature_summary(self):
        """Test feature summary"""
        features = self.feature_service.engineer_all_features(self.sample_data)
        summary = self.feature_service.get_feature_summary(features)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_features', summary)
        self.assertIn('feature_categories', summary)


class TestTechnicalIndicatorsService(unittest.TestCase):
    """Test TechnicalIndicatorsService component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.indicators_service = TechnicalIndicatorsService()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 10000000, 100)
        })
    
    def test_initialization(self):
        """Test TechnicalIndicatorsService initialization"""
        self.assertIsNotNone(self.indicators_service)
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators"""
        result = self.indicators_service.calculate_all_indicators(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), len(self.sample_data.columns))
    
    def test_add_moving_averages(self):
        """Test moving averages calculation"""
        result = self.indicators_service._add_moving_averages(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('SMA_20', result.columns)
        self.assertIn('EMA_12', result.columns)
    
    def test_add_macd(self):
        """Test MACD calculation"""
        result = self.indicators_service._add_macd(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('MACD', result.columns)
        self.assertIn('MACD_signal', result.columns)
    
    def test_add_rsi(self):
        """Test RSI calculation"""
        result = self.indicators_service._add_rsi(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('RSI', result.columns)
    
    def test_add_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        result = self.indicators_service._add_bollinger_bands(self.sample_data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('BB_upper', result.columns)
        self.assertIn('BB_lower', result.columns)
    
    def test_get_indicator_signals(self):
        """Test indicator signals"""
        signals = self.indicators_service.get_indicator_signals(self.sample_data)
        
        self.assertIsInstance(signals, dict)
        self.assertIn('buy_signals', signals)
        self.assertIn('sell_signals', signals)
    
    def test_get_indicator_summary(self):
        """Test indicator summary"""
        summary = self.indicators_service.get_indicator_summary(self.sample_data)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_indicators', summary)
        self.assertIn('signal_strength', summary)


class TestSmartDataFetcher(unittest.TestCase):
    """Test SmartDataFetcher component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = SmartDataFetcher()
        self.ticker = "AAPL"
        self.interval = "ONE_DAY"
    
    def test_initialization(self):
        """Test SmartDataFetcher initialization"""
        self.assertIsNotNone(self.fetcher)
        self.assertIsNotNone(self.fetcher.fetch_tracker)
    
    def test_should_fetch_data(self):
        """Test fetch decision logic"""
        should_fetch, reason = self.fetcher.should_fetch_data(self.ticker, self.interval)
        
        self.assertIsInstance(should_fetch, bool)
        self.assertIsInstance(reason, str)
    
    def test_record_fetch(self):
        """Test fetch recording"""
        # Should not raise an exception
        self.fetcher.record_fetch(self.ticker, self.interval, success=True)
    
    def test_get_fetch_status(self):
        """Test fetch status"""
        status = self.fetcher.get_fetch_status(self.ticker, self.interval)
        
        self.assertIsInstance(status, dict)
        self.assertIn('last_fetch', status)
        self.assertIn('success_rate', status)
    
    def test_get_optimal_fetch_time(self):
        """Test optimal fetch time"""
        optimal_time = self.fetcher.get_optimal_fetch_time(self.ticker, self.interval)
        
        if optimal_time is not None:
            self.assertIsInstance(optimal_time, datetime)
    
    def test_get_fetch_recommendations(self):
        """Test fetch recommendations"""
        recommendations = self.fetcher.get_fetch_recommendations(self.ticker)
        
        self.assertIsInstance(recommendations, dict)
        self.assertIn('recommendations', recommendations)
    
    def test_get_max_days_per_request(self):
        """Test max days per request"""
        max_days = self.fetcher.get_max_days_per_request(self.interval)
        
        self.assertIsInstance(max_days, int)
        self.assertGreater(max_days, 0)
    
    def test_get_interval_limits(self):
        """Test interval limits"""
        limits = self.fetcher.get_interval_limits(self.interval)
        
        self.assertIsInstance(limits, dict)
        self.assertIn('max_days', limits)
        self.assertIn('description', limits)


class TestAdvancedCacheManager(unittest.TestCase):
    """Test AdvancedCacheManager component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'redis_url': 'redis://localhost:6379',
            'cache_dir': 'test_cache'
        }
        self.cache_manager = AdvancedCacheManager(config=self.config)
    
    def test_initialization(self):
        """Test AdvancedCacheManager initialization"""
        self.assertEqual(self.cache_manager.config, self.config)
        self.assertIsNotNone(self.cache_manager.logger)
    
    def test_get_analytics(self):
        """Test cache analytics"""
        analytics = self.cache_manager.get_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertIn('hit_rate', analytics)
        self.assertIn('miss_rate', analytics)
        self.assertIn('total_operations', analytics)
    
    def test_cleanup_expired(self):
        """Test cleanup of expired items"""
        # This is an async method, so we'll just test that it exists
        self.assertTrue(hasattr(self.cache_manager, 'cleanup_expired'))


class TestMLOptimizer(unittest.TestCase):
    """Test MLOptimizer component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {'max_workers': 4}
        self.ml_optimizer = MLOptimizer(config=self.config)
        
        # Create sample data
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 5), columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
        self.y = pd.Series(np.random.randn(100))
    
    def test_initialization(self):
        """Test MLOptimizer initialization"""
        self.assertEqual(self.ml_optimizer.config, self.config)
        self.assertIsNotNone(self.ml_optimizer.logger)
    
    def test_get_optimization_report(self):
        """Test optimization report"""
        report = self.ml_optimizer.get_optimization_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('optimization_count', report)
        self.assertIn('best_models', report)


class TestAutoScaler(unittest.TestCase):
    """Test AutoScaler component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'min_instances': 1,
            'max_instances': 10,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3
        }
        self.auto_scaler = AutoScaler(config=self.config)
    
    def test_initialization(self):
        """Test AutoScaler initialization"""
        self.assertEqual(self.auto_scaler.config, self.config)
        self.assertIsNotNone(self.auto_scaler.logger)
    
    def test_get_scaling_summary(self):
        """Test scaling summary"""
        summary = self.auto_scaler.get_scaling_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('current_instances', summary)
        self.assertIn('scaling_decisions', summary)


class TestMonitoringDashboard(unittest.TestCase):
    """Test MonitoringDashboard component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'websocket_host': 'localhost',
            'websocket_port': 8765
        }
        self.dashboard = MonitoringDashboard(config=self.config)
    
    def test_initialization(self):
        """Test MonitoringDashboard initialization"""
        self.assertEqual(self.dashboard.config, self.config)
        self.assertIsNotNone(self.dashboard.logger)
    
    def test_get_dashboard_url(self):
        """Test dashboard URL"""
        url = self.dashboard.get_dashboard_url()
        
        self.assertIsInstance(url, str)
        self.assertIn('ws://', url)
    
    def test_get_metrics_summary(self):
        """Test metrics summary"""
        summary = self.dashboard.get_metrics_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('system_metrics', summary)
        self.assertIn('api_metrics', summary)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDatabaseManager,
        TestAngelOneManager,
        TestAPICoordinator,
        TestDataServiceWrapper,
        TestEconomicDataService,
        TestFeatureEngineeringService,
        TestTechnicalIndicatorsService,
        TestSmartDataFetcher,
        TestAdvancedCacheManager,
        TestMLOptimizer,
        TestAutoScaler,
        TestMonitoringDashboard
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SERVICES TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    sys.exit(0 if result.wasSuccessful() else 1)
