#!/usr/bin/env python3
"""
Unit tests for strategy components
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from partC_strategy.optimized_trading_strategy import OptimizedTradingStrategy
from partC_strategy.optimized_technical_indicators import OptimizedTechnicalIndicators
from partC_strategy.backtest import BacktestStrategy


class TestOptimizedTradingStrategy(unittest.TestCase):
    """Test cases for OptimizedTradingStrategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = OptimizedTradingStrategy()
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.rand(100) * 100 + 1000,
            'High': np.random.rand(100) * 100 + 1050,
            'Low': np.random.rand(100) * 100 + 950,
            'Close': np.random.rand(100) * 100 + 1000,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
    def test_strategy_initialization(self):
        """Test OptimizedTradingStrategy initialization"""
        self.assertIsNotNone(self.strategy)
        
    def test_generate_signals(self):
        """Test signal generation"""
        signals = self.strategy.generate_signals(self.test_data)
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertIn('Signal', signals.columns)
        
    def test_backtest_strategy(self):
        """Test strategy backtesting"""
        results = self.strategy.backtest_strategy(self.test_data, initial_capital=10000)
        self.assertIsInstance(results, dict)
        self.assertIn('equity_curve', results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)


class TestOptimizedTechnicalIndicators(unittest.TestCase):
    """Test cases for OptimizedTechnicalIndicators class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.indicators = OptimizedTechnicalIndicators()
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'Date': dates,
            'Open': np.random.rand(100) * 100 + 1000,
            'High': np.random.rand(100) * 100 + 1050,
            'Low': np.random.rand(100) * 100 + 950,
            'Close': np.random.rand(100) * 100 + 1000,
            'Volume': np.random.randint(1000, 10000, 100)
        })
        
    def test_indicators_initialization(self):
        """Test OptimizedTechnicalIndicators initialization"""
        self.assertIsNotNone(self.indicators)
        
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        sma = self.indicators.calculate_sma(self.test_data['Close'], window=20)
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.test_data))
        
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = self.indicators.calculate_rsi(self.test_data['Close'], window=14)
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.test_data))
        
    def test_calculate_macd(self):
        """Test MACD calculation"""
        macd, signal, histogram = self.indicators.calculate_macd(self.test_data['Close'])
        self.assertIsInstance(macd, pd.Series)
        self.assertIsInstance(signal, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = self.indicators.calculate_bollinger_bands(self.test_data['Close'])
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)


class TestBacktestStrategy(unittest.TestCase):
    """Test cases for BacktestStrategy class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backtest = BacktestStrategy()
        # Create sample signals data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.test_signals = pd.DataFrame({
            'Date': dates,
            'Signal': np.random.choice([-1, 0, 1], 100),
            'Price': np.random.rand(100) * 100 + 1000
        })
        
    def test_backtest_initialization(self):
        """Test BacktestStrategy initialization"""
        self.assertIsNotNone(self.backtest)
        
    def test_run_backtest(self):
        """Test backtest execution"""
        results = self.backtest.run_backtest(self.test_signals)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
