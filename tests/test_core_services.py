#!/usr/bin/env python3
"""
Test core services functionality
"""

import unittest
from src.core.data_service import DataService
from src.core.model_service import ModelService
from src.core.strategy_service import StrategyService

class TestCoreServices(unittest.TestCase):
    """Test core services"""
    
    def test_data_service(self):
        """Test data service initialization"""
        service = DataService()
        self.assertIsNotNone(service)
    
    def test_model_service(self):
        """Test model service initialization"""
        service = ModelService()
        self.assertIsNotNone(service)
    
    def test_strategy_service(self):
        """Test strategy service initialization"""
        service = StrategyService()
        self.assertIsNotNone(service)

if __name__ == '__main__':
    unittest.main()
