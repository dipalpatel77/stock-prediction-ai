#!/usr/bin/env python3
"""
Unit tests for data loader functionality
"""

import unittest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from partA_preprocessing.data_loader import DataLoader
from angel_one_data_downloader import AngelOneDataDownloader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_loader = DataLoader()
        self.test_ticker = "RELIANCE"
        
    def test_data_loader_initialization(self):
        """Test DataLoader initialization"""
        self.assertIsNotNone(self.data_loader)
        
    def test_load_csv_data(self):
        """Test loading data from CSV files"""
        # Test with existing data file
        test_file = "data/RELIANCE_partA_partC_enhanced.csv"
        if os.path.exists(test_file):
            df = self.data_loader.load_csv_data(test_file)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0)
            
    def test_load_csv_data_nonexistent(self):
        """Test loading data from non-existent file"""
        df = self.data_loader.load_csv_data("nonexistent_file.csv")
        self.assertIsNone(df)


class TestAngelOneDataDownloader(unittest.TestCase):
    """Test cases for AngelOneDataDownloader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.downloader = AngelOneDataDownloader()
        
    def test_downloader_initialization(self):
        """Test AngelOneDataDownloader initialization"""
        self.assertIsNotNone(self.downloader)
        
    @patch('angel_one_data_downloader.requests.get')
    def test_get_latest_data_mock(self, mock_get):
        """Test getting latest data with mocked response"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"timestamp": "2024-01-01", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1000}
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the method
        result = self.downloader.get_latest_data("RELIANCE", 1, "ONE_DAY")
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
