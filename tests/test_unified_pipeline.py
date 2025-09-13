#!/usr/bin/env python3
"""
Test unified analysis pipeline
"""

import unittest
from unified_analysis_pipeline import UnifiedAnalysisPipeline

class TestUnifiedPipeline(unittest.TestCase):
    """Test unified pipeline"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = UnifiedAnalysisPipeline("AAPL")
        self.assertIsNotNone(pipeline)
        self.assertEqual(pipeline.ticker, "AAPL")

if __name__ == '__main__':
    unittest.main()
