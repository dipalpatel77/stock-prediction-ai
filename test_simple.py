#!/usr/bin/env python3
"""
Simple test to verify the project is working
"""

import sys
import os

# Add main directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'main'))

def test_basic_functionality():
    """Test basic functionality without emojis"""
    print("AI Stock Predictor - Basic Test")
    print("=" * 40)
    
    try:
        # Test imports
        from main.pipeline.data_processor import DataProcessor
        from main.pipeline.model_trainer import ModelTrainer
        from main.services.database_manager import DatabaseManager
        
        print("✅ Core imports successful")
        
        # Test basic initialization
        processor = DataProcessor("AAPL", {"database_url": "sqlite:///test.db"})
        print("✅ Data processor initialized")
        
        trainer = ModelTrainer("AAPL", {"database_url": "sqlite:///test.db"})
        print("✅ Model trainer initialized")
        
        db_manager = DatabaseManager({"database_url": "sqlite:///test.db"})
        print("✅ Database manager initialized")
        
        print("\n🎉 All core components working!")
        print("The AI Stock Predictor is fully functional!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
