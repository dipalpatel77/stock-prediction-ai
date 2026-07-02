#!/usr/bin/env python3
"""
News Sentiment Analysis Test Script
Test the complete news sentiment analysis workflow
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_news_sentiment_service():
    """Test news sentiment service"""
    try:
        print("🧪 Testing News Sentiment Service...")
        
        from main.services.news_sentiment_service import NewsSentimentService
        
        # Initialize service
        service = NewsSentimentService()
        
        # Test service status
        status = service.get_service_status()
        print(f"✅ Service Status: {status}")
        
        # Test sentiment analysis
        test_text = "TCS reported strong quarterly earnings with significant growth in revenue and profits."
        sentiment, confidence = service.analyze_sentiment(test_text)
        print(f"✅ Sentiment Analysis: {sentiment} (confidence: {confidence})")
        
        # Test entity extraction
        entities = service.extract_entities_and_sector(test_text)
        print(f"✅ Entity Extraction: {entities}")
        
        # Test balance sheet analysis
        balance_score = service.get_balance_sheet_strength("TCS.NS")
        print(f"✅ Balance Sheet Analysis: {balance_score}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ News sentiment service test failed: {e}")
        return False

def test_news_sentiment_database():
    """Test news sentiment database manager"""
    try:
        print("\n🧪 Testing News Sentiment Database Manager...")
        
        from main.services.news_sentiment_database_manager import NewsSentimentDatabaseManager
        
        # Initialize database manager
        db_manager = NewsSentimentDatabaseManager("test_news_sentiment.db")
        
        # Test database stats
        stats = db_manager.get_database_stats()
        print(f"✅ Database Stats: {stats}")
        
        # Test cleanup
        db_manager.cleanup_old_data(days_to_keep=0)
        print("✅ Database cleanup completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ News sentiment database test failed: {e}")
        return False

def test_news_sentiment_interface():
    """Test news sentiment interface"""
    try:
        print("\n🧪 Testing News Sentiment Interface...")
        
        from main.interfaces.news_sentiment_interface import NewsSentimentInterface
        
        # Initialize interface
        interface = NewsSentimentInterface()
        
        # Test with test configuration
        test_config = {
            'stock_name': 'TCS',
            'news_topic': 'TCS earnings India',
            'analysis_params': {
                'max_articles': 3,
                'include_balance_sheet': True,
                'sentiment_weight': 0.7,
                'balance_weight': 0.3
            }
        }
        
        inputs = interface.get_news_analysis_inputs(test_config)
        print(f"✅ Interface Test: {inputs['success']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ News sentiment interface test failed: {e}")
        return False

def test_news_sentiment_analyzer():
    """Test news sentiment analyzer pipeline component"""
    try:
        print("\n🧪 Testing News Sentiment Analyzer...")
        
        from main.pipeline.news_sentiment_analyzer import NewsSentimentAnalyzer
        
        # Initialize analyzer
        analyzer = NewsSentimentAnalyzer()
        
        # Test component status
        status = analyzer.get_component_status()
        print(f"✅ Component Status: {status}")
        
        # Test analysis (with limited articles for testing)
        test_params = {
            'max_articles': 2,
            'include_balance_sheet': True,
            'sentiment_weight': 0.7,
            'balance_weight': 0.3
        }
        
        print("🔍 Running test analysis (this may take a moment)...")
        results = analyzer.analyze_stock_news("TCS", "TCS earnings India", test_params)
        
        if results.get('success'):
            print(f"✅ Analysis Results: {results['articles_analyzed']} articles analyzed")
            print(f"   Execution Time: {results['execution_time']:.2f} seconds")
        else:
            print(f"⚠️ Analysis failed: {results.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ News sentiment analyzer test failed: {e}")
        return False

def test_full_integration():
    """Test full integration with main pipeline"""
    try:
        print("\n🧪 Testing Full Integration...")
        
        from main.pipeline.core_pipeline import UnifiedAnalysisPipeline
        
        # Initialize pipeline
        config = {
            'database': {
                'news_sentiment_db': 'test_news_sentiment.db'
            }
        }
        
        pipeline = UnifiedAnalysisPipeline("TCS", config)
        
        # Test news sentiment analysis
        test_params = {
            'max_articles': 2,
            'include_balance_sheet': True,
            'sentiment_weight': 0.7,
            'balance_weight': 0.3
        }
        
        print("🔍 Running full integration test...")
        results = pipeline.run_news_sentiment_analysis("TCS", "TCS earnings India", test_params)
        
        if results.get('success'):
            print(f"✅ Full Integration Test: {results['articles_analyzed']} articles analyzed")
            print(f"   Execution Time: {results['execution_time']:.2f} seconds")
        else:
            print(f"⚠️ Full integration test failed: {results.get('error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Full integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 News Sentiment Analysis Test Suite")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("News Sentiment Service", test_news_sentiment_service),
        ("News Sentiment Database", test_news_sentiment_database),
        ("News Sentiment Interface", test_news_sentiment_interface),
        ("News Sentiment Analyzer", test_news_sentiment_analyzer),
        ("Full Integration", test_full_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
        
        print()
    
    # Summary
    print("=" * 50)
    print("🎉 Test Summary")
    print("=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! News sentiment analysis is ready to use.")
        print("   Run: python main.py")
        print("   Select 'Yes' for news sentiment analysis when prompted")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Check the logs above for details.")
        print("   You may need to install missing dependencies:")
        print("   python install_news_sentiment.py")

if __name__ == "__main__":
    main()
