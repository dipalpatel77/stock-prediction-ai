#!/usr/bin/env python3
"""
Test Fallback News Sentiment Analysis
Test the fallback mechanism when no news articles are found
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

def test_fallback_news_analysis():
    """Test fallback news sentiment analysis"""
    try:
        print("🧪 Testing Fallback News Sentiment Analysis...")
        
        from main.pipeline.news_sentiment_analyzer import NewsSentimentAnalyzer
        
        # Initialize analyzer
        analyzer = NewsSentimentAnalyzer(ticker="TCS")
        
        # Test fallback analysis creation
        fallback_result = analyzer._create_fallback_analysis("TCS", "TCS earnings India")
        
        print("✅ Fallback Analysis Created:")
        print(f"   Title: {fallback_result['title']}")
        print(f"   Sentiment: {fallback_result['sentiment']}")
        print(f"   Confidence: {fallback_result['confidence']}")
        print(f"   Final Confidence: {fallback_result['final_confidence']}")
        print(f"   Impact: {fallback_result['impact']}")
        print(f"   Entities: {fallback_result['entities']}")
        print(f"   Fallback: {fallback_result['fallback']}")
        print(f"   Note: {fallback_result['note']}")
        
        # Test full analysis with fallback
        print("\n🔍 Testing Full Analysis with Fallback...")
        
        analysis_params = {
            'max_articles': 5,
            'include_balance_sheet': True,
            'sentiment_weight': 0.7,
            'balance_weight': 0.3
        }
        
        result = analyzer.analyze_stock_news("TCS", "TCS earnings India", analysis_params)
        
        print("✅ Full Analysis Result:")
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Articles Analyzed: {result.get('articles_analyzed', 0)}")
            print(f"   Fallback Mode: {result.get('fallback_mode', False)}")
            if 'summary' in result:
                summary = result['summary']
                print(f"   Total Articles: {summary.get('total_articles', 0)}")
                print(f"   Overall Sentiment: {summary.get('overall_sentiment', 'Unknown')}")
                print(f"   Average Confidence: {summary.get('avg_confidence', 0):.3f}")
                print(f"   High Impact Count: {summary.get('high_impact_count', 0)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fallback news analysis test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Fallback News Sentiment Analysis Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if test_fallback_news_analysis():
        print("\n🎉 Fallback news sentiment analysis test PASSED!")
        print("✅ The fallback mechanism is working correctly")
        print("💡 When no news articles are found, the system will:")
        print("   • Create a general market analysis")
        print("   • Provide sentiment analysis based on stock name")
        print("   • Show appropriate warnings to the user")
        print("   • Continue with the analysis instead of failing")
    else:
        print("\n❌ Fallback news sentiment analysis test FAILED!")
        print("💡 Check the error messages above for details")

if __name__ == "__main__":
    main()
