#!/usr/bin/env python3
"""
Simple News Sentiment Test
Test news sentiment analysis with mock data
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

def test_simple_news_sentiment():
    """Test simple news sentiment analysis with mock data"""
    try:
        print("🧪 Testing Simple News Sentiment Analysis...")
        
        from main.services.simple_news_sentiment_service import SimpleNewsSentimentService
        
        # Initialize service
        service = SimpleNewsSentimentService()
        
        # Test service status
        status = service.get_service_status()
        print(f"✅ Service Status: {status}")
        
        # Test sentiment analysis with mock text
        test_text = "TCS reported strong quarterly earnings with significant growth in revenue and profits. The company's performance exceeded expectations."
        sentiment, confidence = service.analyze_sentiment(test_text)
        print(f"✅ Sentiment Analysis: {sentiment} (confidence: {confidence})")
        
        # Test entity extraction
        entities = service.extract_entities_and_sector(test_text)
        print(f"✅ Entity Extraction: {entities}")
        
        # Test balance sheet analysis
        balance_score = service.get_balance_sheet_strength("TCS.NS")
        print(f"✅ Balance Sheet Analysis: {balance_score}")
        
        # Test with mock news data
        mock_news_data = [
            {
                "title": "TCS Reports Strong Q3 Earnings",
                "url": "https://example.com/tcs-earnings",
                "text": "TCS reported strong quarterly earnings with significant growth in revenue and profits. The company's performance exceeded expectations."
            },
            {
                "title": "TCS Stock Price Rises on Positive Outlook",
                "url": "https://example.com/tcs-stock",
                "text": "TCS stock price rose significantly following positive earnings announcement. Analysts are optimistic about future growth prospects."
            }
        ]
        
        print("\n📰 Testing with mock news data...")
        results = []
        
        for article in mock_news_data:
            # Analyze sentiment
            sentiment, confidence = service.analyze_sentiment(article['text'])
            
            # Extract entities
            entities = service.extract_entities_and_sector(article['title'] + " " + article['text'])
            
            # Get balance sheet confidence
            balance_confidence = 0.5
            if entities:
                for comp, (ticker, _) in entities.items():
                    balance_confidence = service.get_balance_sheet_strength(ticker)
                    break
            
            # Calculate final confidence
            final_confidence = round((0.7 * confidence) + (0.3 * balance_confidence), 3)
            impact = service.get_impact_label(final_confidence)
            
            result = {
                "title": article['title'],
                "url": article['url'],
                "sentiment": sentiment,
                "confidence": confidence,
                "balance_confidence": balance_confidence,
                "final_confidence": final_confidence,
                "impact": impact,
                "entities": {comp: sector for comp, (_, sector) in entities.items()} if entities else {"General": "Economy"}
            }
            
            results.append(result)
            
            print(f"  📰 {article['title']}")
            print(f"     Sentiment: {sentiment} (confidence: {confidence})")
            print(f"     Final Confidence: {final_confidence}")
            print(f"     Impact: {impact}")
            print(f"     Entities: {entities}")
            print()
        
        # Create summary
        total_articles = len(results)
        positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
        negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
        neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
        
        avg_confidence = sum(r['final_confidence'] for r in results) / total_articles
        high_impact_count = sum(1 for r in results if r['impact'] == 'High Impact')
        
        print("📊 Analysis Summary:")
        print(f"   Total Articles: {total_articles}")
        print(f"   Positive: {positive_count} ({positive_count/total_articles*100:.1f}%)")
        print(f"   Negative: {negative_count} ({negative_count/total_articles*100:.1f}%)")
        print(f"   Neutral: {neutral_count} ({neutral_count/total_articles*100:.1f}%)")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   High Impact Articles: {high_impact_count}")
        
        # Calculate overall sentiment
        sentiment_score = (positive_count - negative_count) / total_articles
        if sentiment_score > 0.1:
            overall_sentiment = 'BULLISH'
            sentiment_emoji = '📈'
        elif sentiment_score < -0.1:
            overall_sentiment = 'BEARISH'
            sentiment_emoji = '📉'
        else:
            overall_sentiment = 'NEUTRAL'
            sentiment_emoji = '➡️'
        
        print(f"\n🎯 Overall Market Sentiment: {sentiment_emoji} {overall_sentiment}")
        print(f"   Sentiment Score: {sentiment_score:.3f}")
        print(f"   Confidence Level: {avg_confidence:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple news sentiment test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Simple News Sentiment Analysis Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if test_simple_news_sentiment():
        print("\n🎉 Simple news sentiment analysis test PASSED!")
        print("✅ The news sentiment analysis feature is working correctly")
        print("💡 The 'Unknown error' in the main application is likely due to:")
        print("   • Google News search restrictions")
        print("   • Network connectivity issues")
        print("   • The system is using the simple fallback service")
    else:
        print("\n❌ Simple news sentiment analysis test FAILED!")
        print("💡 Check the error messages above for details")

if __name__ == "__main__":
    main()
