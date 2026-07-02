#!/usr/bin/env python3
"""
News Sentiment Analysis Interface
User interface for news sentiment analysis input and configuration
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsSentimentInterface:
    """
    User interface for news sentiment analysis
    
    Features:
    - Stock name input from user
    - News topic configuration
    - Analysis parameters
    - Results display
    """
    
    def __init__(self):
        """Initialize News Sentiment Interface"""
        logger.info("News Sentiment Interface initialized")
    
    def get_news_analysis_inputs(self, test_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get user inputs for news sentiment analysis
        
        Args:
            test_config: Optional test configuration for non-interactive testing
            
        Returns:
            Dictionary with news analysis inputs
        """
        try:
            print("\n📰 News Sentiment Analysis Interface")
            print("=" * 50)
            
            if test_config:
                print("✅ Using test configuration for non-interactive testing")
                return self._process_test_config(test_config)
            
            # Get stock name from user
            stock_name = self._get_stock_name_input()
            
            # Get news topic
            news_topic = self._get_news_topic_input(stock_name)
            
            # Get analysis parameters
            analysis_params = self._get_analysis_parameters()
            
            # Compile inputs
            inputs = {
                'stock_name': stock_name,
                'news_topic': news_topic,
                'analysis_params': analysis_params,
                'timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            print("\n✅ News analysis configuration completed!")
            return inputs
            
        except Exception as e:
            logger.error(f"Failed to get news analysis inputs: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_test_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process test configuration"""
        return {
            'stock_name': test_config.get('stock_name', 'TCS'),
            'news_topic': test_config.get('news_topic', 'TCS earnings India'),
            'analysis_params': test_config.get('analysis_params', {
                'max_articles': 5,
                'include_balance_sheet': True,
                'sentiment_weight': 0.7,
                'balance_weight': 0.3
            }),
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
    
    def _get_stock_name_input(self) -> str:
        """Get stock name from user"""
        try:
            print("\n📈 Stock Name Input:")
            print("-" * 25)
            print("Enter the stock name or company name for news analysis")
            print("Examples: TCS, Reliance, Infosys, HDFC Bank, ICICI Bank")
            
            while True:
                stock_name = input("Enter stock/company name: ").strip()
                
                if not stock_name:
                    print("❌ Please enter a valid stock/company name")
                    continue
                
                print(f"✅ Stock name: {stock_name}")
                return stock_name
                
        except Exception as e:
            logger.error(f"Failed to get stock name input: {e}")
            return "TCS"  # Default fallback
    
    def _get_news_topic_input(self, stock_name: str) -> str:
        """Get news topic from user"""
        try:
            print(f"\n🔍 News Topic Configuration:")
            print("-" * 35)
            print(f"Stock: {stock_name}")
            print("\nChoose news topic:")
            print("1. Earnings and financial results")
            print("2. Market performance and stock price")
            print("3. Business news and developments")
            print("4. Industry news and trends")
            print("5. Custom topic")
            
            while True:
                choice = input("Select topic (1-5, default: 1): ").strip()
                
                if not choice:
                    choice = '1'
                
                topic_map = {
                    '1': f"{stock_name} earnings India",
                    '2': f"{stock_name} stock price market performance",
                    '3': f"{stock_name} business developments news",
                    '4': f"{stock_name} industry trends sector news",
                    '5': None  # Custom input
                }
                
                if choice in topic_map:
                    if choice == '5':
                        # Custom topic
                        custom_topic = input("Enter custom news topic: ").strip()
                        if custom_topic:
                            print(f"✅ Custom topic: {custom_topic}")
                            return custom_topic
                        else:
                            print("❌ Please enter a custom topic")
                            continue
                    else:
                        topic = topic_map[choice]
                        print(f"✅ Selected topic: {topic}")
                        return topic
                else:
                    print("❌ Invalid choice. Please select 1-5.")
                    
        except Exception as e:
            logger.error(f"Failed to get news topic input: {e}")
            return f"{stock_name} earnings India"  # Default fallback
    
    def _get_analysis_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters from user"""
        try:
            print("\n⚙️ Analysis Parameters:")
            print("-" * 25)
            
            # Get number of articles
            max_articles = self._get_max_articles_input()
            
            # Get balance sheet analysis preference
            include_balance_sheet = self._get_balance_sheet_preference()
            
            # Get confidence weighting
            sentiment_weight, balance_weight = self._get_confidence_weighting()
            
            # Get analysis depth
            analysis_depth = self._get_analysis_depth()
            
            params = {
                'max_articles': max_articles,
                'include_balance_sheet': include_balance_sheet,
                'sentiment_weight': sentiment_weight,
                'balance_weight': balance_weight,
                'analysis_depth': analysis_depth
            }
            
            return params
            
        except Exception as e:
            logger.error(f"Failed to get analysis parameters: {e}")
            return {
                'max_articles': 5,
                'include_balance_sheet': True,
                'sentiment_weight': 0.7,
                'balance_weight': 0.3,
                'analysis_depth': 'standard'
            }
    
    def _get_max_articles_input(self) -> int:
        """Get maximum number of articles to analyze"""
        try:
            print("\n📊 Number of Articles to Analyze:")
            print("1. 3 articles (Quick analysis)")
            print("2. 5 articles (Standard analysis) - Recommended")
            print("3. 10 articles (Comprehensive analysis)")
            print("4. Custom number")
            
            while True:
                choice = input("Select number of articles (1-4, default: 2): ").strip()
                
                if not choice:
                    choice = '2'
                
                if choice == '1':
                    return 3
                elif choice == '2':
                    return 5
                elif choice == '3':
                    return 10
                elif choice == '4':
                    try:
                        custom = int(input("Enter custom number (1-20): "))
                        if 1 <= custom <= 20:
                            return custom
                        else:
                            print("❌ Please enter a number between 1 and 20")
                            continue
                    except ValueError:
                        print("❌ Please enter a valid number")
                        continue
                else:
                    print("❌ Invalid choice. Please select 1-4.")
                    
        except Exception as e:
            logger.error(f"Failed to get max articles input: {e}")
            return 5
    
    def _get_balance_sheet_preference(self) -> bool:
        """Get balance sheet analysis preference"""
        try:
            print("\n💰 Balance Sheet Analysis:")
            print("1. Yes - Include balance sheet analysis (recommended)")
            print("2. No - Sentiment analysis only")
            
            while True:
                choice = input("Include balance sheet analysis? (1/2, default: 1): ").strip()
                
                if not choice:
                    choice = '1'
                
                if choice == '1':
                    print("✅ Balance sheet analysis enabled")
                    return True
                elif choice == '2':
                    print("✅ Sentiment analysis only")
                    return False
                else:
                    print("❌ Invalid choice. Please select 1 or 2.")
                    
        except Exception as e:
            logger.error(f"Failed to get balance sheet preference: {e}")
            return True
    
    def _get_confidence_weighting(self) -> tuple:
        """Get confidence weighting preferences"""
        try:
            print("\n⚖️ Confidence Weighting:")
            print("1. Standard (70% sentiment, 30% balance sheet)")
            print("2. Sentiment-focused (80% sentiment, 20% balance sheet)")
            print("3. Balanced (60% sentiment, 40% balance sheet)")
            print("4. Custom weighting")
            
            while True:
                choice = input("Select weighting (1-4, default: 1): ").strip()
                
                if not choice:
                    choice = '1'
                
                if choice == '1':
                    return 0.7, 0.3
                elif choice == '2':
                    return 0.8, 0.2
                elif choice == '3':
                    return 0.6, 0.4
                elif choice == '4':
                    try:
                        sentiment_weight = float(input("Enter sentiment weight (0.0-1.0): "))
                        if 0.0 <= sentiment_weight <= 1.0:
                            balance_weight = 1.0 - sentiment_weight
                            return sentiment_weight, balance_weight
                        else:
                            print("❌ Please enter a weight between 0.0 and 1.0")
                            continue
                    except ValueError:
                        print("❌ Please enter a valid number")
                        continue
                else:
                    print("❌ Invalid choice. Please select 1-4.")
                    
        except Exception as e:
            logger.error(f"Failed to get confidence weighting: {e}")
            return 0.7, 0.3
    
    def _get_analysis_depth(self) -> str:
        """Get analysis depth preference"""
        try:
            print("\n🔍 Analysis Depth:")
            print("1. Quick (Basic sentiment analysis)")
            print("2. Standard (Sentiment + NER + Balance sheet) - Recommended")
            print("3. Deep (Full analysis with trend analysis)")
            
            while True:
                choice = input("Select analysis depth (1-3, default: 2): ").strip()
                
                if not choice:
                    choice = '2'
                
                depth_map = {
                    '1': 'quick',
                    '2': 'standard',
                    '3': 'deep'
                }
                
                if choice in depth_map:
                    depth = depth_map[choice]
                    print(f"✅ Analysis depth: {depth}")
                    return depth
                else:
                    print("❌ Invalid choice. Please select 1-3.")
                    
        except Exception as e:
            logger.error(f"Failed to get analysis depth: {e}")
            return 'standard'
    
    def display_news_analysis_results(self, results: List[Dict[str, Any]], topic: str):
        """
        Display news analysis results
        
        Args:
            results: List of analysis results
            topic: Analysis topic
        """
        try:
            print(f"\n📊 News Sentiment Analysis Results for: {topic}")
            print("=" * 60)
            
            if not results:
                print("❌ No analysis results available")
                return
            
            # Display summary
            total_articles = len(results)
            positive_count = sum(1 for r in results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in results if r['sentiment'] == 'neutral')
            
            avg_confidence = sum(r['final_confidence'] for r in results) / total_articles
            high_impact_count = sum(1 for r in results if r['impact'] == 'High Impact')
            
            print(f"📈 Analysis Summary:")
            print(f"   Total Articles: {total_articles}")
            print(f"   Positive: {positive_count} ({positive_count/total_articles*100:.1f}%)")
            print(f"   Negative: {negative_count} ({negative_count/total_articles*100:.1f}%)")
            print(f"   Neutral: {neutral_count} ({neutral_count/total_articles*100:.1f}%)")
            print(f"   Average Confidence: {avg_confidence:.3f}")
            print(f"   High Impact Articles: {high_impact_count}")
            
            print(f"\n📰 Detailed Results:")
            print("-" * 60)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. 📰 HEADLINE: {result['title']}")
                print(f"   🔗 URL: {result['url']}")
                print(f"   📅 Published: {result.get('published_date', 'Unknown date')}")
                print(f"   📈 Sentiment: {result['sentiment'].upper()} (Confidence: {result['confidence']:.3f})")
                print(f"   💰 Balance Confidence: {result['balance_confidence']:.3f}")
                print(f"   ✅ Final Confidence: {result['final_confidence']:.3f}")
                print(f"   ⚡ Impact: {result['impact']}")
                print(f"   📊 Relevance Score: {result.get('relevance_score', 0):.3f}")
                
                # Display article text snippet if available
                text = result.get('text', '')
                if text and len(text) > 50:
                    text_snippet = text[:300] + "..." if len(text) > 300 else text
                    print(f"   📝 Text Preview: {text_snippet}")
                
                if result['entities']:
                    print(f"   🏢 Entities Found:")
                    for comp, (ticker, sector) in result['entities'].items():
                        print(f"      - {comp} → {ticker} ({sector})")
                        print(f"      👉 Impact: {result['sentiment'].capitalize()} for {sector} sector")
                else:
                    print(f"   🏢 Entities: General market impact")
                
                print("-" * 60)
            
            # Display overall sentiment
            overall_sentiment = (positive_count - negative_count) / total_articles
            if overall_sentiment > 0.1:
                sentiment_label = "BULLISH"
                sentiment_emoji = "📈"
            elif overall_sentiment < -0.1:
                sentiment_label = "BEARISH"
                sentiment_emoji = "📉"
            else:
                sentiment_label = "NEUTRAL"
                sentiment_emoji = "➡️"
            
            print(f"\n🎯 Overall Market Sentiment: {sentiment_emoji} {sentiment_label}")
            print(f"   Sentiment Score: {overall_sentiment:.3f}")
            print(f"   Confidence Level: {avg_confidence:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to display news analysis results: {e}")
            print(f"❌ Error displaying results: {e}")
    
    def get_interface_status(self) -> Dict[str, Any]:
        """
        Get interface status
        
        Returns:
            Status information
        """
        try:
            return {
                'initialized': True,
                'interface_type': 'news_sentiment',
                'features': [
                    'stock_name_input',
                    'news_topic_configuration',
                    'analysis_parameters',
                    'results_display'
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get interface status: {e}")
            return {'error': str(e)}
