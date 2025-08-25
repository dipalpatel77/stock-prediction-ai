#!/usr/bin/env python3
"""
Optimized Sentiment Analysis Module
Consolidates all sentiment analysis functionality from duplicate files
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import re
import time
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class OptimizedSentimentAnalyzer:
    """
    Optimized sentiment analyzer that consolidates functionality from:
    - enhanced_sentiment_analyzer.py
    - news_sentiment.py
    - public_sentiment.py
    - indian_sentiment_analyzer.py
    - free_sentiment_sources.py
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the optimized sentiment analyzer.
        
        Args:
            api_keys: Dictionary containing API keys for various services
        """
        self.api_keys = api_keys or {}
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # News sources configuration
        self.news_sources = {
            'newsapi': {
                'base_url': 'https://newsapi.org/v2/everything',
                'requires_key': True
            },
            'alphavantage': {
                'base_url': 'https://www.alphavantage.co/query',
                'requires_key': True
            },
            'yahoo_finance': {
                'base_url': 'https://finance.yahoo.com/news',
                'requires_key': False
            }
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'very_positive': 0.6,
            'positive': 0.2,
            'neutral': -0.2,
            'negative': -0.6
        }
        
        # Sentiment weights for different sources
        self.sentiment_weights = {
            'news': 0.35,
            'social_media': 0.25,
            'analyst_ratings': 0.20,
            'public_sentiment': 0.20
        }
    
    def analyze_stock_sentiment(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """
        Comprehensive sentiment analysis for a stock.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with sentiment analysis results
        """
        print(f"🔍 Analyzing sentiment for {ticker}...")
        
        try:
            # Get company info
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName', ticker)
            
            # Collect sentiment from multiple sources
            sentiment_data = {}
            
            # News sentiment
            news_sentiment = self.get_news_sentiment(ticker, company_name, days_back)
            if not news_sentiment.empty:
                sentiment_data['news_sentiment'] = news_sentiment
            
            # Social media sentiment
            social_sentiment = self.get_social_media_sentiment(ticker)
            if social_sentiment:
                sentiment_data['social_sentiment'] = social_sentiment
            
            # Analyst ratings
            analyst_sentiment = self.get_analyst_ratings(ticker)
            if analyst_sentiment:
                sentiment_data['analyst_sentiment'] = analyst_sentiment
            
            # Public sentiment
            public_sentiment = self.get_public_sentiment(ticker)
            if public_sentiment:
                sentiment_data['public_sentiment'] = public_sentiment
            
            # Combine all sentiment data
            combined_sentiment = self._combine_sentiment_data(sentiment_data, days_back)
            
            print(f"✅ Sentiment analysis completed for {ticker}")
            return combined_sentiment
            
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, ticker: str, company_name: str, days_back: int = 7) -> pd.DataFrame:
        """
        Get news sentiment from multiple sources.
        
        Args:
            ticker: Stock ticker
            company_name: Company name
            days_back: Days to look back
            
        Returns:
            DataFrame with news sentiment
        """
        try:
            all_news = []
            
            # Try NewsAPI
            if 'newsapi' in self.api_keys:
                newsapi_news = self._fetch_news_from_newsapi(company_name, ticker, days_back)
                all_news.extend(newsapi_news)
            
            # Try Alpha Vantage
            if 'alphavantage' in self.api_keys:
                av_news = self._fetch_news_from_alphavantage(ticker, days_back)
                all_news.extend(av_news)
            
            # If no API keys, create synthetic data
            if not all_news:
                all_news = self._create_synthetic_news_sentiment(ticker, days_back)
            
            # Analyze sentiment for each news article
            sentiment_results = []
            for article in all_news:
                sentiment_score = self._analyze_text_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                sentiment_results.append({
                    'date': article.get('date', datetime.now()),
                    'title': article.get('title', ''),
                    'sentiment_score': sentiment_score,
                    'source': article.get('source', 'unknown'),
                    'url': article.get('url', '')
                })
            
            return pd.DataFrame(sentiment_results)
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return pd.DataFrame()
    
    def get_social_media_sentiment(self, ticker: str) -> Dict:
        """
        Get social media sentiment.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with social media sentiment
        """
        try:
            # In a real implementation, this would use Twitter API, Reddit API, etc.
            # For now, we'll create synthetic data
            
            social_sentiment = {
                'twitter_sentiment': np.random.normal(0.05, 0.2),
                'reddit_sentiment': np.random.normal(0.02, 0.15),
                'stocktwits_sentiment': np.random.normal(0.03, 0.18),
                'social_volume': np.random.randint(100, 1000),
                'engagement_rate': np.random.uniform(0.01, 0.05),
                'trending_score': np.random.uniform(0.1, 0.9)
            }
            
            return social_sentiment
            
        except Exception as e:
            print(f"Error fetching social media sentiment: {e}")
            return {}
    
    def get_analyst_ratings(self, ticker: str) -> Dict:
        """
        Get analyst ratings and price targets.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with analyst ratings
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get analyst recommendations
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                # Calculate average rating
                rating_mapping = {
                    'Buy': 1.0,
                    'Strong Buy': 1.0,
                    'Outperform': 0.75,
                    'Hold': 0.5,
                    'Underperform': 0.25,
                    'Sell': 0.0,
                    'Strong Sell': 0.0
                }
                
                recommendations['rating_score'] = recommendations['To Grade'].map(rating_mapping)
                avg_rating = recommendations['rating_score'].mean()
                
                # Get price targets
                price_targets = stock.recommendations_summary
                
                return {
                    'average_rating': avg_rating,
                    'rating_count': len(recommendations),
                    'price_targets': price_targets,
                    'recent_recommendations': recommendations.head(10).to_dict('records'),
                    'rating_distribution': recommendations['To Grade'].value_counts().to_dict()
                }
            
            return {}
            
        except Exception as e:
            print(f"Error fetching analyst ratings: {e}")
            return {}
    
    def get_public_sentiment(self, ticker: str) -> Dict:
        """
        Get public sentiment from various sources.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Dictionary with public sentiment
        """
        try:
            # In a real implementation, this would fetch from various public sentiment sources
            # For now, we'll create synthetic data
            
            public_sentiment = {
                'overall_sentiment': np.random.normal(0.1, 0.3),
                'confidence_score': np.random.uniform(0.6, 0.9),
                'sentiment_volume': np.random.randint(500, 2000),
                'sentiment_trend': np.random.choice(['increasing', 'decreasing', 'stable']),
                'market_mood': np.random.choice(['bullish', 'bearish', 'neutral']),
                'fear_greed_index': np.random.randint(20, 80)
            }
            
            return public_sentiment
            
        except Exception as e:
            print(f"Error fetching public sentiment: {e}")
            return {}
    
    def _fetch_news_from_newsapi(self, company_name: str, ticker: str, days_back: int) -> List[Dict]:
        """Fetch news from NewsAPI."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            query = f'"{company_name}" OR "{ticker}"'
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.api_keys['newsapi']
            }
            
            response = requests.get(self.news_sources['newsapi']['base_url'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                return [
                    {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'date': datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                        'source': article.get('source', {}).get('name', 'unknown'),
                        'url': article.get('url', '')
                    }
                    for article in articles
                ]
            
            return []
            
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []
    
    def _fetch_news_from_alphavantage(self, ticker: str, days_back: int) -> List[Dict]:
        """Fetch news from Alpha Vantage."""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'apikey': self.api_keys['alphavantage']
            }
            
            response = requests.get(self.news_sources['alphavantage']['base_url'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' in data:
                articles = data['feed']
                return [
                    {
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),
                        'date': datetime.fromtimestamp(int(article.get('time_published', 0))),
                        'source': article.get('source', 'unknown'),
                        'url': article.get('url', ''),
                        'sentiment_score': float(article.get('overall_sentiment_score', 0))
                    }
                    for article in articles
                ]
            
            return []
            
        except Exception as e:
            print(f"Error fetching from Alpha Vantage: {e}")
            return []
    
    def _create_synthetic_news_sentiment(self, ticker: str, days_back: int) -> List[Dict]:
        """Create synthetic news sentiment data."""
        dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
        
        synthetic_news = []
        for date in dates:
            # Simulate varying sentiment scores
            base_sentiment = np.random.normal(0.1, 0.3)
            synthetic_news.append({
                'title': f'News about {ticker} on {date.strftime("%Y-%m-%d")}',
                'description': f'Synthetic news description for {ticker}',
                'date': date,
                'source': 'synthetic',
                'url': '',
                'sentiment_score': max(-1, min(1, base_sentiment))
            })
        
        return synthetic_news
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using multiple methods.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 and 1
        """
        try:
            # Clean text
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            vader_score = vader_scores['compound']
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # Combine scores (weighted average)
            combined_score = (vader_score * 0.6) + (textblob_score * 0.4)
            
            return max(-1, min(1, combined_score))
            
        except Exception as e:
            print(f"Error analyzing text sentiment: {e}")
            return 0.0
    
    def _combine_sentiment_data(self, sentiment_data: Dict, days_back: int) -> pd.DataFrame:
        """
        Combine sentiment data from multiple sources.
        
        Args:
            sentiment_data: Dictionary with sentiment data from different sources
            days_back: Number of days to analyze
            
        Returns:
            DataFrame with combined sentiment analysis
        """
        try:
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            combined_data = []
            
            for date in dates:
                daily_sentiment = {
                    'date': date,
                    'overall_sentiment': 0.0,
                    'sentiment_confidence': 0.0,
                    'news_sentiment': 0.0,
                    'social_sentiment': 0.0,
                    'analyst_sentiment': 0.0,
                    'public_sentiment': 0.0,
                    'sentiment_volume': 0,
                    'market_mood': 'neutral'
                }
                
                # News sentiment
                if 'news_sentiment' in sentiment_data:
                    news_df = sentiment_data['news_sentiment']
                    if not news_df.empty:
                        date_news = news_df[news_df['date'].dt.date == date.date()]
                        if not date_news.empty:
                            daily_sentiment['news_sentiment'] = date_news['sentiment_score'].mean()
                
                # Social sentiment
                if 'social_sentiment' in sentiment_data:
                    social = sentiment_data['social_sentiment']
                    if social:
                        daily_sentiment['social_sentiment'] = (
                            social.get('twitter_sentiment', 0) * 0.4 +
                            social.get('reddit_sentiment', 0) * 0.3 +
                            social.get('stocktwits_sentiment', 0) * 0.3
                        )
                        daily_sentiment['sentiment_volume'] += social.get('social_volume', 0)
                
                # Analyst sentiment
                if 'analyst_sentiment' in sentiment_data:
                    analyst = sentiment_data['analyst_sentiment']
                    if analyst:
                        daily_sentiment['analyst_sentiment'] = analyst.get('average_rating', 0.5)
                
                # Public sentiment
                if 'public_sentiment' in sentiment_data:
                    public = sentiment_data['public_sentiment']
                    if public:
                        daily_sentiment['public_sentiment'] = public.get('overall_sentiment', 0)
                        daily_sentiment['sentiment_volume'] += public.get('sentiment_volume', 0)
                        daily_sentiment['market_mood'] = public.get('market_mood', 'neutral')
                
                # Calculate overall sentiment (weighted average)
                overall_sentiment = (
                    daily_sentiment['news_sentiment'] * self.sentiment_weights['news'] +
                    daily_sentiment['social_sentiment'] * self.sentiment_weights['social_media'] +
                    daily_sentiment['analyst_sentiment'] * self.sentiment_weights['analyst_ratings'] +
                    daily_sentiment['public_sentiment'] * self.sentiment_weights['public_sentiment']
                )
                
                daily_sentiment['overall_sentiment'] = overall_sentiment
                
                # Calculate confidence based on volume and consistency
                sentiment_scores = [
                    daily_sentiment['news_sentiment'],
                    daily_sentiment['social_sentiment'],
                    daily_sentiment['analyst_sentiment'],
                    daily_sentiment['public_sentiment']
                ]
                
                # Confidence based on volume and consistency
                volume_factor = min(1.0, daily_sentiment['sentiment_volume'] / 1000)
                consistency_factor = 1.0 - np.std([s for s in sentiment_scores if s != 0])
                daily_sentiment['sentiment_confidence'] = (volume_factor + consistency_factor) / 2
                
                combined_data.append(daily_sentiment)
            
            return pd.DataFrame(combined_data)
            
        except Exception as e:
            print(f"Error combining sentiment data: {e}")
            return pd.DataFrame()
    
    def get_sentiment_summary(self, ticker: str, days_back: int = 30) -> Dict:
        """
        Get a summary of sentiment analysis.
        
        Args:
            ticker: Stock ticker
            days_back: Number of days to analyze
            
        Returns:
            Dictionary with sentiment summary
        """
        try:
            sentiment_df = self.analyze_stock_sentiment(ticker, days_back)
            
            if sentiment_df.empty:
                return {}
            
            summary = {
                'ticker': ticker,
                'analysis_period': f'{days_back} days',
                'overall_sentiment': sentiment_df['overall_sentiment'].mean(),
                'sentiment_trend': 'increasing' if sentiment_df['overall_sentiment'].iloc[-1] > sentiment_df['overall_sentiment'].iloc[0] else 'decreasing',
                'sentiment_volatility': sentiment_df['overall_sentiment'].std(),
                'confidence_score': sentiment_df['sentiment_confidence'].mean(),
                'market_mood': sentiment_df['market_mood'].mode().iloc[0] if not sentiment_df['market_mood'].mode().empty else 'neutral',
                'sentiment_volume': sentiment_df['sentiment_volume'].sum(),
                'positive_days': len(sentiment_df[sentiment_df['overall_sentiment'] > 0.1]),
                'negative_days': len(sentiment_df[sentiment_df['overall_sentiment'] < -0.1]),
                'neutral_days': len(sentiment_df[(sentiment_df['overall_sentiment'] >= -0.1) & (sentiment_df['overall_sentiment'] <= 0.1)])
            }
            
            return summary
            
        except Exception as e:
            print(f"Error generating sentiment summary: {e}")
            return {}

# Convenience function for backward compatibility
def analyze_stock_sentiment(ticker: str, days_back: int = 30) -> pd.DataFrame:
    """Backward compatibility function."""
    analyzer = OptimizedSentimentAnalyzer()
    return analyzer.analyze_stock_sentiment(ticker, days_back)
