#!/usr/bin/env python3
"""
News Sentiment Analyzer Pipeline Component
Pipeline component for news sentiment analysis
"""

import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_pipeline import BasePipelineComponent

logger = logging.getLogger(__name__)

class NewsSentimentAnalyzer(BasePipelineComponent):
    """
    News Sentiment Analyzer Pipeline Component
    
    Features:
    - News sentiment analysis integration
    - Stock-specific news analysis
    - Sentiment trend analysis
    - Impact assessment
    """
    
    def __init__(self, ticker: str = "AAPL", config: Dict[str, Any] = None):
        """
        Initialize News Sentiment Analyzer
        
        Args:
            ticker: Stock ticker symbol
            config: Configuration dictionary
        """
        super().__init__("news_sentiment_analyzer", ticker, config)
        
        # Initialize services
        self._setup_services()
        
        logger.info("News Sentiment Analyzer initialized")
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute news sentiment analysis
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Get parameters from kwargs or use defaults
            stock_name = kwargs.get('stock_name', self.ticker)
            news_topic = kwargs.get('news_topic', f"{stock_name} earnings India")
            analysis_params = kwargs.get('analysis_params', {
                'max_articles': 5,
                'include_balance_sheet': True,
                'sentiment_weight': 0.7,
                'balance_weight': 0.3
            })
            
            # Run analysis
            results = self.analyze_stock_news(stock_name, news_topic, analysis_params)
            
            return {
                'success': results.get('success', False),
                'result': results,
                'execution_time': results.get('execution_time', 0),
                'component': self.name
            }
            
        except Exception as e:
            self.logger.error(f"News sentiment analysis execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'component': self.name
            }
    
    def _setup_services(self):
        """Setup required services"""
        try:
            # Try to use the full service first, fallback to simple service
            try:
                from ..services.news_sentiment_service import NewsSentimentService
                self.news_service = NewsSentimentService(self.config)
                logger.info("Full news sentiment service initialized")
            except ImportError:
                from ..services.simple_news_sentiment_service import SimpleNewsSentimentService
                self.news_service = SimpleNewsSentimentService(self.config)
                logger.info("Simple news sentiment service initialized")
            
            # Initialize database manager
            try:
                from ..services.news_sentiment_database_manager import NewsSentimentDatabaseManager
                db_config = self.config.get('database', {})
                self.db_manager = NewsSentimentDatabaseManager(
                    db_path=db_config.get('news_sentiment_db', 'news_sentiment.db')
                )
                logger.info("News sentiment database manager initialized")
            except Exception as e:
                logger.warning(f"Database manager not available: {e}")
                self.db_manager = None
            
            logger.info("News sentiment services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup news sentiment services: {e}")
            self.news_service = None
            self.db_manager = None
    
    def analyze_stock_news(self, stock_name: str, news_topic: str, analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news sentiment for a specific stock
        
        Args:
            stock_name: Name of the stock/company
            news_topic: News topic to analyze
            analysis_params: Analysis parameters
            
        Returns:
            Analysis results dictionary
        """
        try:
            start_time = time.time()
            logger.info(f"🔍 Starting news sentiment analysis for {stock_name}")
            
            # Validate services
            if not self.news_service:
                return {
                    'success': False,
                    'error': 'News sentiment service not available',
                    'execution_time': 0
                }
            
            # Get analysis parameters
            max_articles = analysis_params.get('max_articles', 5)
            include_balance_sheet = analysis_params.get('include_balance_sheet', True)
            sentiment_weight = analysis_params.get('sentiment_weight', 0.7)
            balance_weight = analysis_params.get('balance_weight', 0.3)
            
            # Analyze news
            logger.info(f"📰 Analyzing news for topic: {news_topic}")
            analysis_results = self.news_service.analyze_news_for_topic(
                topic=news_topic,
                max_articles=max_articles
            )
            
            if not analysis_results:
                logger.warning(f"No news articles found for topic: {news_topic}")
                # Provide fallback analysis with general market sentiment
                fallback_result = self._create_fallback_analysis(stock_name, news_topic)
                return {
                    'success': True,
                    'analysis_results': [fallback_result],
                    'summary': {
                        'total_articles': 1,
                        'overall_sentiment': 'neutral',
                        'avg_confidence': 0.5,
                        'high_impact_count': 0,
                        'stock_relevance_avg': 0.5,
                        'sentiment_breakdown': {'positive': 0, 'negative': 0, 'neutral': 1}
                    },
                    'execution_time': time.time() - start_time,
                    'articles_analyzed': 1,
                    'fallback_mode': True
                }
            
            # Process results
            processed_results = self._process_analysis_results(
                analysis_results, stock_name, sentiment_weight, balance_weight
            )
            
            # Save to database if available
            if self.db_manager:
                try:
                    self.db_manager.save_sentiment_analysis(analysis_results, news_topic)
                    logger.info("News sentiment analysis saved to database")
                except Exception as e:
                    logger.warning(f"Failed to save to database: {e}")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create comprehensive results
            results = {
                'success': True,
                'stock_name': stock_name,
                'news_topic': news_topic,
                'analysis_results': processed_results,
                'summary': self._create_analysis_summary(processed_results),
                'execution_time': execution_time,
                'articles_analyzed': len(analysis_results),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ News sentiment analysis completed in {execution_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"News sentiment analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _process_analysis_results(self, analysis_results: List[Dict[str, Any]], 
                                stock_name: str, sentiment_weight: float, 
                                balance_weight: float) -> List[Dict[str, Any]]:
        """
        Process and enhance analysis results
        
        Args:
            analysis_results: Raw analysis results
            stock_name: Stock name
            sentiment_weight: Sentiment weight for final confidence
            balance_weight: Balance sheet weight for final confidence
            
        Returns:
            Processed analysis results
        """
        try:
            processed_results = []
            
            for result in analysis_results:
                # Recalculate final confidence with custom weights
                final_confidence = round(
                    (sentiment_weight * result['confidence']) + 
                    (balance_weight * result['balance_confidence']), 3
                )
                
                # Update impact label
                impact = self.news_service.get_impact_label(final_confidence)
                
                # Create enhanced result
                enhanced_result = {
                    'title': result['title'],
                    'url': result['url'],
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'balance_confidence': result['balance_confidence'],
                    'final_confidence': final_confidence,
                    'impact': impact,
                    'entities': result['entities'],
                    'stock_relevance': self._calculate_stock_relevance(result, stock_name),
                    'sector_impact': self._calculate_sector_impact(result)
                }
                
                processed_results.append(enhanced_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to process analysis results: {e}")
            return analysis_results
    
    def _calculate_stock_relevance(self, result: Dict[str, Any], stock_name: str) -> float:
        """
        Calculate stock relevance score
        
        Args:
            result: Analysis result
            stock_name: Stock name
            
        Returns:
            Relevance score (0-1)
        """
        try:
            title = result.get('title', '').lower()
            entities = result.get('entities', {})
            
            # Check if stock name appears in title
            if stock_name.lower() in title:
                return 1.0
            
            # Check if stock name appears in entities
            for entity in entities.keys():
                if stock_name.lower() in entity.lower():
                    return 0.8
            
            # Check for related terms
            related_terms = ['earnings', 'revenue', 'profit', 'growth', 'stock', 'shares']
            relevance = 0.0
            for term in related_terms:
                if term in title:
                    relevance += 0.2
            
            return min(1.0, relevance)
            
        except Exception as e:
            logger.error(f"Failed to calculate stock relevance: {e}")
            return 0.5
    
    def _calculate_sector_impact(self, result: Dict[str, Any]) -> str:
        """
        Calculate sector impact level
        
        Args:
            result: Analysis result
            
        Returns:
            Sector impact level
        """
        try:
            entities = result.get('entities', {})
            if not entities:
                return "General Market"
            
            # Get primary sector
            sectors = list(entities.values())
            if sectors:
                primary_sector = sectors[0]
                return f"{primary_sector} Sector"
            
            return "General Market"
            
        except Exception as e:
            logger.error(f"Failed to calculate sector impact: {e}")
            return "General Market"
    
    def _create_analysis_summary(self, processed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create analysis summary
        
        Args:
            processed_results: Processed analysis results
            
        Returns:
            Analysis summary dictionary
        """
        try:
            if not processed_results:
                return {
                    'total_articles': 0,
                    'overall_sentiment': 'neutral',
                    'avg_confidence': 0.0,
                    'high_impact_count': 0,
                    'stock_relevance_avg': 0.0
                }
            
            # Calculate summary statistics
            total_articles = len(processed_results)
            positive_count = sum(1 for r in processed_results if r['sentiment'] == 'positive')
            negative_count = sum(1 for r in processed_results if r['sentiment'] == 'negative')
            neutral_count = sum(1 for r in processed_results if r['sentiment'] == 'neutral')
            
            avg_confidence = sum(r['final_confidence'] for r in processed_results) / total_articles
            high_impact_count = sum(1 for r in processed_results if r['impact'] == 'High Impact')
            stock_relevance_avg = sum(r['stock_relevance'] for r in processed_results) / total_articles
            
            # Calculate overall sentiment
            sentiment_score = (positive_count - negative_count) / total_articles
            if sentiment_score > 0.1:
                overall_sentiment = 'positive'
            elif sentiment_score < -0.1:
                overall_sentiment = 'negative'
            else:
                overall_sentiment = 'neutral'
            
            # Get sector breakdown
            sector_breakdown = {}
            for result in processed_results:
                sector_impact = result['sector_impact']
                if sector_impact not in sector_breakdown:
                    sector_breakdown[sector_impact] = 0
                sector_breakdown[sector_impact] += 1
            
            return {
                'total_articles': total_articles,
                'overall_sentiment': overall_sentiment,
                'sentiment_score': sentiment_score,
                'avg_confidence': avg_confidence,
                'high_impact_count': high_impact_count,
                'stock_relevance_avg': stock_relevance_avg,
                'sentiment_breakdown': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'sector_breakdown': sector_breakdown
            }
            
        except Exception as e:
            logger.error(f"Failed to create analysis summary: {e}")
            return {
                'total_articles': 0,
                'overall_sentiment': 'neutral',
                'avg_confidence': 0.0,
                'high_impact_count': 0,
                'stock_relevance_avg': 0.0
            }
    
    def get_sentiment_history(self, stock_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment history for a stock
        
        Args:
            stock_name: Stock name
            days: Number of days to look back
            
        Returns:
            Sentiment history dictionary
        """
        try:
            if not self.db_manager:
                return {'success': False, 'error': 'Database manager not available'}
            
            # Get sentiment history
            history = self.db_manager.get_sentiment_history(stock_name, days)
            
            if not history:
                return {
                    'success': True,
                    'history': [],
                    'summary': 'No historical data available'
                }
            
            # Create summary
            summary = self.db_manager.get_sentiment_summary(stock_name, days)
            
            return {
                'success': True,
                'history': history,
                'summary': summary,
                'days_analyzed': days
            }
            
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_component_status(self) -> Dict[str, Any]:
        """
        Get component status
        
        Returns:
            Component status dictionary
        """
        try:
            return {
                'initialized': True,
                'news_service_available': self.news_service is not None,
                'database_manager_available': self.db_manager is not None,
                'component_type': 'news_sentiment_analyzer'
            }
            
        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return {'error': str(e)}
    
    def _create_fallback_analysis(self, stock_name: str, news_topic: str) -> Dict[str, Any]:
        """
        Create fallback analysis when no news articles are found
        
        Args:
            stock_name: Name of the stock/company
            news_topic: News topic that was searched
            
        Returns:
            Fallback analysis result
        """
        try:
            # Create a general market sentiment analysis
            fallback_text = f"General market analysis for {stock_name}. {news_topic} coverage may be limited. Consider broader market trends and sector performance."
            
            # Analyze sentiment of the fallback text
            sentiment, confidence = self.news_service.analyze_sentiment(fallback_text)
            
            # Extract entities
            entities = self.news_service.extract_entities_and_sector(fallback_text)
            
            # Get balance sheet confidence
            balance_confidence = 0.5
            if entities:
                for comp, (ticker, _) in entities.items():
                    balance_confidence = self.news_service.get_balance_sheet_strength(ticker)
                    break
            
            # Calculate final confidence (lower for fallback)
            final_confidence = round((0.7 * confidence) + (0.3 * balance_confidence), 3)
            impact = self.news_service.get_impact_label(final_confidence)
            
            return {
                "title": f"Market Analysis for {stock_name}",
                "url": "fallback_analysis",
                "sentiment": sentiment,
                "confidence": confidence,
                "balance_confidence": balance_confidence,
                "final_confidence": final_confidence,
                "impact": impact,
                "entities": {comp: sector for comp, (_, sector) in entities.items()} if entities else {"General": "Economy"},
                "fallback": True,
                "note": "No recent news articles found. This is a general market analysis."
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis creation failed: {e}")
            # Return minimal fallback
            return {
                "title": f"Market Analysis for {stock_name}",
                "url": "fallback_analysis",
                "sentiment": "neutral",
                "confidence": 0.5,
                "balance_confidence": 0.5,
                "final_confidence": 0.5,
                "impact": "Low Impact",
                "entities": {"General": "Economy"},
                "fallback": True,
                "note": "Limited news coverage available. Consider broader market analysis."
            }
