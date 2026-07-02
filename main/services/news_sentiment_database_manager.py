#!/usr/bin/env python3
"""
News Sentiment Database Manager
Database operations for news sentiment analysis data
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class NewsSentimentDatabaseManager:
    """
    Database manager for news sentiment analysis data
    
    Features:
    - News sentiment data storage
    - Historical sentiment retrieval
    - Sentiment trend analysis
    - Database optimization
    """
    
    def __init__(self, db_path: str = "news_sentiment.db"):
        """
        Initialize News Sentiment Database Manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        
        logger.info(f"News Sentiment Database Manager initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create news sentiment table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    title TEXT,
                    url TEXT,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entity TEXT,
                    sector TEXT,
                    balance_confidence REAL,
                    final_confidence REAL NOT NULL,
                    impact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create sentiment trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    entity TEXT NOT NULL,
                    sector TEXT,
                    avg_sentiment REAL NOT NULL,
                    avg_confidence REAL NOT NULL,
                    article_count INTEGER NOT NULL,
                    high_impact_count INTEGER DEFAULT 0,
                    medium_impact_count INTEGER DEFAULT 0,
                    low_impact_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, entity)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_sentiment(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_entity ON news_sentiment(entity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_sentiment(sentiment)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_date ON sentiment_trends(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_entity ON sentiment_trends(entity)")
            
            conn.commit()
            conn.close()
            
            logger.info("Database tables initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def save_sentiment_analysis(self, analysis_results: List[Dict[str, Any]], topic: str) -> bool:
        """
        Save sentiment analysis results to database
        
        Args:
            analysis_results: List of analysis results
            topic: Analysis topic
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for result in analysis_results:
                cursor.execute("""
                    INSERT INTO news_sentiment (
                        timestamp, topic, title, url, sentiment, confidence,
                        entity, sector, balance_confidence, final_confidence, impact
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    topic,
                    result.get('title', ''),
                    result.get('url', ''),
                    result['sentiment'],
                    result['confidence'],
                    list(result['entities'].keys())[0] if result['entities'] else 'General',
                    list(result['entities'].values())[0] if result['entities'] else 'Economy',
                    result['balance_confidence'],
                    result['final_confidence'],
                    result['impact']
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Saved {len(analysis_results)} sentiment analysis results for topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save sentiment analysis: {e}")
            return False
    
    def get_sentiment_history(self, entity: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get sentiment history for an entity
        
        Args:
            entity: Entity name (optional)
            days: Number of days to look back
            
        Returns:
            List of sentiment records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            if entity:
                cursor.execute("""
                    SELECT timestamp, title, sentiment, confidence, final_confidence, impact, entity, sector
                    FROM news_sentiment
                    WHERE entity = ? AND date(timestamp) >= ?
                    ORDER BY timestamp DESC
                """, (entity, start_date))
            else:
                cursor.execute("""
                    SELECT timestamp, title, sentiment, confidence, final_confidence, impact, entity, sector
                    FROM news_sentiment
                    WHERE date(timestamp) >= ?
                    ORDER BY timestamp DESC
                """, (start_date,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'timestamp': row[0],
                    'title': row[1],
                    'sentiment': row[2],
                    'confidence': row[3],
                    'final_confidence': row[4],
                    'impact': row[5],
                    'entity': row[6],
                    'sector': row[7]
                })
            
            conn.close()
            
            logger.info(f"Retrieved {len(results)} sentiment records")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get sentiment history: {e}")
            return []
    
    def get_sentiment_trends(self, entity: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get sentiment trends for an entity
        
        Args:
            entity: Entity name (optional)
            days: Number of days to look back
            
        Returns:
            List of sentiment trend records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            if entity:
                cursor.execute("""
                    SELECT date, entity, sector, avg_sentiment, avg_confidence, article_count,
                           high_impact_count, medium_impact_count, low_impact_count
                    FROM sentiment_trends
                    WHERE entity = ? AND date >= ?
                    ORDER BY date DESC
                """, (entity, start_date))
            else:
                cursor.execute("""
                    SELECT date, entity, sector, avg_sentiment, avg_confidence, article_count,
                           high_impact_count, medium_impact_count, low_impact_count
                    FROM sentiment_trends
                    WHERE date >= ?
                    ORDER BY date DESC
                """, (start_date,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'date': row[0],
                    'entity': row[1],
                    'sector': row[2],
                    'avg_sentiment': row[3],
                    'avg_confidence': row[4],
                    'article_count': row[5],
                    'high_impact_count': row[6],
                    'medium_impact_count': row[7],
                    'low_impact_count': row[8]
                })
            
            conn.close()
            
            logger.info(f"Retrieved {len(results)} sentiment trend records")
            return results
            
        except Exception as e:
            logger.error(f"Failed to get sentiment trends: {e}")
            return []
    
    def calculate_daily_trends(self, date: str = None) -> bool:
        """
        Calculate daily sentiment trends
        
        Args:
            date: Date to calculate trends for (default: today)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get daily sentiment data
            cursor.execute("""
                SELECT entity, sector, sentiment, confidence, impact
                FROM news_sentiment
                WHERE date(timestamp) = ?
            """, (date,))
            
            daily_data = cursor.fetchall()
            
            if not daily_data:
                logger.info(f"No sentiment data found for date: {date}")
                conn.close()
                return True
            
            # Group by entity
            entity_data = {}
            for row in daily_data:
                entity, sector, sentiment, confidence, impact = row
                
                if entity not in entity_data:
                    entity_data[entity] = {
                        'sector': sector,
                        'sentiments': [],
                        'confidences': [],
                        'impacts': []
                    }
                
                entity_data[entity]['sentiments'].append(sentiment)
                entity_data[entity]['confidences'].append(confidence)
                entity_data[entity]['impacts'].append(impact)
            
            # Calculate trends for each entity
            for entity, data in entity_data.items():
                sentiments = data['sentiments']
                confidences = data['confidences']
                impacts = data['impacts']
                
                # Calculate averages
                sentiment_scores = []
                for sentiment in sentiments:
                    if sentiment == 'positive':
                        sentiment_scores.append(1.0)
                    elif sentiment == 'negative':
                        sentiment_scores.append(-1.0)
                    else:
                        sentiment_scores.append(0.0)
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                avg_confidence = sum(confidences) / len(confidences)
                
                # Count impacts
                high_impact = impacts.count('High Impact')
                medium_impact = impacts.count('Medium Impact')
                low_impact = impacts.count('Low Impact')
                
                # Insert or update trend
                cursor.execute("""
                    INSERT OR REPLACE INTO sentiment_trends (
                        date, entity, sector, avg_sentiment, avg_confidence, article_count,
                        high_impact_count, medium_impact_count, low_impact_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, entity, data['sector'], avg_sentiment, avg_confidence,
                    len(sentiments), high_impact, medium_impact, low_impact
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Calculated daily trends for {len(entity_data)} entities on {date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to calculate daily trends: {e}")
            return False
    
    def get_sentiment_summary(self, entity: str = None, days: int = 7) -> Dict[str, Any]:
        """
        Get sentiment summary for an entity
        
        Args:
            entity: Entity name (optional)
            days: Number of days to look back
            
        Returns:
            Sentiment summary dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            if entity:
                cursor.execute("""
                    SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence,
                           AVG(final_confidence) as avg_final_confidence
                    FROM news_sentiment
                    WHERE entity = ? AND date(timestamp) >= ?
                    GROUP BY sentiment
                """, (entity, start_date))
            else:
                cursor.execute("""
                    SELECT sentiment, COUNT(*) as count, AVG(confidence) as avg_confidence,
                           AVG(final_confidence) as avg_final_confidence
                    FROM news_sentiment
                    WHERE date(timestamp) >= ?
                    GROUP BY sentiment
                """, (start_date,))
            
            sentiment_counts = {}
            total_articles = 0
            total_confidence = 0
            
            for row in cursor.fetchall():
                sentiment, count, avg_confidence, avg_final_confidence = row
                sentiment_counts[sentiment] = {
                    'count': count,
                    'avg_confidence': avg_confidence,
                    'avg_final_confidence': avg_final_confidence
                }
                total_articles += count
                total_confidence += avg_final_confidence * count
            
            # Calculate overall sentiment
            positive_count = sentiment_counts.get('positive', {}).get('count', 0)
            negative_count = sentiment_counts.get('negative', {}).get('count', 0)
            neutral_count = sentiment_counts.get('neutral', {}).get('count', 0)
            
            if total_articles > 0:
                overall_sentiment = (positive_count - negative_count) / total_articles
                avg_confidence = total_confidence / total_articles
            else:
                overall_sentiment = 0
                avg_confidence = 0
            
            conn.close()
            
            summary = {
                'total_articles': total_articles,
                'overall_sentiment': overall_sentiment,
                'avg_confidence': avg_confidence,
                'sentiment_breakdown': sentiment_counts,
                'positive_ratio': positive_count / total_articles if total_articles > 0 else 0,
                'negative_ratio': negative_count / total_articles if total_articles > 0 else 0,
                'neutral_ratio': neutral_count / total_articles if total_articles > 0 else 0
            }
            
            logger.info(f"Generated sentiment summary for {entity or 'all entities'}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get sentiment summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """
        Clean up old sentiment data
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old sentiment data
            cursor.execute("DELETE FROM news_sentiment WHERE date(timestamp) < ?", (cutoff_date,))
            deleted_sentiment = cursor.rowcount
            
            # Delete old trend data
            cursor.execute("DELETE FROM sentiment_trends WHERE date < ?", (cutoff_date,))
            deleted_trends = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_sentiment} sentiment records and {deleted_trends} trend records")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Database statistics dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM news_sentiment")
            total_sentiment = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sentiment_trends")
            total_trends = cursor.fetchone()[0]
            
            # Get unique entities
            cursor.execute("SELECT COUNT(DISTINCT entity) FROM news_sentiment")
            unique_entities = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM news_sentiment")
            date_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_sentiment_records': total_sentiment,
                'total_trend_records': total_trends,
                'unique_entities': unique_entities,
                'date_range': {
                    'earliest': date_range[0],
                    'latest': date_range[1]
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
