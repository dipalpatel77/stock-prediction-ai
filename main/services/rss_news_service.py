#!/usr/bin/env python3
"""
RSS News Service
RSS feed-based news collection for sentiment analysis
"""

import requests
import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class RSSNewsService:
    """
    RSS News Service for collecting news from RSS feeds
    
    Features:
    - Multiple RSS feed support
    - Topic filtering
    - Article deduplication
    - Error handling and fallbacks
    - Real-time news collection
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize RSS News Service
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # RSS feed sources for Indian financial news
        self.rss_sources = {
            'economic_times': {
                'url': 'https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms',
                'name': 'Economic Times - Markets',
                'category': 'markets',
                'priority': 1,
                'enabled': True
            },
            'money_control': {
                'url': 'https://www.moneycontrol.com/rss/business.xml',
                'name': 'Money Control - Business',
                'category': 'business',
                'priority': 2,
                'enabled': True
            },
            'livemint': {
                'url': 'https://www.livemint.com/rss/markets',
                'name': 'LiveMint - Markets',
                'category': 'markets',
                'priority': 3,
                'enabled': True
            },
            'business_standard': {
                'url': 'https://www.business-standard.com/rss/markets-10608147.rss',
                'name': 'Business Standard - Markets',
                'category': 'markets',
                'priority': 4,
                'enabled': True
            }
        }
        
        # Request headers to avoid blocking
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/rss+xml, application/xml, text/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        }
        
        self.logger.info("RSS News Service initialized")
    
    def get_news_for_topic(self, topic: str, max_articles: int = 5, hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get news articles for a specific topic from RSS feeds
        
        Args:
            topic: News topic to search for
            max_articles: Maximum number of articles to return
            hours_back: How many hours back to look for news
            
        Returns:
            List of news articles
        """
        try:
            self.logger.info(f"🔍 Searching RSS feeds for topic: {topic}")
            
            all_articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Try each RSS source
            for source_id, source_info in self.rss_sources.items():
                if not source_info.get('enabled', True):
                    continue
                    
                try:
                    self.logger.info(f"📡 Checking {source_info['name']}...")
                    articles = self._parse_rss_feed(source_info['url'], topic, cutoff_time)
                    
                    if articles:
                        self.logger.info(f"✅ Found {len(articles)} articles from {source_info['name']}")
                        all_articles.extend(articles)
                    else:
                        self.logger.info(f"ℹ️ No articles found in {source_info['name']}")
                        
                except Exception as e:
                    self.logger.warning(f"❌ Failed to parse {source_info['name']}: {e}")
                    continue
            
            # If no specific articles found, try broader search
            if not all_articles:
                self.logger.info("🔍 No specific articles found, trying broader search...")
                for source_id, source_info in self.rss_sources.items():
                    if not source_info.get('enabled', True):
                        continue
                        
                    try:
                        # Try with broader terms
                        broader_terms = ['market', 'stock', 'business', 'financial']
                        for term in broader_terms:
                            articles = self._parse_rss_feed(source_info['url'], term, cutoff_time)
                            if articles:
                                all_articles.extend(articles[:2])  # Limit to 2 per source
                                break
                    except Exception as e:
                        self.logger.warning(f"❌ Broader search failed for {source_info['name']}: {e}")
                        continue
            
            # If still no articles, get any recent articles (fallback)
            if not all_articles:
                self.logger.info("🔍 No filtered articles found, getting recent articles as fallback...")
                for source_id, source_info in self.rss_sources.items():
                    if not source_info.get('enabled', True):
                        continue
                        
                    try:
                        # Get any recent articles without filtering
                        articles = self._parse_rss_feed_without_filter(source_info['url'], cutoff_time)
                        if articles:
                            all_articles.extend(articles[:3])  # Limit to 3 per source
                            break
                    except Exception as e:
                        self.logger.warning(f"❌ Fallback search failed for {source_info['name']}: {e}")
                        continue
            
            # Deduplicate articles
            unique_articles = self._deduplicate_articles(all_articles)
            
            # Sort by priority and recency
            sorted_articles = self._sort_articles(unique_articles)
            
            # Limit results
            final_articles = sorted_articles[:max_articles]
            
            self.logger.info(f"📰 Total articles found: {len(final_articles)}")
            return final_articles
            
        except Exception as e:
            self.logger.error(f"RSS news collection failed: {e}")
            return []
    
    def _parse_rss_feed(self, rss_url: str, topic: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """
        Parse a single RSS feed
        
        Args:
            rss_url: RSS feed URL
            topic: Topic to filter for
            cutoff_time: Only include articles after this time
            
        Returns:
            List of articles from this feed
        """
        try:
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            articles = []
            
            # Handle different RSS formats
            items = root.findall('.//item')
            if not items:
                # Try alternative RSS structure
                items = root.findall('.//entry')
            
            for item in items:
                try:
                    article = self._parse_rss_item(item, topic, cutoff_time)
                    if article:
                        articles.append(article)
                except Exception as e:
                    self.logger.warning(f"Failed to parse RSS item: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"RSS feed parsing failed for {rss_url}: {e}")
            return []
    
    def _parse_rss_feed_without_filter(self, rss_url: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """
        Parse RSS feed without topic filtering (fallback method)
        
        Args:
            rss_url: RSS feed URL
            cutoff_time: Only include articles after this time
            
        Returns:
            List of articles from this feed
        """
        try:
            response = requests.get(rss_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            articles = []
            
            # Handle different RSS formats
            items = root.findall('.//item')
            if not items:
                # Try alternative RSS structure
                items = root.findall('.//entry')
            
            for i, item in enumerate(items):
                try:
                    self.logger.debug(f"Processing item {i+1}/{len(items)}")
                    # Debug: Check item structure
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    self.logger.debug(f"Item {i+1} - title_elem: {title_elem}, link_elem: {link_elem}")
                    
                    article = self._parse_rss_item_without_filter(item, cutoff_time)
                    if article:
                        articles.append(article)
                        self.logger.debug(f"Item {i+1} parsed successfully")
                    else:
                        self.logger.debug(f"Item {i+1} parsing returned None")
                except Exception as e:
                    self.logger.warning(f"Failed to parse RSS item {i+1}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"RSS feed parsing failed for {rss_url}: {e}")
            return []
    
    def _parse_rss_item_without_filter(self, item: ET.Element, cutoff_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Parse RSS item without topic filtering
        
        Args:
            item: XML element for the RSS item
            cutoff_time: Only include articles after this time
            
        Returns:
            Parsed article or None if not recent
        """
        try:
            # Extract basic fields
            title_elem = item.find('title')
            link_elem = item.find('link')
            description_elem = item.find('description')
            pub_date_elem = item.find('pubDate')
            
            self.logger.debug(f"Inside _parse_rss_item_without_filter - title_elem: {title_elem}, link_elem: {link_elem}")
            # Check if elements exist and have content
            if title_elem is None or link_elem is None:
                self.logger.debug("Missing title or link element")
                return None
            
            # Check if elements have text content
            if not title_elem.text or not link_elem.text:
                self.logger.debug(f"Empty text - title: '{title_elem.text}', link: '{link_elem.text}'")
                return None
            
            title = title_elem.text.strip() if title_elem.text else ""
            link = link_elem.text.strip() if link_elem.text else ""
            description = description_elem.text.strip() if description_elem and description_elem.text else ""
            
            # Parse publication date
            pub_date = self._parse_pub_date(pub_date_elem.text if pub_date_elem is not None else None)
            if cutoff_time and pub_date and pub_date < cutoff_time:
                return None
            
            # Extract additional metadata
            author = self._extract_author(item)
            category = self._extract_category(item)
            
            return {
                'title': title,
                'url': link,
                'text': f"{title}. {description}",
                'description': description,
                'published_date': pub_date.isoformat() if pub_date else None,
                'author': author,
                'category': category,
                'source': 'rss_feed',
                'relevance_score': 0.5  # Default relevance for fallback
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse RSS item: {e}")
            return None
    
    def _parse_rss_item(self, item: ET.Element, topic: str, cutoff_time: datetime) -> Optional[Dict[str, Any]]:
        """
        Parse a single RSS item
        
        Args:
            item: XML element for the RSS item
            topic: Topic to filter for
            cutoff_time: Only include articles after this time
            
        Returns:
            Parsed article or None if not relevant
        """
        try:
            # Extract basic fields
            title_elem = item.find('title')
            link_elem = item.find('link')
            description_elem = item.find('description')
            pub_date_elem = item.find('pubDate')
            
            if not title_elem or not link_elem:
                return None
            
            title = title_elem.text.strip() if title_elem.text else ""
            link = link_elem.text.strip() if link_elem.text else ""
            description = description_elem.text.strip() if description_elem and description_elem.text else ""
            
            # Parse publication date
            pub_date = self._parse_pub_date(pub_date_elem.text if pub_date_elem is not None else None)
            if cutoff_time and pub_date and pub_date < cutoff_time:
                return None
            
            # Check if article is relevant to topic
            if not self._is_relevant_to_topic(title, description, topic):
                return None
            
            # Extract additional metadata
            author = self._extract_author(item)
            category = self._extract_category(item)
            
            return {
                'title': title,
                'url': link,
                'text': f"{title}. {description}",
                'description': description,
                'published_date': pub_date.isoformat() if pub_date else None,
                'author': author,
                'category': category,
                'source': 'rss_feed',
                'relevance_score': self._calculate_relevance_score(title, description, topic)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to parse RSS item: {e}")
            return None
    
    def _is_relevant_to_topic(self, title: str, description: str, topic: str) -> bool:
        """
        Check if article is relevant to the topic
        
        Args:
            title: Article title
            description: Article description
            topic: Search topic
            
        Returns:
            True if relevant
        """
        try:
            # Convert to lowercase for case-insensitive matching
            title_lower = title.lower()
            description_lower = description.lower()
            topic_lower = topic.lower()
            
            # Extract company/stock names from topic
            topic_words = topic_lower.split()
            
            # Check for direct matches
            for word in topic_words:
                if word in title_lower or word in description_lower:
                    self.logger.debug(f"Direct match found: '{word}' in '{title_lower}'")
                    return True
            
            # Check for common financial terms
            financial_terms = ['earnings', 'revenue', 'profit', 'stock', 'market', 'trading', 'investment', 'business', 'company', 'corporate', 'financial']
            if any(term in title_lower or term in description_lower for term in financial_terms):
                self.logger.debug(f"Financial term match found in '{title_lower}'")
                return True
            
            # For specific companies, be more inclusive
            if len(topic_words) == 1 and topic_lower in ['tcs', 'reliance', 'infosys', 'hdfc', 'icici']:
                # For specific companies, include any market-related news
                market_terms = ['stock', 'market', 'trading', 'price', 'share', 'equity', 'investor', 'sensex', 'nifty', 'bse', 'nse']
                if any(term in title_lower or term in description_lower for term in market_terms):
                    self.logger.debug(f"Market term match found for {topic_lower} in '{title_lower}'")
                    return True
            
            # For IT companies, include technology-related terms
            if topic_lower in ['tcs', 'infosys', 'wipro', 'hcl', 'tech mahindra']:
                tech_terms = ['technology', 'tech', 'software', 'it', 'digital', 'cloud', 'ai', 'artificial intelligence']
                if any(term in title_lower or term in description_lower for term in tech_terms):
                    self.logger.debug(f"Tech term match found for {topic_lower} in '{title_lower}'")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Relevance check failed: {e}")
            return False
    
    def _calculate_relevance_score(self, title: str, description: str, topic: str) -> float:
        """
        Calculate relevance score for article
        
        Args:
            title: Article title
            description: Article description
            topic: Search topic
            
        Returns:
            Relevance score (0-1)
        """
        try:
            score = 0.0
            title_lower = title.lower()
            description_lower = description.lower()
            topic_lower = topic.lower()
            
            # Title matches are more important
            if topic_lower in title_lower:
                score += 0.5
            
            # Description matches
            if topic_lower in description_lower:
                score += 0.3
            
            # Word-level matches
            topic_words = topic_lower.split()
            for word in topic_words:
                if word in title_lower:
                    score += 0.1
                if word in description_lower:
                    score += 0.05
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.warning(f"Relevance score calculation failed: {e}")
            return 0.0
    
    def _parse_pub_date(self, pub_date_str: str) -> Optional[datetime]:
        """
        Parse publication date from RSS
        
        Args:
            pub_date_str: Publication date string
            
        Returns:
            Parsed datetime or None
        """
        try:
            if not pub_date_str:
                return None
            
            # Common RSS date formats
            date_formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ'
            ]
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(pub_date_str.strip(), fmt)
                    # Ensure the datetime is offset-naive
                    if parsed_date.tzinfo is not None:
                        parsed_date = parsed_date.replace(tzinfo=None)
                    return parsed_date
                except ValueError:
                    continue
            
            # If no format matches, return current time
            return datetime.now()
            
        except Exception as e:
            self.logger.warning(f"Date parsing failed: {e}")
            return datetime.now()
    
    def _extract_author(self, item: ET.Element) -> Optional[str]:
        """Extract author from RSS item"""
        try:
            author_elem = item.find('author')
            if author_elem is not None and author_elem.text:
                return author_elem.text.strip()
            return None
        except:
            return None
    
    def _extract_category(self, item: ET.Element) -> Optional[str]:
        """Extract category from RSS item"""
        try:
            category_elem = item.find('category')
            if category_elem is not None and category_elem.text:
                return category_elem.text.strip()
            return None
        except:
            return None
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate articles based on title similarity
        
        Args:
            articles: List of articles
            
        Returns:
            Deduplicated list
        """
        try:
            unique_articles = []
            seen_titles = set()
            
            for article in articles:
                title = article.get('title', '').lower().strip()
                
                # Simple deduplication based on title
                if title not in seen_titles:
                    seen_titles.add(title)
                    unique_articles.append(article)
            
            return unique_articles
            
        except Exception as e:
            self.logger.warning(f"Deduplication failed: {e}")
            return articles
    
    def _sort_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort articles by relevance and recency
        
        Args:
            articles: List of articles
            
        Returns:
            Sorted list
        """
        try:
            def sort_key(article):
                relevance = article.get('relevance_score', 0)
                pub_date = article.get('published_date')
                
                # Higher relevance score is better
                score = relevance
                
                # Recent articles are better
                if pub_date:
                    try:
                        pub_dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        hours_ago = (datetime.now() - pub_dt).total_seconds() / 3600
                        score += max(0, 1 - hours_ago / 24)  # Decay over 24 hours
                    except:
                        pass
                
                return score
            
            return sorted(articles, key=sort_key, reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Article sorting failed: {e}")
            return articles
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get service status and capabilities
        
        Returns:
            Service status dictionary
        """
        return {
            'service_type': 'rss_news',
            'sources_configured': len(self.rss_sources),
            'sources': list(self.rss_sources.keys()),
            'initialized': True,
            'capabilities': [
                'multi_source_aggregation',
                'topic_filtering',
                'article_deduplication',
                'relevance_scoring',
                'real_time_news'
            ]
        }
