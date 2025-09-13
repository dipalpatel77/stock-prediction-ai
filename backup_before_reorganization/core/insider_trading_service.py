#!/usr/bin/env python3
"""
Advanced Insider Trading Analysis Service
========================================

This module provides comprehensive insider trading analysis capabilities:
- Insider transaction tracking and analysis
- Pattern recognition and anomaly detection
- Executive trading behavior analysis
- Market impact assessment
- Compliance monitoring
- Sentiment analysis based on insider activity
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import time
from collections import defaultdict

@dataclass
class InsiderTransaction:
    """Data class for insider transactions."""
    transaction_id: str
    ticker: str
    insider_name: str
    insider_title: str
    transaction_type: str  # 'buy', 'sell', 'option_exercise', 'gift', 'other'
    shares: int
    price_per_share: float
    total_value: float
    transaction_date: datetime
    filing_date: datetime
    ownership_change: float  # percentage change in ownership
    remaining_shares: int
    source: str

@dataclass
class InsiderProfile:
    """Data class for insider profiles."""
    insider_id: str
    name: str
    title: str
    company: str
    total_transactions: int
    total_buys: int
    total_sells: int
    net_position: int
    current_ownership: int
    avg_buy_price: float
    avg_sell_price: float
    last_transaction_date: datetime
    confidence_score: float  # 0-100 scale

@dataclass
class InsiderTradingAnalysis:
    """Data class for insider trading analysis."""
    ticker: str
    total_transactions: int
    buy_transactions: int
    sell_transactions: int
    net_insider_activity: int  # positive = net buying, negative = net selling
    insider_sentiment_score: float  # 0-100 scale
    unusual_activity_score: float  # 0-100 scale
    top_insiders: List[InsiderProfile]
    recent_transactions: List[InsiderTransaction]
    pattern_analysis: Dict[str, Any]
    market_impact_prediction: float

class InsiderTradingService:
    """Service for analyzing insider trading patterns and market impact."""
    
    def __init__(self):
        self.base_url = "https://api.example.com/insider"  # Placeholder
        self.cache_duration = 3600  # 1 hour
        self.cache = {}
        self.last_update = None
        
        # Transaction type weights for sentiment calculation
        self.transaction_weights = {
            'buy': 1.0,
            'sell': -1.0,
            'option_exercise': 0.5,
            'gift': 0.0,
            'other': 0.0
        }
        
        # Insider title weights (higher positions have more weight)
        self.title_weights = {
            'CEO': 1.0,
            'CFO': 0.9,
            'CTO': 0.8,
            'COO': 0.8,
            'President': 0.7,
            'VP': 0.6,
            'Director': 0.5,
            'Officer': 0.4,
            'Other': 0.2
        }
        
        # Unusual activity thresholds
        self.unusual_thresholds = {
            'large_transaction': 100000,  # $100k
            'high_volume': 10000,  # 10k shares
            'frequent_trading': 5,  # 5+ transactions in 30 days
            'significant_ownership_change': 0.1  # 10% change
        }
    
    def get_insider_transactions(self, ticker: str, days_back: int = 90) -> List[InsiderTransaction]:
        """
        Get recent insider transactions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            List of insider transactions
        """
        try:
            # Simulated insider transactions data
            transactions = [
                InsiderTransaction(
                    transaction_id="IT_001",
                    ticker=ticker,
                    insider_name="John Smith",
                    insider_title="CEO",
                    transaction_type="buy",
                    shares=5000,
                    price_per_share=150.00,
                    total_value=750000.00,
                    transaction_date=datetime.now() - timedelta(days=2),
                    filing_date=datetime.now() - timedelta(days=1),
                    ownership_change=0.05,
                    remaining_shares=105000,
                    source="SEC Form 4"
                ),
                InsiderTransaction(
                    transaction_id="IT_002",
                    ticker=ticker,
                    insider_name="Jane Doe",
                    insider_title="CFO",
                    transaction_type="sell",
                    shares=2000,
                    price_per_share=148.50,
                    total_value=297000.00,
                    transaction_date=datetime.now() - timedelta(days=5),
                    filing_date=datetime.now() - timedelta(days=4),
                    ownership_change=-0.02,
                    remaining_shares=98000,
                    source="SEC Form 4"
                ),
                InsiderTransaction(
                    transaction_id="IT_003",
                    ticker=ticker,
                    insider_name="Mike Johnson",
                    insider_title="CTO",
                    transaction_type="option_exercise",
                    shares=10000,
                    price_per_share=100.00,
                    total_value=1000000.00,
                    transaction_date=datetime.now() - timedelta(days=10),
                    filing_date=datetime.now() - timedelta(days=9),
                    ownership_change=0.08,
                    remaining_shares=125000,
                    source="SEC Form 4"
                ),
                InsiderTransaction(
                    transaction_id="IT_004",
                    ticker=ticker,
                    insider_name="Sarah Wilson",
                    insider_title="VP Sales",
                    transaction_type="buy",
                    shares=1000,
                    price_per_share=152.00,
                    total_value=152000.00,
                    transaction_date=datetime.now() - timedelta(days=15),
                    filing_date=datetime.now() - timedelta(days=14),
                    ownership_change=0.01,
                    remaining_shares=11000,
                    source="SEC Form 4"
                ),
                InsiderTransaction(
                    transaction_id="IT_005",
                    ticker=ticker,
                    insider_name="David Brown",
                    insider_title="Director",
                    transaction_type="sell",
                    shares=5000,
                    price_per_share=147.00,
                    total_value=735000.00,
                    transaction_date=datetime.now() - timedelta(days=20),
                    filing_date=datetime.now() - timedelta(days=19),
                    ownership_change=-0.03,
                    remaining_shares=145000,
                    source="SEC Form 4"
                ),
                InsiderTransaction(
                    transaction_id="IT_006",
                    ticker=ticker,
                    insider_name="John Smith",
                    insider_title="CEO",
                    transaction_type="buy",
                    shares=3000,
                    price_per_share=151.50,
                    total_value=454500.00,
                    transaction_date=datetime.now() - timedelta(days=25),
                    filing_date=datetime.now() - timedelta(days=24),
                    ownership_change=0.03,
                    remaining_shares=100000,
                    source="SEC Form 4"
                )
            ]
            
            # Filter transactions by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_transactions = [t for t in transactions if t.transaction_date >= cutoff_date]
            
            return recent_transactions
            
        except Exception as e:
            print(f"Error fetching insider transactions: {e}")
            return []
    
    def build_insider_profiles(self, transactions: List[InsiderTransaction]) -> List[InsiderProfile]:
        """
        Build insider profiles from transactions.
        
        Args:
            transactions: List of insider transactions
            
        Returns:
            List of insider profiles
        """
        try:
            insider_data = defaultdict(lambda: {
                'transactions': [],
                'buys': [],
                'sells': [],
                'current_ownership': 0,
                'total_bought': 0,
                'total_sold': 0
            })
            
            # Group transactions by insider
            for transaction in transactions:
                insider_key = f"{transaction.insider_name}_{transaction.insider_title}"
                insider_data[insider_key]['transactions'].append(transaction)
                
                if transaction.transaction_type == 'buy':
                    insider_data[insider_key]['buys'].append(transaction)
                    insider_data[insider_key]['current_ownership'] += transaction.shares
                    insider_data[insider_key]['total_bought'] += transaction.shares
                elif transaction.transaction_type == 'sell':
                    insider_data[insider_key]['sells'].append(transaction)
                    insider_data[insider_key]['current_ownership'] -= transaction.shares
                    insider_data[insider_key]['total_sold'] += transaction.shares
            
            # Build profiles
            profiles = []
            for insider_key, data in insider_data.items():
                if not data['transactions']:
                    continue
                
                # Get most recent transaction for basic info
                latest_tx = max(data['transactions'], key=lambda x: x.transaction_date)
                
                # Calculate averages
                avg_buy_price = 0
                if data['buys']:
                    total_buy_value = sum(tx.total_value for tx in data['buys'])
                    total_buy_shares = sum(tx.shares for tx in data['buys'])
                    avg_buy_price = total_buy_value / total_buy_shares if total_buy_shares > 0 else 0
                
                avg_sell_price = 0
                if data['sells']:
                    total_sell_value = sum(tx.total_value for tx in data['sells'])
                    total_sell_shares = sum(tx.shares for tx in data['sells'])
                    avg_sell_price = total_sell_value / total_sell_shares if total_sell_shares > 0 else 0
                
                # Calculate confidence score based on transaction history
                confidence_score = min(100, len(data['transactions']) * 10)
                
                profile = InsiderProfile(
                    insider_id=insider_key,
                    name=latest_tx.insider_name,
                    title=latest_tx.insider_title,
                    company=latest_tx.ticker,
                    total_transactions=len(data['transactions']),
                    total_buys=len(data['buys']),
                    total_sells=len(data['sells']),
                    net_position=data['current_ownership'],
                    current_ownership=data['current_ownership'],
                    avg_buy_price=avg_buy_price,
                    avg_sell_price=avg_sell_price,
                    last_transaction_date=latest_tx.transaction_date,
                    confidence_score=confidence_score
                )
                
                profiles.append(profile)
            
            # Sort by confidence score (most active insiders first)
            profiles.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return profiles
            
        except Exception as e:
            print(f"Error building insider profiles: {e}")
            return []
    
    def calculate_insider_sentiment(self, transactions: List[InsiderTransaction]) -> float:
        """
        Calculate insider sentiment score.
        
        Args:
            transactions: List of insider transactions
            
        Returns:
            Sentiment score (0-100, higher = more bullish)
        """
        try:
            if not transactions:
                return 50.0  # Neutral
            
            weighted_scores = []
            total_weight = 0
            
            for transaction in transactions:
                # Base sentiment from transaction type
                base_sentiment = self.transaction_weights.get(transaction.transaction_type, 0)
                
                # Weight by insider title
                title_weight = self.title_weights.get(transaction.insider_title, 0.2)
                
                # Weight by transaction size (larger transactions have more impact)
                size_weight = min(1.0, transaction.total_value / 1000000)  # Normalize to $1M
                
                # Weight by recency (more recent transactions have more weight)
                days_ago = (datetime.now() - transaction.transaction_date).days
                recency_weight = max(0.1, 1.0 - (days_ago / 90))  # Decay over 90 days
                
                # Calculate weighted score
                weighted_score = base_sentiment * title_weight * size_weight * recency_weight
                weighted_scores.append(weighted_score)
                total_weight += title_weight * size_weight * recency_weight
            
            if total_weight == 0:
                return 50.0
            
            # Convert to 0-100 scale
            sentiment_score = (sum(weighted_scores) / total_weight + 1) * 50
            return max(0, min(100, sentiment_score))
            
        except Exception as e:
            print(f"Error calculating insider sentiment: {e}")
            return 50.0
    
    def detect_unusual_activity(self, transactions: List[InsiderTransaction]) -> float:
        """
        Detect unusual insider trading activity.
        
        Args:
            transactions: List of insider transactions
            
        Returns:
            Unusual activity score (0-100, higher = more unusual)
        """
        try:
            if not transactions:
                return 0.0
            
            unusual_indicators = []
            
            # Check for large transactions
            large_transactions = [t for t in transactions if t.total_value > self.unusual_thresholds['large_transaction']]
            if large_transactions:
                unusual_indicators.append(len(large_transactions) * 10)
            
            # Check for high volume transactions
            high_volume = [t for t in transactions if t.shares > self.unusual_thresholds['high_volume']]
            if high_volume:
                unusual_indicators.append(len(high_volume) * 8)
            
            # Check for frequent trading
            recent_transactions = [t for t in transactions if (datetime.now() - t.transaction_date).days <= 30]
            if len(recent_transactions) > self.unusual_thresholds['frequent_trading']:
                unusual_indicators.append((len(recent_transactions) - self.unusual_thresholds['frequent_trading']) * 5)
            
            # Check for significant ownership changes
            significant_changes = [t for t in transactions if abs(t.ownership_change) > self.unusual_thresholds['significant_ownership_change']]
            if significant_changes:
                unusual_indicators.append(len(significant_changes) * 12)
            
            # Check for CEO/CFO activity (higher weight)
            executive_transactions = [t for t in transactions if t.insider_title in ['CEO', 'CFO']]
            if executive_transactions:
                unusual_indicators.append(len(executive_transactions) * 6)
            
            # Calculate unusual activity score
            if unusual_indicators:
                unusual_score = min(100, sum(unusual_indicators))
            else:
                unusual_score = 0.0
            
            return unusual_score
            
        except Exception as e:
            print(f"Error detecting unusual activity: {e}")
            return 0.0
    
    def analyze_patterns(self, transactions: List[InsiderTransaction]) -> Dict[str, Any]:
        """
        Analyze insider trading patterns.
        
        Args:
            transactions: List of insider transactions
            
        Returns:
            Dictionary with pattern analysis
        """
        try:
            if not transactions:
                return {
                    'buy_sell_ratio': 0.0,
                    'avg_transaction_size': 0.0,
                    'transaction_frequency': 0.0,
                    'price_trend': 'neutral',
                    'volume_trend': 'neutral',
                    'executive_activity': 0.0
                }
            
            # Buy/sell ratio
            buys = [t for t in transactions if t.transaction_type == 'buy']
            sells = [t for t in transactions if t.transaction_type == 'sell']
            buy_sell_ratio = len(buys) / len(sells) if sells else float('inf')
            
            # Average transaction size
            avg_transaction_size = sum(t.total_value for t in transactions) / len(transactions)
            
            # Transaction frequency (transactions per month)
            date_range = (max(t.transaction_date for t in transactions) - min(t.transaction_date for t in transactions)).days
            transaction_frequency = len(transactions) / (date_range / 30) if date_range > 0 else 0
            
            # Price trend analysis
            sorted_transactions = sorted(transactions, key=lambda x: x.transaction_date)
            if len(sorted_transactions) >= 2:
                price_change = sorted_transactions[-1].price_per_share - sorted_transactions[0].price_per_share
                price_trend = 'increasing' if price_change > 0 else 'decreasing' if price_change < 0 else 'stable'
            else:
                price_trend = 'neutral'
            
            # Volume trend
            recent_volume = sum(t.shares for t in transactions if (datetime.now() - t.transaction_date).days <= 30)
            older_volume = sum(t.shares for t in transactions if (datetime.now() - t.transaction_date).days > 30)
            volume_trend = 'increasing' if recent_volume > older_volume else 'decreasing' if recent_volume < older_volume else 'stable'
            
            # Executive activity
            executive_transactions = [t for t in transactions if t.insider_title in ['CEO', 'CFO', 'CTO', 'COO']]
            executive_activity = len(executive_transactions) / len(transactions) if transactions else 0
            
            return {
                'buy_sell_ratio': buy_sell_ratio,
                'avg_transaction_size': avg_transaction_size,
                'transaction_frequency': transaction_frequency,
                'price_trend': price_trend,
                'volume_trend': volume_trend,
                'executive_activity': executive_activity
            }
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {
                'buy_sell_ratio': 0.0,
                'avg_transaction_size': 0.0,
                'transaction_frequency': 0.0,
                'price_trend': 'neutral',
                'volume_trend': 'neutral',
                'executive_activity': 0.0
            }
    
    def analyze_insider_trading(self, ticker: str) -> InsiderTradingAnalysis:
        """
        Analyze insider trading for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            InsiderTradingAnalysis
        """
        try:
            # Get insider transactions
            transactions = self.get_insider_transactions(ticker, days_back=90)
            
            if not transactions:
                return InsiderTradingAnalysis(
                    ticker=ticker,
                    total_transactions=0,
                    buy_transactions=0,
                    sell_transactions=0,
                    net_insider_activity=0,
                    insider_sentiment_score=50.0,
                    unusual_activity_score=0.0,
                    top_insiders=[],
                    recent_transactions=[],
                    pattern_analysis={},
                    market_impact_prediction=0.0
                )
            
            # Calculate basic metrics
            total_transactions = len(transactions)
            buy_transactions = len([t for t in transactions if t.transaction_type == 'buy'])
            sell_transactions = len([t for t in transactions if t.transaction_type == 'sell'])
            
            # Calculate net insider activity
            net_buys = sum(t.shares for t in transactions if t.transaction_type == 'buy')
            net_sells = sum(t.shares for t in transactions if t.transaction_type == 'sell')
            net_insider_activity = net_buys - net_sells
            
            # Calculate sentiment and unusual activity scores
            insider_sentiment = self.calculate_insider_sentiment(transactions)
            unusual_activity = self.detect_unusual_activity(transactions)
            
            # Build insider profiles
            insider_profiles = self.build_insider_profiles(transactions)
            top_insiders = insider_profiles[:5]  # Top 5 most active insiders
            
            # Get recent transactions (last 30 days)
            recent_transactions = [t for t in transactions if (datetime.now() - t.transaction_date).days <= 30]
            
            # Analyze patterns
            pattern_analysis = self.analyze_patterns(transactions)
            
            # Predict market impact
            market_impact_prediction = (insider_sentiment - 50) * 0.5 + unusual_activity * 0.3
            
            return InsiderTradingAnalysis(
                ticker=ticker,
                total_transactions=total_transactions,
                buy_transactions=buy_transactions,
                sell_transactions=sell_transactions,
                net_insider_activity=net_insider_activity,
                insider_sentiment_score=insider_sentiment,
                unusual_activity_score=unusual_activity,
                top_insiders=top_insiders,
                recent_transactions=recent_transactions,
                pattern_analysis=pattern_analysis,
                market_impact_prediction=market_impact_prediction
            )
            
        except Exception as e:
            print(f"Error analyzing insider trading: {e}")
            return InsiderTradingAnalysis(
                ticker=ticker,
                total_transactions=0,
                buy_transactions=0,
                sell_transactions=0,
                net_insider_activity=0,
                insider_sentiment_score=50.0,
                unusual_activity_score=0.0,
                top_insiders=[],
                recent_transactions=[],
                pattern_analysis={},
                market_impact_prediction=0.0
            )
    
    def save_insider_analysis(self, ticker: str, analysis: InsiderTradingAnalysis) -> bool:
        """
        Save insider trading analysis to file.
        
        Args:
            ticker: Stock ticker
            analysis: Insider trading analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            import os
            os.makedirs('data/insider_trading', exist_ok=True)
            
            # Prepare data for saving
            data = {
                'ticker': ticker,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_transactions': analysis.total_transactions,
                'buy_transactions': analysis.buy_transactions,
                'sell_transactions': analysis.sell_transactions,
                'net_insider_activity': analysis.net_insider_activity,
                'insider_sentiment_score': analysis.insider_sentiment_score,
                'unusual_activity_score': analysis.unusual_activity_score,
                'market_impact_prediction': analysis.market_impact_prediction,
                'pattern_analysis': analysis.pattern_analysis,
                'top_insiders': [
                    {
                        'name': insider.name,
                        'title': insider.title,
                        'total_transactions': insider.total_transactions,
                        'net_position': insider.net_position,
                        'confidence_score': insider.confidence_score
                    }
                    for insider in analysis.top_insiders
                ],
                'recent_transactions': [
                    {
                        'insider_name': tx.insider_name,
                        'insider_title': tx.insider_title,
                        'transaction_type': tx.transaction_type,
                        'shares': tx.shares,
                        'price_per_share': tx.price_per_share,
                        'total_value': tx.total_value,
                        'transaction_date': tx.transaction_date.strftime('%Y-%m-%d')
                    }
                    for tx in analysis.recent_transactions
                ]
            }
            
            # Save to JSON file
            filename = f"data/insider_trading/{ticker}_insider_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Insider trading analysis saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving insider analysis: {e}")
            return False
