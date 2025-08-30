#!/usr/bin/env python3
"""
Enhanced Corporate Action Tracking Service
==========================================

This module provides comprehensive corporate action tracking capabilities:
- Dividend announcements and payments
- Stock splits and reverse splits
- Mergers and acquisitions
- Share buybacks and offerings
- Earnings announcements
- Board changes and executive appointments
- Regulatory filings and compliance
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
class CorporateAction:
    """Data class for corporate actions."""
    action_id: str
    action_type: str
    ticker: str
    company_name: str
    title: str
    description: str
    announcement_date: datetime
    effective_date: Optional[datetime]
    status: str  # 'announced', 'pending', 'completed', 'cancelled'
    impact_score: float  # 0-100 scale
    market_impact: Dict[str, float]  # metric -> impact value
    details: Dict[str, Any]

@dataclass
class DividendAction:
    """Data class for dividend-specific actions."""
    dividend_id: str
    ticker: str
    dividend_type: str  # 'regular', 'special', 'stock'
    amount: float
    currency: str
    ex_date: datetime
    record_date: datetime
    payment_date: datetime
    yield_percentage: float
    payout_ratio: float
    frequency: str  # 'quarterly', 'monthly', 'annual', 'special'

@dataclass
class CorporateActionSummary:
    """Data class for corporate action summary."""
    ticker: str
    total_actions: int
    pending_actions: int
    completed_actions: int
    dividend_yield: float
    payout_ratio: float
    buyback_amount: float
    recent_earnings: Dict[str, Any]
    upcoming_events: List[CorporateAction]
    market_sentiment: float
    action_score: float

class CorporateActionService:
    """Service for tracking and analyzing corporate actions."""
    
    def __init__(self):
        self.base_url = "https://api.example.com/corporate"  # Placeholder
        self.cache_duration = 1800  # 30 minutes
        self.cache = {}
        self.last_update = None
        
        # Action type weights for impact calculation
        self.action_weights = {
            'dividend': 0.25,
            'stock_split': 0.15,
            'merger': 0.30,
            'acquisition': 0.25,
            'buyback': 0.20,
            'earnings': 0.35,
            'board_change': 0.10,
            'regulatory': 0.15
        }
        
        # Market impact factors
        self.impact_factors = {
            'price_impact': 0.30,
            'volume_impact': 0.20,
            'volatility_impact': 0.25,
            'sentiment_impact': 0.25
        }
    
    def get_corporate_actions(self, ticker: str, days_back: int = 90) -> List[CorporateAction]:
        """
        Get recent corporate actions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            
        Returns:
            List of corporate actions
        """
        try:
            # Simulated corporate actions data
            actions = [
                CorporateAction(
                    action_id="CA_001",
                    action_type="dividend",
                    ticker=ticker,
                    company_name=f"{ticker} Corporation",
                    title="Quarterly Dividend Announcement",
                    description="Board declares quarterly dividend of $0.50 per share",
                    announcement_date=datetime.now() - timedelta(days=5),
                    effective_date=datetime.now() + timedelta(days=30),
                    status="announced",
                    impact_score=45.0,
                    market_impact={"price": 1.2, "volume": 0.8, "volatility": -0.5},
                    details={
                        "amount": 0.50,
                        "currency": "USD",
                        "ex_date": (datetime.now() + timedelta(days=25)).strftime('%Y-%m-%d'),
                        "payment_date": (datetime.now() + timedelta(days=45)).strftime('%Y-%m-%d'),
                        "yield": 2.1
                    }
                ),
                CorporateAction(
                    action_id="CA_002",
                    action_type="earnings",
                    ticker=ticker,
                    company_name=f"{ticker} Corporation",
                    title="Q3 2025 Earnings Announcement",
                    description="Company reports strong quarterly results",
                    announcement_date=datetime.now() - timedelta(days=15),
                    effective_date=datetime.now() - timedelta(days=15),
                    status="completed",
                    impact_score=75.0,
                    market_impact={"price": 3.5, "volume": 2.1, "volatility": 1.8},
                    details={
                        "eps": 1.25,
                        "revenue": 8500000000,
                        "guidance": "positive",
                        "beat_estimate": True
                    }
                ),
                CorporateAction(
                    action_id="CA_003",
                    action_type="buyback",
                    ticker=ticker,
                    company_name=f"{ticker} Corporation",
                    title="Share Buyback Program",
                    description="Board approves $2 billion share repurchase program",
                    announcement_date=datetime.now() - timedelta(days=25),
                    effective_date=datetime.now() + timedelta(days=60),
                    status="announced",
                    impact_score=60.0,
                    market_impact={"price": 2.1, "volume": 1.5, "volatility": 0.8},
                    details={
                        "amount": 2000000000,
                        "duration": "12 months",
                        "method": "open market"
                    }
                ),
                CorporateAction(
                    action_id="CA_004",
                    action_type="board_change",
                    ticker=ticker,
                    company_name=f"{ticker} Corporation",
                    title="New CEO Appointment",
                    description="Company appoints new Chief Executive Officer",
                    announcement_date=datetime.now() - timedelta(days=40),
                    effective_date=datetime.now() + timedelta(days=30),
                    status="announced",
                    impact_score=35.0,
                    market_impact={"price": 1.8, "volume": 1.2, "volatility": 1.5},
                    details={
                        "position": "CEO",
                        "name": "John Smith",
                        "background": "Former CFO of major competitor"
                    }
                ),
                CorporateAction(
                    action_id="CA_005",
                    action_type="regulatory",
                    ticker=ticker,
                    company_name=f"{ticker} Corporation",
                    title="SEC Filing - 10-K Annual Report",
                    description="Company files annual report with SEC",
                    announcement_date=datetime.now() - timedelta(days=50),
                    effective_date=datetime.now() - timedelta(days=50),
                    status="completed",
                    impact_score=25.0,
                    market_impact={"price": 0.5, "volume": 0.3, "volatility": 0.2},
                    details={
                        "filing_type": "10-K",
                        "fiscal_year": "2024",
                        "compliance_status": "compliant"
                    }
                )
            ]
            
            # Filter actions by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_actions = [action for action in actions if action.announcement_date >= cutoff_date]
            
            return recent_actions
            
        except Exception as e:
            print(f"Error fetching corporate actions: {e}")
            return []
    
    def get_dividend_history(self, ticker: str, years_back: int = 3) -> List[DividendAction]:
        """
        Get dividend history for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            years_back: Number of years to look back
            
        Returns:
            List of dividend actions
        """
        try:
            # Simulated dividend history
            dividends = [
                DividendAction(
                    dividend_id="DIV_001",
                    ticker=ticker,
                    dividend_type="regular",
                    amount=0.50,
                    currency="USD",
                    ex_date=datetime.now() - timedelta(days=30),
                    record_date=datetime.now() - timedelta(days=32),
                    payment_date=datetime.now() - timedelta(days=15),
                    yield_percentage=2.1,
                    payout_ratio=0.35,
                    frequency="quarterly"
                ),
                DividendAction(
                    dividend_id="DIV_002",
                    ticker=ticker,
                    dividend_type="regular",
                    amount=0.48,
                    currency="USD",
                    ex_date=datetime.now() - timedelta(days=120),
                    record_date=datetime.now() - timedelta(days=122),
                    payment_date=datetime.now() - timedelta(days=105),
                    yield_percentage=2.0,
                    payout_ratio=0.33,
                    frequency="quarterly"
                ),
                DividendAction(
                    dividend_id="DIV_003",
                    ticker=ticker,
                    dividend_type="regular",
                    amount=0.45,
                    currency="USD",
                    ex_date=datetime.now() - timedelta(days=210),
                    record_date=datetime.now() - timedelta(days=212),
                    payment_date=datetime.now() - timedelta(days=195),
                    yield_percentage=1.9,
                    payout_ratio=0.31,
                    frequency="quarterly"
                ),
                DividendAction(
                    dividend_id="DIV_004",
                    ticker=ticker,
                    dividend_type="special",
                    amount=1.00,
                    currency="USD",
                    ex_date=datetime.now() - timedelta(days=300),
                    record_date=datetime.now() - timedelta(days=302),
                    payment_date=datetime.now() - timedelta(days=285),
                    yield_percentage=4.2,
                    payout_ratio=0.65,
                    frequency="special"
                )
            ]
            
            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=years_back * 365)
            recent_dividends = [div for div in dividends if div.ex_date >= cutoff_date]
            
            return recent_dividends
            
        except Exception as e:
            print(f"Error fetching dividend history: {e}")
            return []
    
    def calculate_dividend_metrics(self, dividends: List[DividendAction]) -> Dict[str, float]:
        """
        Calculate dividend-related metrics.
        
        Args:
            dividends: List of dividend actions
            
        Returns:
            Dictionary with dividend metrics
        """
        if not dividends:
            return {
                'current_yield': 0.0,
                'average_yield': 0.0,
                'payout_ratio': 0.0,
                'growth_rate': 0.0,
                'consistency_score': 0.0
            }
        
        try:
            # Current yield (most recent dividend)
            current_yield = dividends[0].yield_percentage if dividends else 0.0
            
            # Average yield
            average_yield = sum(div.yield_percentage for div in dividends) / len(dividends)
            
            # Average payout ratio
            payout_ratios = [div.payout_ratio for div in dividends if div.payout_ratio > 0]
            avg_payout_ratio = sum(payout_ratios) / len(payout_ratios) if payout_ratios else 0.0
            
            # Growth rate (if multiple dividends)
            if len(dividends) >= 2:
                amounts = [div.amount for div in dividends]
                growth_rate = ((amounts[0] - amounts[-1]) / amounts[-1]) * 100
            else:
                growth_rate = 0.0
            
            # Consistency score (regular vs special dividends)
            regular_dividends = [div for div in dividends if div.dividend_type == 'regular']
            consistency_score = (len(regular_dividends) / len(dividends)) * 100
            
            return {
                'current_yield': current_yield,
                'average_yield': average_yield,
                'payout_ratio': avg_payout_ratio,
                'growth_rate': growth_rate,
                'consistency_score': consistency_score
            }
            
        except Exception as e:
            print(f"Error calculating dividend metrics: {e}")
            return {
                'current_yield': 0.0,
                'average_yield': 0.0,
                'payout_ratio': 0.0,
                'growth_rate': 0.0,
                'consistency_score': 0.0
            }
    
    def analyze_corporate_actions(self, ticker: str) -> CorporateActionSummary:
        """
        Analyze corporate actions for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CorporateActionSummary
        """
        try:
            # Get corporate actions and dividend history
            actions = self.get_corporate_actions(ticker, days_back=90)
            dividends = self.get_dividend_history(ticker, years_back=3)
            
            # Calculate metrics
            total_actions = len(actions)
            pending_actions = len([a for a in actions if a.status in ['announced', 'pending']])
            completed_actions = len([a for a in actions if a.status == 'completed'])
            
            # Dividend metrics
            dividend_metrics = self.calculate_dividend_metrics(dividends)
            current_yield = dividend_metrics['current_yield']
            payout_ratio = dividend_metrics['payout_ratio']
            
            # Buyback amount
            buyback_actions = [a for a in actions if a.action_type == 'buyback']
            buyback_amount = sum(a.details.get('amount', 0) for a in buyback_actions)
            
            # Recent earnings
            earnings_actions = [a for a in actions if a.action_type == 'earnings']
            recent_earnings = earnings_actions[0].details if earnings_actions else {}
            
            # Upcoming events
            upcoming_events = [a for a in actions if a.status in ['announced', 'pending']]
            
            # Market sentiment (based on action impacts)
            if actions:
                sentiment_scores = [a.impact_score for a in actions]
                market_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            else:
                market_sentiment = 50.0  # Neutral
            
            # Overall action score
            if actions:
                weighted_scores = []
                for action in actions:
                    weight = self.action_weights.get(action.action_type, 0.1)
                    weighted_scores.append(action.impact_score * weight)
                action_score = sum(weighted_scores) / len(weighted_scores)
            else:
                action_score = 25.0
            
            return CorporateActionSummary(
                ticker=ticker,
                total_actions=total_actions,
                pending_actions=pending_actions,
                completed_actions=completed_actions,
                dividend_yield=current_yield,
                payout_ratio=payout_ratio,
                buyback_amount=buyback_amount,
                recent_earnings=recent_earnings,
                upcoming_events=upcoming_events,
                market_sentiment=market_sentiment,
                action_score=action_score
            )
            
        except Exception as e:
            print(f"Error analyzing corporate actions: {e}")
            return CorporateActionSummary(
                ticker=ticker,
                total_actions=0,
                pending_actions=0,
                completed_actions=0,
                dividend_yield=0.0,
                payout_ratio=0.0,
                buyback_amount=0.0,
                recent_earnings={},
                upcoming_events=[],
                market_sentiment=50.0,
                action_score=25.0
            )
    
    def get_upcoming_events(self, ticker: str, days_ahead: int = 30) -> List[CorporateAction]:
        """
        Get upcoming corporate events.
        
        Args:
            ticker: Stock ticker symbol
            days_ahead: Number of days to look ahead
            
        Returns:
            List of upcoming corporate actions
        """
        try:
            actions = self.get_corporate_actions(ticker, days_back=365)
            
            # Filter for upcoming events
            cutoff_date = datetime.now() + timedelta(days=days_ahead)
            upcoming = [
                action for action in actions 
                if action.effective_date and action.effective_date <= cutoff_date
                and action.status in ['announced', 'pending']
            ]
            
            # Sort by effective date
            upcoming.sort(key=lambda x: x.effective_date or datetime.max)
            
            return upcoming
            
        except Exception as e:
            print(f"Error getting upcoming events: {e}")
            return []
    
    def save_corporate_analysis(self, ticker: str, summary: CorporateActionSummary) -> bool:
        """
        Save corporate action analysis to file.
        
        Args:
            ticker: Stock ticker
            summary: Corporate action summary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            import os
            os.makedirs('data/corporate_actions', exist_ok=True)
            
            # Prepare data for saving
            data = {
                'ticker': ticker,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_actions': summary.total_actions,
                'pending_actions': summary.pending_actions,
                'completed_actions': summary.completed_actions,
                'dividend_yield': summary.dividend_yield,
                'payout_ratio': summary.payout_ratio,
                'buyback_amount': summary.buyback_amount,
                'recent_earnings': summary.recent_earnings,
                'market_sentiment': summary.market_sentiment,
                'action_score': summary.action_score,
                'upcoming_events': [
                    {
                        'action_id': event.action_id,
                        'action_type': event.action_type,
                        'title': event.title,
                        'effective_date': event.effective_date.strftime('%Y-%m-%d') if event.effective_date else None,
                        'impact_score': event.impact_score
                    }
                    for event in summary.upcoming_events
                ]
            }
            
            # Save to JSON file
            filename = f"data/corporate_actions/{ticker}_corporate_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Corporate action analysis saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving corporate analysis: {e}")
            return False
