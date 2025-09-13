#!/usr/bin/env python3
"""
Geopolitical Risk Assessment Service
====================================

This module provides comprehensive geopolitical risk assessment capabilities:
- Global political event tracking
- Trade war and tariff monitoring
- Regional conflict analysis
- Economic sanction tracking
- Political stability indices
- Market sentiment impact assessment
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
class GeopoliticalEvent:
    """Data class for geopolitical events."""
    event_id: str
    event_type: str
    title: str
    description: str
    country: str
    region: str
    severity: float  # 0-10 scale
    impact_score: float  # 0-100 scale
    affected_sectors: List[str]
    date: datetime
    source: str
    market_impact: Dict[str, float]  # sector -> impact score

@dataclass
class GeopoliticalRisk:
    """Data class for geopolitical risk assessment."""
    overall_risk_score: float  # 0-100 scale
    regional_risks: Dict[str, float]  # region -> risk score
    sector_risks: Dict[str, float]  # sector -> risk score
    event_count: int
    high_impact_events: List[GeopoliticalEvent]
    risk_factors: List[str]
    market_sentiment_impact: float
    volatility_forecast: float

class GeopoliticalRiskService:
    """Service for assessing geopolitical risks and their market impact."""
    
    def __init__(self):
        self.base_url = "https://api.example.com/geopolitical"  # Placeholder
        self.cache_duration = 3600  # 1 hour
        self.cache = {}
        self.last_update = None
        
        # Risk categories and weights
        self.risk_categories = {
            'political_instability': 0.25,
            'trade_conflicts': 0.20,
            'sanctions': 0.15,
            'regional_conflicts': 0.20,
            'regulatory_changes': 0.10,
            'elections': 0.10
        }
        
        # Sector sensitivity mapping
        self.sector_sensitivity = {
            'technology': ['trade_conflicts', 'sanctions', 'regulatory_changes'],
            'energy': ['regional_conflicts', 'sanctions', 'political_instability'],
            'finance': ['political_instability', 'regulatory_changes', 'sanctions'],
            'healthcare': ['regulatory_changes', 'trade_conflicts'],
            'consumer_goods': ['trade_conflicts', 'political_instability'],
            'industrials': ['trade_conflicts', 'regional_conflicts'],
            'materials': ['regional_conflicts', 'sanctions'],
            'utilities': ['political_instability', 'regulatory_changes']
        }
    
    def get_geopolitical_events(self, days_back: int = 30) -> List[GeopoliticalEvent]:
        """
        Get recent geopolitical events.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of geopolitical events
        """
        try:
            # Simulated geopolitical events data
            events = [
                GeopoliticalEvent(
                    event_id="GEO_001",
                    event_type="trade_conflicts",
                    title="US-China Trade Tensions Escalate",
                    description="New tariffs announced on technology imports",
                    country="United States",
                    region="Asia-Pacific",
                    severity=7.5,
                    impact_score=65.0,
                    affected_sectors=["technology", "consumer_goods"],
                    date=datetime.now() - timedelta(days=2),
                    source="Reuters",
                    market_impact={"technology": -3.2, "consumer_goods": -2.1}
                ),
                GeopoliticalEvent(
                    event_id="GEO_002",
                    event_type="political_instability",
                    title="Election Uncertainty in Major Market",
                    description="Close election results causing market volatility",
                    country="United Kingdom",
                    region="Europe",
                    severity=6.0,
                    impact_score=45.0,
                    affected_sectors=["finance", "utilities"],
                    date=datetime.now() - timedelta(days=5),
                    source="Bloomberg",
                    market_impact={"finance": -1.8, "utilities": -0.9}
                ),
                GeopoliticalEvent(
                    event_id="GEO_003",
                    event_type="regional_conflicts",
                    title="Energy Supply Disruption",
                    description="Pipeline issues affecting energy markets",
                    country="Russia",
                    region="Europe",
                    severity=8.0,
                    impact_score=75.0,
                    affected_sectors=["energy", "materials"],
                    date=datetime.now() - timedelta(days=1),
                    source="Financial Times",
                    market_impact={"energy": 4.5, "materials": 2.3}
                ),
                GeopoliticalEvent(
                    event_id="GEO_004",
                    event_type="sanctions",
                    title="New Economic Sanctions Announced",
                    description="Sanctions on key trading partner",
                    country="Iran",
                    region="Middle East",
                    severity=6.5,
                    impact_score=55.0,
                    affected_sectors=["energy", "finance"],
                    date=datetime.now() - timedelta(days=3),
                    source="Wall Street Journal",
                    market_impact={"energy": 2.1, "finance": -1.2}
                ),
                GeopoliticalEvent(
                    event_id="GEO_005",
                    event_type="regulatory_changes",
                    title="New Financial Regulations",
                    description="Stricter banking regulations proposed",
                    country="United States",
                    region="North America",
                    severity=5.5,
                    impact_score=40.0,
                    affected_sectors=["finance", "technology"],
                    date=datetime.now() - timedelta(days=4),
                    source="CNBC",
                    market_impact={"finance": -2.5, "technology": -1.1}
                )
            ]
            
            # Filter events by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_events = [event for event in events if event.date >= cutoff_date]
            
            return recent_events
            
        except Exception as e:
            print(f"Error fetching geopolitical events: {e}")
            return []
    
    def calculate_regional_risk(self, events: List[GeopoliticalEvent]) -> Dict[str, float]:
        """
        Calculate risk scores by region.
        
        Args:
            events: List of geopolitical events
            
        Returns:
            Dictionary mapping regions to risk scores
        """
        regional_risks = defaultdict(list)
        
        for event in events:
            regional_risks[event.region].append(event.impact_score)
        
        # Calculate weighted average risk for each region
        risk_scores = {}
        for region, scores in regional_risks.items():
            if scores:
                # Weight recent events more heavily
                weighted_scores = []
                for i, score in enumerate(scores):
                    weight = 1.0 / (i + 1)  # More recent events get higher weight
                    weighted_scores.append(score * weight)
                
                risk_scores[region] = sum(weighted_scores) / len(weighted_scores)
        
        return dict(risk_scores)
    
    def calculate_sector_risk(self, events: List[GeopoliticalEvent]) -> Dict[str, float]:
        """
        Calculate risk scores by sector.
        
        Args:
            events: List of geopolitical events
            
        Returns:
            Dictionary mapping sectors to risk scores
        """
        sector_risks = defaultdict(list)
        
        for event in events:
            for sector in event.affected_sectors:
                sector_risks[sector].append(event.impact_score)
        
        # Calculate average risk for each sector
        risk_scores = {}
        for sector, scores in sector_risks.items():
            if scores:
                risk_scores[sector] = sum(scores) / len(scores)
        
        return dict(risk_scores)
    
    def assess_geopolitical_risk(self, ticker: str, sector: str = None) -> GeopoliticalRisk:
        """
        Assess geopolitical risk for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            sector: Industry sector (optional)
            
        Returns:
            GeopoliticalRisk assessment
        """
        try:
            # Get recent events
            events = self.get_geopolitical_events(days_back=30)
            
            if not events:
                return GeopoliticalRisk(
                    overall_risk_score=25.0,
                    regional_risks={},
                    sector_risks={},
                    event_count=0,
                    high_impact_events=[],
                    risk_factors=["No recent geopolitical events"],
                    market_sentiment_impact=0.0,
                    volatility_forecast=0.0
                )
            
            # Calculate regional and sector risks
            regional_risks = self.calculate_regional_risk(events)
            sector_risks = self.calculate_sector_risk(events)
            
            # Calculate overall risk score
            overall_risk = sum(regional_risks.values()) / len(regional_risks) if regional_risks else 25.0
            
            # Identify high impact events (impact score > 50)
            high_impact_events = [event for event in events if event.impact_score > 50]
            
            # Identify risk factors
            risk_factors = []
            event_types = [event.event_type for event in events]
            for risk_type, weight in self.risk_categories.items():
                if event_types.count(risk_type) > 0:
                    risk_factors.append(f"{risk_type.replace('_', ' ').title()}")
            
            # Calculate market sentiment impact
            sentiment_impact = sum(event.impact_score for event in events) / len(events)
            
            # Calculate volatility forecast
            volatility_forecast = min(100, overall_risk * 1.2)  # Risk tends to increase volatility
            
            return GeopoliticalRisk(
                overall_risk_score=overall_risk,
                regional_risks=dict(regional_risks),
                sector_risks=dict(sector_risks),
                event_count=len(events),
                high_impact_events=high_impact_events,
                risk_factors=risk_factors,
                market_sentiment_impact=sentiment_impact,
                volatility_forecast=volatility_forecast
            )
            
        except Exception as e:
            print(f"Error assessing geopolitical risk: {e}")
            return GeopoliticalRisk(
                overall_risk_score=50.0,
                regional_risks={},
                sector_risks={},
                event_count=0,
                high_impact_events=[],
                risk_factors=[f"Assessment error: {e}"],
                market_sentiment_impact=0.0,
                volatility_forecast=0.0
            )
    
    def get_sector_specific_risk(self, sector: str) -> Dict[str, Any]:
        """
        Get sector-specific geopolitical risk analysis.
        
        Args:
            sector: Industry sector
            
        Returns:
            Dictionary with sector-specific risk analysis
        """
        try:
            events = self.get_geopolitical_events(days_back=30)
            
            # Filter events affecting this sector
            sector_events = [event for event in events if sector in event.affected_sectors]
            
            if not sector_events:
                return {
                    'sector': sector,
                    'risk_level': 'Low',
                    'risk_score': 20.0,
                    'key_concerns': [],
                    'market_impact': 0.0,
                    'recommendations': ['Monitor global events for sector impact']
                }
            
            # Calculate sector-specific metrics
            risk_score = sum(event.impact_score for event in sector_events) / len(sector_events)
            
            # Determine risk level
            if risk_score > 70:
                risk_level = 'Critical'
            elif risk_score > 50:
                risk_level = 'High'
            elif risk_score > 30:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Identify key concerns
            key_concerns = list(set([event.event_type for event in sector_events]))
            
            # Calculate market impact
            market_impact = sum(event.market_impact.get(sector, 0) for event in sector_events)
            
            # Generate recommendations
            recommendations = []
            if 'trade_conflicts' in key_concerns:
                recommendations.append('Monitor trade policy developments')
            if 'sanctions' in key_concerns:
                recommendations.append('Review supply chain dependencies')
            if 'political_instability' in key_concerns:
                recommendations.append('Assess regional exposure')
            
            return {
                'sector': sector,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'key_concerns': key_concerns,
                'market_impact': market_impact,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error getting sector-specific risk: {e}")
            return {
                'sector': sector,
                'risk_level': 'Unknown',
                'risk_score': 50.0,
                'key_concerns': [f'Analysis error: {e}'],
                'market_impact': 0.0,
                'recommendations': ['Unable to assess risk']
            }
    
    def save_geopolitical_analysis(self, ticker: str, risk_assessment: GeopoliticalRisk) -> bool:
        """
        Save geopolitical risk analysis to file.
        
        Args:
            ticker: Stock ticker
            risk_assessment: Geopolitical risk assessment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            import os
            os.makedirs('data/geopolitical', exist_ok=True)
            
            # Prepare data for saving
            data = {
                'ticker': ticker,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'overall_risk_score': risk_assessment.overall_risk_score,
                'regional_risks': risk_assessment.regional_risks,
                'sector_risks': risk_assessment.sector_risks,
                'event_count': risk_assessment.event_count,
                'risk_factors': risk_assessment.risk_factors,
                'market_sentiment_impact': risk_assessment.market_sentiment_impact,
                'volatility_forecast': risk_assessment.volatility_forecast,
                'high_impact_events': [
                    {
                        'event_id': event.event_id,
                        'title': event.title,
                        'severity': event.severity,
                        'impact_score': event.impact_score,
                        'date': event.date.strftime('%Y-%m-%d')
                    }
                    for event in risk_assessment.high_impact_events
                ]
            }
            
            # Save to JSON file
            filename = f"data/geopolitical/{ticker}_geopolitical_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Geopolitical risk analysis saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving geopolitical analysis: {e}")
            return False
