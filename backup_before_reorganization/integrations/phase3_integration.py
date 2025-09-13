#!/usr/bin/env python3
"""
Phase 3 Integration Module
==========================

This module integrates Phase 3 low priority features:
- Geopolitical risk assessment
- Enhanced corporate action tracking  
- Advanced insider trading analysis

Provides comprehensive analysis combining all three components.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import os

# Import Phase 3 services
try:
    from core.geopolitical_risk_service import GeopoliticalRiskService, GeopoliticalRisk
    from core.corporate_action_service import CorporateActionService, CorporateActionSummary
    from core.insider_trading_service import InsiderTradingService, InsiderTradingAnalysis
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Phase 3 services not available: {e}")
    PHASE3_AVAILABLE = False

@dataclass
class Phase3Analysis:
    """Data class for Phase 3 comprehensive analysis."""
    ticker: str
    geopolitical_risk: GeopoliticalRisk
    corporate_actions: CorporateActionSummary
    insider_trading: InsiderTradingAnalysis
    combined_risk_score: float  # 0-100 scale
    market_impact_score: float  # 0-100 scale
    confidence_level: float  # 0-100 scale
    key_insights: List[str]
    recommendations: List[str]
    analysis_timestamp: datetime

class Phase3Integration:
    """Integration class for Phase 3 features."""
    
    def __init__(self):
        if not PHASE3_AVAILABLE:
            raise ImportError("Phase 3 services are not available")
        
        self.geopolitical_service = GeopoliticalRiskService()
        self.corporate_service = CorporateActionService()
        self.insider_service = InsiderTradingService()
        
        # Weights for combining different analyses
        self.analysis_weights = {
            'geopolitical': 0.25,
            'corporate': 0.35,
            'insider': 0.40
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 30,
            'medium': 60,
            'high': 80
        }
    
    def run_phase3_analysis(self, ticker: str) -> Phase3Analysis:
        """
        Run comprehensive Phase 3 analysis.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Phase3Analysis object
        """
        try:
            print(f"🌍 Running Phase 3 Analysis for {ticker}...")
            
            # Run individual analyses
            geopolitical_risk = self.geopolitical_service.assess_geopolitical_risk(ticker)
            corporate_actions = self.corporate_service.analyze_corporate_actions(ticker)
            insider_trading = self.insider_service.analyze_insider_trading(ticker)
            
            # Calculate combined metrics
            combined_risk_score = self._calculate_combined_risk_score(
                geopolitical_risk, corporate_actions, insider_trading
            )
            
            market_impact_score = self._calculate_market_impact_score(
                geopolitical_risk, corporate_actions, insider_trading
            )
            
            confidence_level = self._calculate_confidence_level(
                geopolitical_risk, corporate_actions, insider_trading
            )
            
            # Generate insights and recommendations
            key_insights = self._generate_key_insights(
                geopolitical_risk, corporate_actions, insider_trading
            )
            
            recommendations = self._generate_recommendations(
                geopolitical_risk, corporate_actions, insider_trading
            )
            
            return Phase3Analysis(
                ticker=ticker,
                geopolitical_risk=geopolitical_risk,
                corporate_actions=corporate_actions,
                insider_trading=insider_trading,
                combined_risk_score=combined_risk_score,
                market_impact_score=market_impact_score,
                confidence_level=confidence_level,
                key_insights=key_insights,
                recommendations=recommendations,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error running Phase 3 analysis: {e}")
            raise
    
    def _calculate_combined_risk_score(self, geopolitical: GeopoliticalRisk, 
                                     corporate: CorporateActionSummary, 
                                     insider: InsiderTradingAnalysis) -> float:
        """Calculate combined risk score from all three analyses."""
        try:
            # Geopolitical risk contribution
            geo_risk = geopolitical.overall_risk_score * self.analysis_weights['geopolitical']
            
            # Corporate action risk (inverse of action score - higher score = lower risk)
            corp_risk = (100 - corporate.action_score) * self.analysis_weights['corporate']
            
            # Insider trading risk (based on unusual activity and sentiment)
            insider_risk = (insider.unusual_activity_score + (100 - insider.insider_sentiment_score)) / 2
            insider_risk *= self.analysis_weights['insider']
            
            combined_risk = geo_risk + corp_risk + insider_risk
            return min(100, max(0, combined_risk))
            
        except Exception as e:
            print(f"Error calculating combined risk score: {e}")
            return 50.0
    
    def _calculate_market_impact_score(self, geopolitical: GeopoliticalRisk, 
                                     corporate: CorporateActionSummary, 
                                     insider: InsiderTradingAnalysis) -> float:
        """Calculate market impact score from all three analyses."""
        try:
            # Geopolitical market impact
            geo_impact = geopolitical.market_sentiment_impact * self.analysis_weights['geopolitical']
            
            # Corporate action market impact
            corp_impact = corporate.market_sentiment * self.analysis_weights['corporate']
            
            # Insider trading market impact
            insider_impact = insider.market_impact_prediction * self.analysis_weights['insider']
            
            # Combine impacts (normalize to 0-100 scale)
            combined_impact = (geo_impact + corp_impact + insider_impact) / 3
            return min(100, max(0, combined_impact))
            
        except Exception as e:
            print(f"Error calculating market impact score: {e}")
            return 50.0
    
    def _calculate_confidence_level(self, geopolitical: GeopoliticalRisk, 
                                  corporate: CorporateActionSummary, 
                                  insider: InsiderTradingAnalysis) -> float:
        """Calculate confidence level based on data quality and consistency."""
        try:
            confidence_factors = []
            
            # Geopolitical confidence (based on event count and data quality)
            geo_confidence = min(100, geopolitical.event_count * 10)
            confidence_factors.append(geo_confidence * self.analysis_weights['geopolitical'])
            
            # Corporate action confidence (based on transaction count)
            corp_confidence = min(100, corporate.total_actions * 15)
            confidence_factors.append(corp_confidence * self.analysis_weights['corporate'])
            
            # Insider trading confidence (based on transaction count and profile quality)
            insider_confidence = min(100, insider.total_transactions * 12)
            if insider.top_insiders:
                avg_confidence = sum(insider.confidence_score for insider in insider.top_insiders) / len(insider.top_insiders)
                insider_confidence = (insider_confidence + avg_confidence) / 2
            confidence_factors.append(insider_confidence * self.analysis_weights['insider'])
            
            return min(100, sum(confidence_factors))
            
        except Exception as e:
            print(f"Error calculating confidence level: {e}")
            return 50.0
    
    def _generate_key_insights(self, geopolitical: GeopoliticalRisk, 
                             corporate: CorporateActionSummary, 
                             insider: InsiderTradingAnalysis) -> List[str]:
        """Generate key insights from all three analyses."""
        insights = []
        
        try:
            # Geopolitical insights
            if geopolitical.overall_risk_score > self.risk_thresholds['medium']:
                insights.append(f"High geopolitical risk ({geopolitical.overall_risk_score:.1f}/100) - Monitor global events")
            
            if geopolitical.high_impact_events:
                insights.append(f"{len(geopolitical.high_impact_events)} high-impact geopolitical events detected")
            
            # Corporate action insights
            if corporate.dividend_yield > 3.0:
                insights.append(f"Attractive dividend yield: {corporate.dividend_yield:.1f}%")
            
            if corporate.buyback_amount > 1000000000:  # $1B
                insights.append(f"Significant buyback program: ${corporate.buyback_amount/1000000000:.1f}B")
            
            if corporate.pending_actions > 0:
                insights.append(f"{corporate.pending_actions} pending corporate actions")
            
            # Insider trading insights
            if insider.insider_sentiment_score > 70:
                insights.append("Strong insider buying sentiment")
            elif insider.insider_sentiment_score < 30:
                insights.append("Weak insider sentiment - net selling activity")
            
            if insider.unusual_activity_score > 50:
                insights.append("Unusual insider trading activity detected")
            
            if insider.net_insider_activity > 0:
                insights.append(f"Net insider buying: {insider.net_insider_activity:,} shares")
            elif insider.net_insider_activity < 0:
                insights.append(f"Net insider selling: {abs(insider.net_insider_activity):,} shares")
            
            # Combined insights
            if len(insights) == 0:
                insights.append("No significant insights detected - normal market conditions")
            
            return insights
            
        except Exception as e:
            print(f"Error generating insights: {e}")
            return ["Unable to generate insights due to analysis error"]
    
    def _generate_recommendations(self, geopolitical: GeopoliticalRisk, 
                                corporate: CorporateActionSummary, 
                                insider: InsiderTradingAnalysis) -> List[str]:
        """Generate recommendations based on all three analyses."""
        recommendations = []
        
        try:
            # Risk-based recommendations
            if geopolitical.overall_risk_score > self.risk_thresholds['high']:
                recommendations.append("Consider reducing exposure due to high geopolitical risk")
            
            if insider.unusual_activity_score > 70:
                recommendations.append("Monitor for potential insider trading signals")
            
            # Opportunity-based recommendations
            if insider.insider_sentiment_score > 75 and corporate.action_score > 70:
                recommendations.append("Strong insider confidence with positive corporate actions")
            
            if corporate.dividend_yield > 4.0 and corporate.payout_ratio < 0.6:
                recommendations.append("Attractive dividend with sustainable payout ratio")
            
            # Sector-specific recommendations
            if geopolitical.sector_risks:
                high_risk_sectors = [sector for sector, risk in geopolitical.sector_risks.items() if risk > 60]
                if high_risk_sectors:
                    recommendations.append(f"Monitor sector risks: {', '.join(high_risk_sectors)}")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Maintain current position - balanced risk/reward profile")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def save_phase3_analysis(self, ticker: str, analysis: Phase3Analysis) -> bool:
        """
        Save Phase 3 analysis to file.
        
        Args:
            ticker: Stock ticker
            analysis: Phase 3 analysis
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data/phase3', exist_ok=True)
            
            # Prepare data for saving
            data = {
                'ticker': ticker,
                'analysis_date': analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'combined_risk_score': analysis.combined_risk_score,
                'market_impact_score': analysis.market_impact_score,
                'confidence_level': analysis.confidence_level,
                'key_insights': analysis.key_insights,
                'recommendations': analysis.recommendations,
                'geopolitical_risk': {
                    'overall_risk_score': analysis.geopolitical_risk.overall_risk_score,
                    'event_count': analysis.geopolitical_risk.event_count,
                    'risk_factors': analysis.geopolitical_risk.risk_factors,
                    'market_sentiment_impact': analysis.geopolitical_risk.market_sentiment_impact
                },
                'corporate_actions': {
                    'total_actions': analysis.corporate_actions.total_actions,
                    'dividend_yield': analysis.corporate_actions.dividend_yield,
                    'buyback_amount': analysis.corporate_actions.buyback_amount,
                    'action_score': analysis.corporate_actions.action_score
                },
                'insider_trading': {
                    'total_transactions': analysis.insider_trading.total_transactions,
                    'insider_sentiment_score': analysis.insider_trading.insider_sentiment_score,
                    'unusual_activity_score': analysis.insider_trading.unusual_activity_score,
                    'net_insider_activity': analysis.insider_trading.net_insider_activity
                }
            }
            
            # Save to JSON file
            filename = f"data/phase3/{ticker}_phase3_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Phase 3 analysis saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving Phase 3 analysis: {e}")
            return False
    
    def get_analysis_summary(self, analysis: Phase3Analysis) -> Dict[str, Any]:
        """
        Get a summary of the Phase 3 analysis.
        
        Args:
            analysis: Phase 3 analysis
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            return {
                'ticker': analysis.ticker,
                'analysis_date': analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'combined_risk_score': analysis.combined_risk_score,
                'market_impact_score': analysis.market_impact_score,
                'confidence_level': analysis.confidence_level,
                'risk_level': self._get_risk_level(analysis.combined_risk_score),
                'key_insights_count': len(analysis.key_insights),
                'recommendations_count': len(analysis.recommendations),
                'geopolitical_events': analysis.geopolitical_risk.event_count,
                'corporate_actions': analysis.corporate_actions.total_actions,
                'insider_transactions': analysis.insider_trading.total_transactions
            }
            
        except Exception as e:
            print(f"Error getting analysis summary: {e}")
            return {}
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description based on score."""
        if risk_score < self.risk_thresholds['low']:
            return "Low"
        elif risk_score < self.risk_thresholds['medium']:
            return "Medium"
        elif risk_score < self.risk_thresholds['high']:
            return "High"
        else:
            return "Critical"
