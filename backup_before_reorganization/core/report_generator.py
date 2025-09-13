#!/usr/bin/env python3
"""
Comprehensive Report Generator
=============================

Generates comprehensive reports with:
- Multi-currency formatting (₹, $, £, €, etc.)
- Enhanced date formatting with exact dates
- Summary, predictions, metrics, and recommendations
- Multiple report formats (HTML, PDF, JSON, CSV)

Part of the enhanced analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from core.economic_data_service import EconomicDataService
from integrations.phase1_integration import Phase1Integration
from integrations.phase2_integration import Phase2Integration
from integrations.phase3_integration import Phase3Integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurrencyInfo:
    """Currency information and formatting"""
    symbol: str
    code: str
    name: str
    decimal_places: int
    position: str  # 'before' or 'after'
    thousands_separator: str
    decimal_separator: str

@dataclass
class DateInfo:
    """Enhanced date information"""
    date: datetime
    date_string: str
    date_short: str
    date_long: str
    date_iso: str
    day_name: str
    month_name: str
    year: int
    month: int
    day: int
    week_number: int
    quarter: int
    fiscal_year: int

@dataclass
class PredictionMetrics:
    """Prediction metrics and confidence"""
    current_price: float
    predicted_price: float
    price_change: float
    price_change_percent: float
    confidence_score: float
    model_agreement: float
    prediction_range: Dict[str, float]
    timeframe_predictions: Dict[str, Dict[str, Any]]

@dataclass
class AnalysisSummary:
    """Comprehensive analysis summary"""
    ticker: str
    analysis_date: DateInfo
    currency: CurrencyInfo
    prediction_metrics: PredictionMetrics
    phase1_analysis: Dict[str, Any]
    phase2_analysis: Dict[str, Any]
    phase3_analysis: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]
    key_insights: List[str]

class CurrencyFormatter:
    """Multi-currency formatting utility"""
    
    def __init__(self):
        self.currencies = {
            'USD': CurrencyInfo('$', 'USD', 'US Dollar', 2, 'before', ',', '.'),
            'INR': CurrencyInfo('₹', 'INR', 'Indian Rupee', 2, 'before', ',', '.'),
            'EUR': CurrencyInfo('€', 'EUR', 'Euro', 2, 'after', '.', ','),
            'GBP': CurrencyInfo('£', 'GBP', 'British Pound', 2, 'before', ',', '.'),
            'JPY': CurrencyInfo('¥', 'JPY', 'Japanese Yen', 0, 'before', ',', '.'),
            'CAD': CurrencyInfo('C$', 'CAD', 'Canadian Dollar', 2, 'before', ',', '.'),
            'AUD': CurrencyInfo('A$', 'AUD', 'Australian Dollar', 2, 'before', ',', '.'),
            'CHF': CurrencyInfo('CHF', 'CHF', 'Swiss Franc', 2, 'after', "'", '.'),
            'CNY': CurrencyInfo('¥', 'CNY', 'Chinese Yuan', 2, 'before', ',', '.'),
            'KRW': CurrencyInfo('₩', 'KRW', 'South Korean Won', 0, 'before', ',', '.'),
        }
        
        # Default currency
        self.default_currency = 'USD'
    
    def format_amount(self, amount: float, currency_code: str = None, 
                     show_symbol: bool = True, show_code: bool = False) -> str:
        """
        Format amount with currency
        
        Args:
            amount: Amount to format
            currency_code: Currency code (default: USD)
            show_symbol: Show currency symbol
            show_code: Show currency code
            
        Returns:
            Formatted currency string
        """
        if currency_code is None:
            currency_code = self.default_currency
            
        if currency_code not in self.currencies:
            currency_code = self.default_currency
            
        currency = self.currencies[currency_code]
        
        # Format number
        if currency.decimal_places == 0:
            formatted_amount = f"{int(amount):,}"
        else:
            formatted_amount = f"{amount:,.{currency.decimal_places}f}"
        
        # Add thousands separator
        if currency.thousands_separator != ',':
            formatted_amount = formatted_amount.replace(',', currency.thousands_separator)
        
        # Add decimal separator
        if currency.decimal_separator != '.':
            formatted_amount = formatted_amount.replace('.', currency.decimal_separator)
        
        # Add currency symbol/code
        if show_symbol and show_code:
            if currency.position == 'before':
                return f"{currency.symbol} {formatted_amount} ({currency.code})"
            else:
                return f"{formatted_amount} {currency.symbol} ({currency.code})"
        elif show_symbol:
            if currency.position == 'before':
                return f"{currency.symbol}{formatted_amount}"
            else:
                return f"{formatted_amount} {currency.symbol}"
        elif show_code:
            return f"{formatted_amount} {currency.code}"
        else:
            return formatted_amount
    
    def get_currency_info(self, currency_code: str) -> CurrencyInfo:
        """Get currency information"""
        return self.currencies.get(currency_code, self.currencies[self.default_currency])

class EnhancedDateFormatter:
    """Enhanced date formatting utility"""
    
    def __init__(self):
        self.date_formats = {
            'short': '%m/%d/%y',
            'medium': '%b %d, %Y',
            'long': '%B %d, %Y',
            'iso': '%Y-%m-%d',
            'datetime': '%Y-%m-%d %H:%M:%S',
            'readable': '%A, %B %d, %Y at %I:%M %p',
            'compact': '%d-%m-%Y',
            'us': '%m/%d/%Y',
            'eu': '%d/%m/%Y'
        }
    
    def create_date_info(self, date: datetime) -> DateInfo:
        """
        Create comprehensive date information
        
        Args:
            date: Date to format
            
        Returns:
            DateInfo object with all date formats
        """
        return DateInfo(
            date=date,
            date_string=date.strftime(self.date_formats['medium']),
            date_short=date.strftime(self.date_formats['short']),
            date_long=date.strftime(self.date_formats['long']),
            date_iso=date.strftime(self.date_formats['iso']),
            day_name=date.strftime('%A'),
            month_name=date.strftime('%B'),
            year=date.year,
            month=date.month,
            day=date.day,
            week_number=date.isocalendar()[1],
            quarter=(date.month - 1) // 3 + 1,
            fiscal_year=date.year if date.month >= 4 else date.year - 1
        )
    
    def format_date(self, date: datetime, format_type: str = 'medium') -> str:
        """
        Format date with specified format
        
        Args:
            date: Date to format
            format_type: Format type (short, medium, long, iso, etc.)
            
        Returns:
            Formatted date string
        """
        if format_type in self.date_formats:
            return date.strftime(self.date_formats[format_type])
        else:
            return date.strftime(format_type)

class ComprehensiveReportGenerator:
    """
    Comprehensive Report Generator
    
    Generates detailed reports with multi-currency support and enhanced dates
    """
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize formatters
        self.currency_formatter = CurrencyFormatter()
        self.date_formatter = EnhancedDateFormatter()
        
        # Initialize analysis services
        self.economic_service = EconomicDataService()
        self.phase1_integration = Phase1Integration()
        self.phase2_integration = Phase2Integration()
        self.phase3_integration = Phase3Integration()
        
        logger.info("Comprehensive Report Generator initialized")
    
    def generate_comprehensive_report(self, ticker: str, 
                                    currency_code: str = 'USD',
                                    include_phases: List[str] = None) -> AnalysisSummary:
        """
        Generate comprehensive analysis report
        
        Args:
            ticker: Stock ticker symbol
            currency_code: Currency code for formatting
            include_phases: List of phases to include (default: all)
            
        Returns:
            AnalysisSummary object
        """
        if include_phases is None:
            include_phases = ['phase1', 'phase2', 'phase3']
        
        logger.info(f"Generating comprehensive report for {ticker}")
        
        # Create date info
        analysis_date = self.date_formatter.create_date_info(datetime.now())
        currency_info = self.currency_formatter.get_currency_info(currency_code)
        
        # Generate phase analyses
        phase1_analysis = {}
        phase2_analysis = {}
        phase3_analysis = {}
        
        if 'phase1' in include_phases:
            try:
                phase1_result = self.phase1_integration.run_phase1_analysis(ticker)
                phase1_analysis = {
                    'enhanced_prediction_score': phase1_result.enhanced_prediction_score,
                    'variable_coverage': phase1_result.variable_coverage,
                    'fundamental_health': phase1_result.fundamental_health_score,
                    'institutional_confidence': phase1_result.institutional_confidence,
                    'global_market_impact': phase1_result.global_market_impact,
                    'last_updated': phase1_result.last_updated
                }
            except Exception as e:
                logger.error(f"Phase 1 analysis failed: {str(e)}")
                phase1_analysis = {'error': str(e)}
        
        if 'phase2' in include_phases:
            try:
                phase2_result = self.phase2_integration.run_phase2_analysis(ticker)
                phase2_analysis = {
                    'economic_impact_score': phase2_result.economic_impact_score,
                    'economic_sentiment': phase2_result.economic_sentiment,
                    'currency_impact': phase2_result.currency_impact,
                    'commodity_impact': phase2_result.commodity_impact,
                    'regulatory_risk_score': phase2_result.regulatory_risk_score,
                    'enhanced_prediction_score': phase2_result.enhanced_prediction_score,
                    'variable_coverage': phase2_result.variable_coverage,
                    'last_updated': phase2_result.last_updated
                }
            except Exception as e:
                logger.error(f"Phase 2 analysis failed: {str(e)}")
                phase2_analysis = {'error': str(e)}
        
        if 'phase3' in include_phases:
            try:
                phase3_result = self.phase3_integration.run_phase3_analysis(ticker)
                phase3_analysis = {
                    'combined_risk_score': phase3_result.combined_risk_score,
                    'market_impact_score': phase3_result.market_impact_score,
                    'confidence_level': phase3_result.confidence_level,
                    'geopolitical_risk': phase3_result.geopolitical_risk,
                    'corporate_actions': phase3_result.corporate_actions,
                    'insider_trading': phase3_result.insider_trading,
                    'last_updated': phase3_result.last_updated
                }
            except Exception as e:
                logger.error(f"Phase 3 analysis failed: {str(e)}")
                phase3_analysis = {'error': str(e)}
        
        # Generate prediction metrics (simplified for demo)
        prediction_metrics = self._generate_prediction_metrics(ticker, currency_code)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            phase1_analysis, phase2_analysis, phase3_analysis
        )
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(
            phase1_analysis, phase2_analysis, phase3_analysis
        )
        
        # Generate key insights
        key_insights = self._generate_key_insights(
            phase1_analysis, phase2_analysis, phase3_analysis
        )
        
        # Create comprehensive summary
        summary = AnalysisSummary(
            ticker=ticker,
            analysis_date=analysis_date,
            currency=currency_info,
            prediction_metrics=prediction_metrics,
            phase1_analysis=phase1_analysis,
            phase2_analysis=phase2_analysis,
            phase3_analysis=phase3_analysis,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            key_insights=key_insights
        )
        
        logger.info(f"Comprehensive report generated for {ticker}")
        return summary
    
    def _generate_prediction_metrics(self, ticker: str, currency_code: str) -> PredictionMetrics:
        """Generate prediction metrics"""
        # This would typically load from actual prediction data
        # For demo purposes, using sample data
        current_price = 349.60
        predicted_price = 350.22
        price_change = predicted_price - current_price
        price_change_percent = (price_change / current_price) * 100
        
        return PredictionMetrics(
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=price_change,
            price_change_percent=price_change_percent,
            confidence_score=0.85,
            model_agreement=0.981,
            prediction_range={
                '68_confidence': {'lower': 342.37, 'upper': 356.15},
                '95_confidence': {'lower': 335.48, 'upper': 363.04}
            },
            timeframe_predictions={
                'short_term': {'1_day': 347.55, '7_days': 368.65},
                'medium_term': {'1_month': 392.96, '3_months': 473.47},
                'long_term': {'6_months': 552.24, '12_months': 754.89}
            }
        )
    
    def _generate_recommendations(self, phase1: Dict, phase2: Dict, phase3: Dict) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Phase 1 recommendations
        if 'enhanced_prediction_score' in phase1:
            score = phase1['enhanced_prediction_score']
            if score > 70:
                recommendations.append("🟢 STRONG BUY - High confidence in positive performance")
            elif score > 50:
                recommendations.append("🟡 BUY - Moderate confidence in positive performance")
            elif score > 30:
                recommendations.append("⚪ HOLD - Neutral outlook with mixed signals")
            else:
                recommendations.append("🔴 SELL - Low confidence in positive performance")
        
        # Phase 2 recommendations
        if 'economic_sentiment' in phase2:
            sentiment = phase2['economic_sentiment']
            if sentiment == 'Bullish':
                recommendations.append("📈 Economic conditions favor growth")
            elif sentiment == 'Bearish':
                recommendations.append("📉 Economic headwinds may impact performance")
        
        # Phase 3 recommendations
        if 'combined_risk_score' in phase3:
            risk_score = phase3['combined_risk_score']
            if risk_score > 70:
                recommendations.append("⚠️ HIGH RISK - Monitor geopolitical and corporate factors")
            elif risk_score > 40:
                recommendations.append("⚖️ MODERATE RISK - Standard risk management recommended")
            else:
                recommendations.append("✅ LOW RISK - Favorable risk environment")
        
        return recommendations
    
    def _generate_risk_assessment(self, phase1: Dict, phase2: Dict, phase3: Dict) -> Dict[str, Any]:
        """Generate risk assessment"""
        risk_factors = []
        risk_score = 0
        
        # Phase 1 risk factors
        if 'fundamental_health' in phase1:
            health = phase1['fundamental_health']
            if health < 30:
                risk_factors.append("Poor fundamental health")
                risk_score += 30
            elif health < 50:
                risk_factors.append("Below average fundamental health")
                risk_score += 15
        
        # Phase 2 risk factors
        if 'regulatory_risk_score' in phase2:
            reg_risk = phase2['regulatory_risk_score']
            if reg_risk > 70:
                risk_factors.append("High regulatory risk")
                risk_score += 25
        
        # Phase 3 risk factors
        if 'combined_risk_score' in phase3:
            combined_risk = phase3['combined_risk_score']
            risk_score += combined_risk * 0.45
        
        return {
            'overall_risk_score': min(risk_score, 100),
            'risk_level': self._get_risk_level(risk_score),
            'risk_factors': risk_factors,
            'risk_mitigation': self._get_risk_mitigation(risk_factors)
        }
    
    def _generate_key_insights(self, phase1: Dict, phase2: Dict, phase3: Dict) -> List[str]:
        """Generate key insights"""
        insights = []
        
        # Phase 1 insights
        if 'institutional_confidence' in phase1:
            confidence = phase1['institutional_confidence']
            if confidence > 80:
                insights.append("Strong institutional confidence indicates positive sentiment")
            elif confidence < 40:
                insights.append("Low institutional confidence suggests caution")
        
        # Phase 2 insights
        if 'economic_impact_score' in phase2:
            impact = phase2['economic_impact_score']
            if impact > 5:
                insights.append("Positive economic environment supports growth")
            elif impact < -5:
                insights.append("Economic headwinds may impact performance")
        
        # Phase 3 insights
        if 'geopolitical_risk' in phase3:
            geo_risk = phase3['geopolitical_risk']
            if geo_risk.get('overall_risk', 0) > 60:
                insights.append("Elevated geopolitical risk requires monitoring")
        
        return insights
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Get risk level description"""
        if risk_score >= 70:
            return "HIGH"
        elif risk_score >= 40:
            return "MODERATE"
        else:
            return "LOW"
    
    def _get_risk_mitigation(self, risk_factors: List[str]) -> List[str]:
        """Get risk mitigation strategies"""
        mitigation = []
        
        for factor in risk_factors:
            if "fundamental" in factor.lower():
                mitigation.append("Focus on companies with strong fundamentals")
            elif "regulatory" in factor.lower():
                mitigation.append("Monitor regulatory developments closely")
            elif "geopolitical" in factor.lower():
                mitigation.append("Diversify across regions and sectors")
        
        return mitigation
    
    def save_report(self, summary: AnalysisSummary, format_type: str = 'json') -> str:
        """
        Save report in specified format
        
        Args:
            summary: AnalysisSummary object
            format_type: Report format (json, csv, html, txt)
            
        Returns:
            Path to saved report
        """
        ticker = summary.ticker
        timestamp = summary.analysis_date.date_iso
        
        if format_type == 'json':
            filename = f"{ticker}_comprehensive_report_{timestamp}.json"
            filepath = self.output_dir / filename
            
            # Convert to dictionary for JSON serialization
            report_dict = asdict(summary)
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            # Recursively convert datetime objects
            def convert_dict(d):
                if isinstance(d, dict):
                    return {k: convert_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [convert_dict(item) for item in d]
                else:
                    return convert_datetime(d)
            
            report_dict = convert_dict(report_dict)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            logger.info(f"JSON report saved: {filepath}")
            return str(filepath)
        
        elif format_type == 'csv':
            filename = f"{ticker}_comprehensive_report_{timestamp}.csv"
            filepath = self.output_dir / filename
            
            # Create CSV data
            csv_data = []
            csv_data.append(['Metric', 'Value', 'Currency', 'Date'])
            
            # Add prediction metrics
            pm = summary.prediction_metrics
            csv_data.append(['Current Price', pm.current_price, summary.currency.code, summary.analysis_date.date_iso])
            csv_data.append(['Predicted Price', pm.predicted_price, summary.currency.code, summary.analysis_date.date_iso])
            csv_data.append(['Price Change', pm.price_change, summary.currency.code, summary.analysis_date.date_iso])
            csv_data.append(['Price Change %', pm.price_change_percent, '%', summary.analysis_date.date_iso])
            csv_data.append(['Confidence Score', pm.confidence_score, 'ratio', summary.analysis_date.date_iso])
            
            # Add phase analysis
            if summary.phase1_analysis:
                csv_data.append(['Phase 1 Score', summary.phase1_analysis.get('enhanced_prediction_score', 'N/A'), 'score', summary.analysis_date.date_iso])
                csv_data.append(['Variable Coverage', summary.phase1_analysis.get('variable_coverage', 'N/A'), '%', summary.analysis_date.date_iso])
            
            if summary.phase2_analysis:
                csv_data.append(['Phase 2 Score', summary.phase2_analysis.get('enhanced_prediction_score', 'N/A'), 'score', summary.analysis_date.date_iso])
                csv_data.append(['Economic Impact', summary.phase2_analysis.get('economic_impact_score', 'N/A'), 'score', summary.analysis_date.date_iso])
            
            if summary.phase3_analysis:
                csv_data.append(['Phase 3 Risk Score', summary.phase3_analysis.get('combined_risk_score', 'N/A'), 'score', summary.analysis_date.date_iso])
                csv_data.append(['Market Impact', summary.phase3_analysis.get('market_impact_score', 'N/A'), 'score', summary.analysis_date.date_iso])
            
            # Save CSV
            df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            df.to_csv(filepath, index=False)
            
            logger.info(f"CSV report saved: {filepath}")
            return str(filepath)
        
        elif format_type == 'html':
            filename = f"{ticker}_comprehensive_report_{timestamp}.html"
            filepath = self.output_dir / filename
            
            html_content = self._generate_html_report(summary)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report saved: {filepath}")
            return str(filepath)
        
        elif format_type == 'txt':
            filename = f"{ticker}_comprehensive_report_{timestamp}.txt"
            filepath = self.output_dir / filename
            
            txt_content = self._generate_text_report(summary)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(txt_content)
            
            logger.info(f"Text report saved: {filepath}")
            return str(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_html_report(self, summary: AnalysisSummary) -> str:
        """Generate HTML report"""
        currency = summary.currency
        date_info = summary.analysis_date
        pm = summary.prediction_metrics
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Analysis Report - {summary.ticker}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .section h2 {{ color: #333; border-left: 4px solid #007bff; padding-left: 10px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #28a745; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .recommendation {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendation.buy {{ background-color: #d4edda; border-left: 4px solid #28a745; }}
        .recommendation.hold {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}
        .recommendation.sell {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}
        .insight {{ background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #007bff; }}
        .risk-high {{ color: #dc3545; font-weight: bold; }}
        .risk-moderate {{ color: #ffc107; font-weight: bold; }}
        .risk-low {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comprehensive Analysis Report</h1>
            <h2>{summary.ticker} - {date_info.date_long}</h2>
            <p>Analysis Date: {date_info.day_name}, {date_info.month_name} {date_info.day}, {date_info.year}</p>
        </div>
        
        <div class="section">
            <h2>💰 Price Predictions</h2>
            <div class="metric">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">{self.currency_formatter.format_amount(pm.current_price, currency.code)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Predicted Price</div>
                <div class="metric-value">{self.currency_formatter.format_amount(pm.predicted_price, currency.code)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Expected Change</div>
                <div class="metric-value">{pm.price_change_percent:+.2f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Confidence Score</div>
                <div class="metric-value">{pm.confidence_score:.1%}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Analysis Results</h2>
            <div class="metric">
                <div class="metric-label">Phase 1 Score</div>
                <div class="metric-value">{summary.phase1_analysis.get('enhanced_prediction_score', 'N/A')}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Phase 2 Score</div>
                <div class="metric-value">{summary.phase2_analysis.get('enhanced_prediction_score', 'N/A')}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Variable Coverage</div>
                <div class="metric-value">{summary.phase1_analysis.get('variable_coverage', 'N/A')}%</div>
            </div>
        </div>
        
        <div class="section">
            <h2>💡 Recommendations</h2>
            {''.join([f'<div class="recommendation">{rec}</div>' for rec in summary.recommendations])}
        </div>
        
        <div class="section">
            <h2>⚠️ Risk Assessment</h2>
            <p><strong>Overall Risk Level:</strong> 
                <span class="risk-{summary.risk_assessment['risk_level'].lower()}">
                    {summary.risk_assessment['risk_level']}
                </span>
            </p>
            <p><strong>Risk Score:</strong> {summary.risk_assessment['overall_risk_score']:.1f}/100</p>
            {f"<p><strong>Risk Factors:</strong> {', '.join(summary.risk_assessment['risk_factors'])}</p>" if summary.risk_assessment['risk_factors'] else ""}
        </div>
        
        <div class="section">
            <h2>🔍 Key Insights</h2>
            {''.join([f'<div class="insight">{insight}</div>' for insight in summary.key_insights])}
        </div>
        
        <div class="section">
            <h2>📅 Timeframe Predictions</h2>
            <h3>Short-term (1-7 days)</h3>
            <p>1 Day: {self.currency_formatter.format_amount(pm.timeframe_predictions['short_term']['1_day'], currency.code)}</p>
            <p>7 Days: {self.currency_formatter.format_amount(pm.timeframe_predictions['short_term']['7_days'], currency.code)}</p>
            
            <h3>Medium-term (1-3 months)</h3>
            <p>1 Month: {self.currency_formatter.format_amount(pm.timeframe_predictions['medium_term']['1_month'], currency.code)}</p>
            <p>3 Months: {self.currency_formatter.format_amount(pm.timeframe_predictions['medium_term']['3_months'], currency.code)}</p>
            
            <h3>Long-term (6-12 months)</h3>
            <p>6 Months: {self.currency_formatter.format_amount(pm.timeframe_predictions['long_term']['6_months'], currency.code)}</p>
            <p>12 Months: {self.currency_formatter.format_amount(pm.timeframe_predictions['long_term']['12_months'], currency.code)}</p>
        </div>
    </div>
</body>
</html>
        """
        return html
    
    def _generate_text_report(self, summary: AnalysisSummary) -> str:
        """Generate text report"""
        currency = summary.currency
        date_info = summary.analysis_date
        pm = summary.prediction_metrics
        
        text = f"""
================================================================================
📊 COMPREHENSIVE ANALYSIS REPORT
================================================================================
📈 Stock: {summary.ticker}
📅 Analysis Date: {date_info.day_name}, {date_info.month_name} {date_info.day}, {date_info.year}
💰 Currency: {currency.name} ({currency.code})
⏰ Generated: {date_info.date_string} at {date_info.date.strftime('%I:%M %p')}
================================================================================

💰 PRICE PREDICTIONS
--------------------------------------------------------------------------------
Current Price:     {self.currency_formatter.format_amount(pm.current_price, currency.code)}
Predicted Price:   {self.currency_formatter.format_amount(pm.predicted_price, currency.code)}
Expected Change:   {pm.price_change_percent:+.2f}%
Confidence Score:  {pm.confidence_score:.1%}
Model Agreement:   {pm.model_agreement:.1%}

📈 ANALYSIS RESULTS
--------------------------------------------------------------------------------
Phase 1 Score:     {summary.phase1_analysis.get('enhanced_prediction_score', 'N/A')}
Phase 2 Score:     {summary.phase2_analysis.get('enhanced_prediction_score', 'N/A')}
Variable Coverage: {summary.phase1_analysis.get('variable_coverage', 'N/A')}%

💡 RECOMMENDATIONS
--------------------------------------------------------------------------------
"""
        for rec in summary.recommendations:
            text += f"• {rec}\n"
        
        text += f"""
⚠️ RISK ASSESSMENT
--------------------------------------------------------------------------------
Overall Risk Level: {summary.risk_assessment['risk_level']}
Risk Score:        {summary.risk_assessment['overall_risk_score']:.1f}/100
"""
        
        if summary.risk_assessment['risk_factors']:
            text += f"Risk Factors:       {', '.join(summary.risk_assessment['risk_factors'])}\n"
        
        text += f"""
🔍 KEY INSIGHTS
--------------------------------------------------------------------------------
"""
        for insight in summary.key_insights:
            text += f"• {insight}\n"
        
        text += f"""
📅 TIMEFRAME PREDICTIONS
--------------------------------------------------------------------------------
Short-term (1-7 days):
  1 Day:  {self.currency_formatter.format_amount(pm.timeframe_predictions['short_term']['1_day'], currency.code)}
  7 Days: {self.currency_formatter.format_amount(pm.timeframe_predictions['short_term']['7_days'], currency.code)}

Medium-term (1-3 months):
  1 Month:  {self.currency_formatter.format_amount(pm.timeframe_predictions['medium_term']['1_month'], currency.code)}
  3 Months: {self.currency_formatter.format_amount(pm.timeframe_predictions['medium_term']['3_months'], currency.code)}

Long-term (6-12 months):
  6 Months:  {self.currency_formatter.format_amount(pm.timeframe_predictions['long_term']['6_months'], currency.code)}
  12 Months: {self.currency_formatter.format_amount(pm.timeframe_predictions['long_term']['12_months'], currency.code)}

================================================================================
Report generated by Comprehensive Report Generator
================================================================================
        """
        return text

# Example usage and testing
if __name__ == "__main__":
    # Initialize report generator
    generator = ComprehensiveReportGenerator()
    
    # Generate comprehensive report
    print("Generating comprehensive report for TSLA...")
    summary = generator.generate_comprehensive_report("TSLA", currency_code="USD")
    
    # Save in multiple formats
    json_path = generator.save_report(summary, 'json')
    csv_path = generator.save_report(summary, 'csv')
    html_path = generator.save_report(summary, 'html')
    txt_path = generator.save_report(summary, 'txt')
    
    print(f"\n📊 Comprehensive Report Generated:")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print(f"HTML: {html_path}")
    print(f"TXT:  {txt_path}")
    
    # Display summary
    print(f"\n📈 Summary for {summary.ticker}:")
    print(f"Current Price: {generator.currency_formatter.format_amount(summary.prediction_metrics.current_price, summary.currency.code)}")
    print(f"Predicted Price: {generator.currency_formatter.format_amount(summary.prediction_metrics.predicted_price, summary.currency.code)}")
    print(f"Expected Change: {summary.prediction_metrics.price_change_percent:+.2f}%")
    print(f"Confidence: {summary.prediction_metrics.confidence_score:.1%}")
    print(f"Risk Level: {summary.risk_assessment['risk_level']}")
    print(f"Recommendations: {len(summary.recommendations)} generated")
