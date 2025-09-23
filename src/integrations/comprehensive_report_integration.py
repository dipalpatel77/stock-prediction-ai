#!/usr/bin/env python3
"""
Comprehensive Report Integration
================================

Integrates all report generation features:
- Multi-currency support
- Enhanced date formatting
- Comprehensive analysis reports
- Multiple output formats
- Real-time data integration

This module serves as the main entry point for generating comprehensive reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.core.report_generator import ComprehensiveReportGenerator, AnalysisSummary
from src.core.currency_service import CurrencyService
from src.utils.enhanced_date_utils import EnhancedDateUtils
from main.services.economic_data_service import EconomicDataService
from .phase1_integration import Phase1Integration
from .phase2_integration import Phase2Integration
from .phase3_integration import Phase3Integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveReportIntegration:
    """
    Comprehensive Report Integration Service
    
    Orchestrates all report generation features with multi-currency support
    and enhanced date formatting
    """
    
    def __init__(self, output_dir: str = "comprehensive_reports"):
        """Initialize comprehensive report integration"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize services
        self.report_generator = ComprehensiveReportGenerator(str(self.output_dir))
        self.currency_service = CurrencyService()
        self.date_utils = EnhancedDateUtils()
        self.economic_service = EconomicDataService()
        
        # Initialize phase integrations
        self.phase1_integration = Phase1Integration()
        self.phase2_integration = Phase2Integration()
        self.phase3_integration = Phase3Integration()
        
        logger.info("Comprehensive Report Integration initialized")
    
    def generate_multi_currency_report(self, ticker: str, 
                                     currencies: List[str] = None,
                                     include_phases: List[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report in multiple currencies
        
        Args:
            ticker: Stock ticker symbol
            currencies: List of currency codes (default: ['USD', 'INR', 'EUR', 'GBP'])
            include_phases: List of phases to include (default: all)
            
        Returns:
            Dictionary with reports in multiple currencies
        """
        if currencies is None:
            currencies = ['USD', 'INR', 'EUR', 'GBP']
        
        if include_phases is None:
            include_phases = ['phase1', 'phase2', 'phase3']
        
        logger.info(f"Generating multi-currency report for {ticker}")
        
        # Generate base report in USD
        base_report = self.report_generator.generate_comprehensive_report(
            ticker, currency_code='USD', include_phases=include_phases
        )
        
        # Convert to other currencies
        multi_currency_reports = {}
        
        for currency in currencies:
            try:
                # Convert prediction metrics
                converted_metrics = self._convert_prediction_metrics(
                    base_report.prediction_metrics, 'USD', currency
                )
                
                # Create currency-specific report
                currency_report = AnalysisSummary(
                    ticker=base_report.ticker,
                    analysis_date=base_report.analysis_date,
                    currency=self.currency_service.get_currency_info(currency),
                    prediction_metrics=converted_metrics,
                    phase1_analysis=base_report.phase1_analysis,
                    phase2_analysis=base_report.phase2_analysis,
                    phase3_analysis=base_report.phase3_analysis,
                    recommendations=base_report.recommendations,
                    risk_assessment=base_report.risk_assessment,
                    key_insights=base_report.key_insights
                )
                
                multi_currency_reports[currency] = currency_report
                
                logger.info(f"Generated report for {ticker} in {currency}")
                
            except Exception as e:
                logger.error(f"Failed to generate report for {currency}: {str(e)}")
                multi_currency_reports[currency] = {'error': str(e)}
        
        return multi_currency_reports
    
    def _convert_prediction_metrics(self, metrics, from_currency: str, to_currency: str):
        """Convert prediction metrics to different currency"""
        from dataclasses import replace
        
        # Convert main prices
        current_price = self.currency_service.convert_amount(
            metrics.current_price, from_currency, to_currency
        )
        predicted_price = self.currency_service.convert_amount(
            metrics.predicted_price, from_currency, to_currency
        )
        
        # Convert price change (absolute)
        price_change = predicted_price - current_price
        
        # Convert prediction range
        converted_range = {}
        for confidence_level, range_data in metrics.prediction_range.items():
            converted_range[confidence_level] = {
                'lower': self.currency_service.convert_amount(
                    range_data['lower'], from_currency, to_currency
                ),
                'upper': self.currency_service.convert_amount(
                    range_data['upper'], from_currency, to_currency
                )
            }
        
        # Convert timeframe predictions
        converted_timeframe = {}
        for timeframe, predictions in metrics.timeframe_predictions.items():
            converted_timeframe[timeframe] = {}
            for period, price in predictions.items():
                converted_timeframe[timeframe][period] = self.currency_service.convert_amount(
                    price, from_currency, to_currency
                )
        
        # Create new metrics object
        return replace(
            metrics,
            current_price=current_price,
            predicted_price=predicted_price,
            price_change=price_change,
            prediction_range=converted_range,
            timeframe_predictions=converted_timeframe
        )
    
    def save_multi_currency_reports(self, reports: Dict[str, Any], 
                                  formats: List[str] = None) -> Dict[str, Dict[str, str]]:
        """
        Save reports in multiple currencies and formats
        
        Args:
            reports: Dictionary of reports by currency
            formats: List of formats to save (default: ['json', 'html', 'csv'])
            
        Returns:
            Dictionary with file paths by currency and format
        """
        if formats is None:
            formats = ['json', 'html', 'csv', 'txt']
        
        saved_files = {}
        
        for currency, report in reports.items():
            if isinstance(report, dict) and 'error' in report:
                logger.warning(f"Skipping {currency} due to error: {report['error']}")
                continue
            
            saved_files[currency] = {}
            
            for format_type in formats:
                try:
                    filepath = self.report_generator.save_report(report, format_type)
                    saved_files[currency][format_type] = filepath
                    logger.info(f"Saved {currency} report in {format_type} format: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to save {currency} report in {format_type}: {str(e)}")
                    saved_files[currency][format_type] = f"Error: {str(e)}"
        
        return saved_files
    
    def generate_enhanced_date_report(self, ticker: str, 
                                    analysis_date: datetime = None,
                                    timezone: str = 'UTC') -> Dict[str, Any]:
        """
        Generate report with enhanced date formatting
        
        Args:
            ticker: Stock ticker symbol
            analysis_date: Analysis date (default: now)
            timezone: Timezone for date formatting
            
        Returns:
            Dictionary with enhanced date information
        """
        if analysis_date is None:
            analysis_date = datetime.now()
        
        logger.info(f"Generating enhanced date report for {ticker}")
        
        # Get comprehensive date information
        date_info = self.date_utils.get_week_info(analysis_date)
        fiscal_period = self.date_utils.get_fiscal_period(analysis_date)
        
        # Generate base report
        base_report = self.report_generator.generate_comprehensive_report(ticker)
        
        # Enhance with date information
        enhanced_report = {
            'ticker': ticker,
            'analysis_date': {
                'datetime': analysis_date,
                'timezone': timezone,
                'formatted_dates': {
                    'iso': self.date_utils.format_date(analysis_date, 'iso', timezone),
                    'us': self.date_utils.format_date(analysis_date, 'us', timezone),
                    'eu': self.date_utils.format_date(analysis_date, 'eu', timezone),
                    'readable': self.date_utils.format_date(analysis_date, 'readable', timezone),
                    'timestamp': self.date_utils.format_date(analysis_date, 'timestamp', timezone),
                    'filename': self.date_utils.format_date(analysis_date, 'filename', timezone)
                },
                'week_info': date_info,
                'fiscal_period': {
                    'fiscal_year': fiscal_period.fiscal_year,
                    'quarter': fiscal_period.quarter,
                    'period_name': fiscal_period.period_name,
                    'period_start': fiscal_period.period_start.isoformat(),
                    'period_end': fiscal_period.period_end.isoformat(),
                    'days_in_period': fiscal_period.days_in_period,
                    'business_days_in_period': fiscal_period.business_days_in_period
                }
            },
            'report_data': base_report
        }
        
        return enhanced_report
    
    def generate_market_analysis_report(self, ticker: str, 
                                      markets: List[str] = None) -> Dict[str, Any]:
        """
        Generate report with market hours analysis
        
        Args:
            ticker: Stock ticker symbol
            markets: List of markets to analyze (default: major markets)
            
        Returns:
            Dictionary with market analysis
        """
        if markets is None:
            markets = ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'NSE', 'BSE']
        
        logger.info(f"Generating market analysis report for {ticker}")
        
        # Generate base report
        base_report = self.report_generator.generate_comprehensive_report(ticker)
        
        # Analyze market hours
        market_analysis = {}
        current_time = datetime.now()
        
        for market in markets:
            try:
                is_open = self.date_utils.is_market_open(market, current_time)
                next_trading_day = self.date_utils.get_next_trading_day(market)
                previous_trading_day = self.date_utils.get_previous_trading_day(market)
                
                market_analysis[market] = {
                    'is_open': is_open,
                    'status': '🟢 Open' if is_open else '🔴 Closed',
                    'next_trading_day': next_trading_day.isoformat(),
                    'previous_trading_day': previous_trading_day.isoformat(),
                    'current_time': current_time.isoformat(),
                    'timezone_info': self.date_utils.get_timezone_info(
                        self.date_utils.market_hours[market].timezone
                    )
                }
            except Exception as e:
                logger.error(f"Failed to analyze market {market}: {str(e)}")
                market_analysis[market] = {'error': str(e)}
        
        return {
            'ticker': ticker,
            'market_analysis': market_analysis,
            'report_data': base_report
        }
    
    def generate_currency_analysis_report(self, ticker: str, 
                                        base_currency: str = 'USD') -> Dict[str, Any]:
        """
        Generate report with currency analysis
        
        Args:
            ticker: Stock ticker symbol
            base_currency: Base currency for analysis
            
        Returns:
            Dictionary with currency analysis
        """
        logger.info(f"Generating currency analysis report for {ticker}")
        
        # Generate base report
        base_report = self.report_generator.generate_comprehensive_report(ticker)
        
        # Get currency pairs
        currency_pairs = self.currency_service.get_currency_pairs(base_currency)
        
        # Get market summary
        market_summary = self.currency_service.get_market_summary()
        
        # Get historical rates for major pairs
        historical_rates = {}
        major_pairs = ['USD/INR', 'EUR/USD', 'GBP/USD', 'USD/JPY']
        
        for pair in major_pairs:
            try:
                from_curr, to_curr = pair.split('/')
                historical = self.currency_service.get_historical_rates(from_curr, to_curr, 30)
                historical_rates[pair] = {
                    'data': historical.to_dict('records'),
                    'current_rate': historical['rate'].iloc[-1],
                    'change_30d': ((historical['rate'].iloc[-1] - historical['rate'].iloc[0]) / historical['rate'].iloc[0]) * 100
                }
            except Exception as e:
                logger.error(f"Failed to get historical rates for {pair}: {str(e)}")
                historical_rates[pair] = {'error': str(e)}
        
        return {
            'ticker': ticker,
            'base_currency': base_currency,
            'currency_pairs': [
                {
                    'pair': pair.pair,
                    'rate': pair.rate,
                    'change_24h': pair.change_24h,
                    'change_percent_24h': pair.change_percent_24h,
                    'last_updated': pair.last_updated.isoformat()
                }
                for pair in currency_pairs[:10]  # Top 10 pairs
            ],
            'market_summary': market_summary,
            'historical_rates': historical_rates,
            'report_data': base_report
        }
    
    def generate_comprehensive_summary_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate comprehensive summary report with all features
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with comprehensive summary
        """
        logger.info(f"Generating comprehensive summary report for {ticker}")
        
        # Generate all types of reports
        multi_currency_reports = self.generate_multi_currency_report(ticker)
        enhanced_date_report = self.generate_enhanced_date_report(ticker)
        market_analysis_report = self.generate_market_analysis_report(ticker)
        currency_analysis_report = self.generate_currency_analysis_report(ticker)
        
        # Create comprehensive summary
        comprehensive_summary = {
            'ticker': ticker,
            'generated_at': datetime.now().isoformat(),
            'report_types': {
                'multi_currency': list(multi_currency_reports.keys()),
                'enhanced_dates': True,
                'market_analysis': True,
                'currency_analysis': True
            },
            'multi_currency_reports': multi_currency_reports,
            'enhanced_date_report': enhanced_date_report,
            'market_analysis_report': market_analysis_report,
            'currency_analysis_report': currency_analysis_report,
            'summary_statistics': {
                'total_currencies': len(multi_currency_reports),
                'markets_analyzed': len(market_analysis_report['market_analysis']),
                'currency_pairs': len(currency_analysis_report['currency_pairs']),
                'report_generation_time': datetime.now().isoformat()
            }
        }
        
        return comprehensive_summary
    
    def save_comprehensive_summary(self, summary: Dict[str, Any], 
                                 filename: str = None) -> str:
        """
        Save comprehensive summary report
        
        Args:
            summary: Comprehensive summary dictionary
            filename: Custom filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            ticker = summary['ticker']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{ticker}_comprehensive_summary_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Comprehensive summary saved: {filepath}")
        return str(filepath)

# Example usage and testing
if __name__ == "__main__":
    # Initialize comprehensive report integration
    integration = ComprehensiveReportIntegration()
    
    print("📊 Comprehensive Report Integration Test")
    print("=" * 60)
    
    # Test multi-currency report
    print("\n🌍 Multi-Currency Report Generation:")
    multi_currency_reports = integration.generate_multi_currency_report("TSLA")
    
    for currency, report in multi_currency_reports.items():
        if isinstance(report, dict) and 'error' in report:
            print(f"{currency}: Error - {report['error']}")
        else:
            pm = report.prediction_metrics
            print(f"{currency}: {integration.currency_service.format_currency(pm.current_price, currency)} → {integration.currency_service.format_currency(pm.predicted_price, currency)} ({pm.price_change_percent:+.2f}%)")
    
    # Test enhanced date report
    print("\n📅 Enhanced Date Report Generation:")
    enhanced_date_report = integration.generate_enhanced_date_report("TSLA")
    date_info = enhanced_date_report['analysis_date']
    print(f"Analysis Date: {date_info['formatted_dates']['readable']}")
    print(f"Fiscal Period: {date_info['fiscal_period']['period_name']}")
    print(f"Week: {date_info['week_info']['iso_week']} of {date_info['week_info']['iso_year']}")
    
    # Test market analysis report
    print("\n🏪 Market Analysis Report Generation:")
    market_analysis_report = integration.generate_market_analysis_report("TSLA")
    for market, analysis in market_analysis_report['market_analysis'].items():
        if isinstance(analysis, dict) and 'error' in analysis:
            print(f"{market}: Error - {analysis['error']}")
        else:
            print(f"{market}: {analysis['status']}")
    
    # Test currency analysis report
    print("\n💱 Currency Analysis Report Generation:")
    currency_analysis_report = integration.generate_currency_analysis_report("TSLA")
    print(f"Currency Pairs Analyzed: {len(currency_analysis_report['currency_pairs'])}")
    print(f"Market Status: {currency_analysis_report['market_summary']['market_status']}")
    
    # Test comprehensive summary
    print("\n📋 Comprehensive Summary Generation:")
    comprehensive_summary = integration.generate_comprehensive_summary_report("TSLA")
    print(f"Total Currencies: {comprehensive_summary['summary_statistics']['total_currencies']}")
    print(f"Markets Analyzed: {comprehensive_summary['summary_statistics']['markets_analyzed']}")
    print(f"Currency Pairs: {comprehensive_summary['summary_statistics']['currency_pairs']}")
    
    # Save comprehensive summary
    summary_file = integration.save_comprehensive_summary(comprehensive_summary)
    print(f"\n💾 Comprehensive Summary Saved: {summary_file}")
    
    # Save multi-currency reports
    print("\n💾 Saving Multi-Currency Reports:")
    saved_files = integration.save_multi_currency_reports(multi_currency_reports)
    
    for currency, formats in saved_files.items():
        print(f"\n{currency} Reports:")
        for format_type, filepath in formats.items():
            if not filepath.startswith('Error'):
                print(f"  {format_type.upper()}: {filepath}")
            else:
                print(f"  {format_type.upper()}: {filepath}")
    
    print("\n✅ Comprehensive Report Integration test completed!")
    print(f"📁 All reports saved to: {integration.output_dir}")
