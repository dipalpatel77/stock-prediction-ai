#!/usr/bin/env python3
"""
Enhanced Fundamental Analysis Module
====================================

This module provides comprehensive fundamental analysis capabilities including:
- EPS (Earnings Per Share) calculation
- Net Profit Margin analysis
- Dividend announcement tracking
- Financial ratio calculations
- Corporate action monitoring

Part of Phase 1 implementation to achieve 60% variable coverage.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FundamentalMetrics:
    """Data class to store fundamental analysis results"""
    eps: float
    eps_growth: float
    net_profit_margin: float
    revenue_growth: float
    dividend_yield: float
    dividend_announcement: str
    debt_to_equity: float
    current_ratio: float
    roe: float
    roa: float
    pe_ratio: float
    pb_ratio: float
    financial_health_score: float
    last_updated: datetime

class FundamentalAnalyzer:
    """
    Enhanced fundamental analysis for stock prediction
    
    Implements critical missing variables from Phase 1:
    - EPS calculation and growth
    - Net profit margin analysis
    - Dividend announcement tracking
    - Comprehensive financial ratios
    """
    
    def __init__(self, cache_duration: int = 24):
        """
        Initialize the fundamental analyzer
        
        Args:
            cache_duration: Cache duration in hours
        """
        self.cache_duration = cache_duration
        self.cache = {}
        self.cache_timestamps = {}
        
    def get_fundamental_metrics(self, ticker: str) -> FundamentalMetrics:
        """
        Get comprehensive fundamental metrics for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            FundamentalMetrics object with all calculated metrics
        """
        try:
            # Check cache first
            if self._is_cache_valid(ticker):
                logger.info(f"Using cached fundamental data for {ticker}")
                return self.cache[ticker]
            
            logger.info(f"Calculating fundamental metrics for {ticker}")
            
            # Get stock data
            stock = yf.Ticker(ticker)
            
            # Calculate EPS
            eps, eps_growth = self._calculate_eps(stock)
            
            # Calculate profit margins
            net_profit_margin, revenue_growth = self._calculate_profit_margins(stock)
            
            # Get dividend information
            dividend_yield, dividend_announcement = self._get_dividend_info(stock)
            
            # Calculate financial ratios
            ratios = self._calculate_financial_ratios(stock)
            
            # Calculate financial health score
            financial_health_score = self._calculate_financial_health_score(
                eps, net_profit_margin, ratios
            )
            
            # Create metrics object
            metrics = FundamentalMetrics(
                eps=eps,
                eps_growth=eps_growth,
                net_profit_margin=net_profit_margin,
                revenue_growth=revenue_growth,
                dividend_yield=dividend_yield,
                dividend_announcement=dividend_announcement,
                debt_to_equity=ratios.get('debt_to_equity', 0.0),
                current_ratio=ratios.get('current_ratio', 0.0),
                roe=ratios.get('roe', 0.0),
                roa=ratios.get('roa', 0.0),
                pe_ratio=ratios.get('pe_ratio', 0.0),
                pb_ratio=ratios.get('pb_ratio', 0.0),
                financial_health_score=financial_health_score,
                last_updated=datetime.now()
            )
            
            # Cache the results
            self._cache_results(ticker, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating fundamental metrics for {ticker}: {str(e)}")
            return self._get_fallback_metrics(ticker)
    
    def _calculate_eps(self, stock: yf.Ticker) -> Tuple[float, float]:
        """
        Calculate EPS and EPS growth
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Tuple of (current_eps, eps_growth_percentage)
        """
        try:
            # Get income statement
            income_stmt = stock.income_stmt
            
            if income_stmt is None or income_stmt.empty:
                logger.warning(f"No income statement data available for EPS calculation")
                return 0.0, 0.0
            
            # Get net income and shares outstanding
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            shares_outstanding = income_stmt.loc['Basic Average Shares'].iloc[0] if 'Basic Average Shares' in income_stmt.index else 1
            
            # Calculate current EPS
            current_eps = net_income / shares_outstanding if shares_outstanding > 0 else 0
            
            # Calculate EPS growth (compare with previous period)
            if len(income_stmt.columns) >= 2:
                prev_net_income = income_stmt.loc['Net Income'].iloc[1] if 'Net Income' in income_stmt.index else 0
                prev_shares = income_stmt.loc['Basic Average Shares'].iloc[1] if 'Basic Average Shares' in income_stmt.index else 1
                prev_eps = prev_net_income / prev_shares if prev_shares > 0 else 0
                
                eps_growth = ((current_eps - prev_eps) / prev_eps * 100) if prev_eps != 0 else 0
            else:
                eps_growth = 0.0
            
            logger.info(f"EPS: {current_eps:.2f}, Growth: {eps_growth:.2f}%")
            return current_eps, eps_growth
            
        except Exception as e:
            logger.error(f"Error calculating EPS: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_profit_margins(self, stock: yf.Ticker) -> Tuple[float, float]:
        """
        Calculate net profit margin and revenue growth
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Tuple of (net_profit_margin_percentage, revenue_growth_percentage)
        """
        try:
            income_stmt = stock.income_stmt
            
            if income_stmt is None or income_stmt.empty:
                return 0.0, 0.0
            
            # Get current period data
            net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
            revenue = income_stmt.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_stmt.index else 1
            
            # Calculate net profit margin
            net_profit_margin = (net_income / revenue * 100) if revenue > 0 else 0
            
            # Calculate revenue growth
            if len(income_stmt.columns) >= 2:
                prev_revenue = income_stmt.loc['Total Revenue'].iloc[1] if 'Total Revenue' in income_stmt.index else 0
                revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
            else:
                revenue_growth = 0.0
            
            logger.info(f"Net Profit Margin: {net_profit_margin:.2f}%, Revenue Growth: {revenue_growth:.2f}%")
            return net_profit_margin, revenue_growth
            
        except Exception as e:
            logger.error(f"Error calculating profit margins: {str(e)}")
            return 0.0, 0.0
    
    def _get_dividend_info(self, stock: yf.Ticker) -> Tuple[float, str]:
        """
        Get dividend yield and recent dividend announcements
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Tuple of (dividend_yield_percentage, dividend_announcement_status)
        """
        try:
            # Get dividend yield
            dividend_yield = stock.info.get('dividendYield', 0) * 100 if stock.info.get('dividendYield') else 0
            
            # Get dividend history
            dividend_history = stock.dividends
            
            # Determine dividend announcement status
            if dividend_history is None or dividend_history.empty:
                dividend_announcement = "No Dividends"
            else:
                # Check if there's a recent dividend announcement (last 30 days)
                # Convert timezone-aware datetime to timezone-naive for comparison
                current_time = datetime.now()
                recent_dividends = dividend_history[
                    dividend_history.index.tz_localize(None) >= current_time - timedelta(days=30)
                ]
                if not recent_dividends.empty:
                    latest_dividend = recent_dividends.iloc[-1]
                    dividend_announcement = f"Recent: {latest_dividend:.2f}"
                else:
                    dividend_announcement = "No Recent Changes"
            
            logger.info(f"Dividend Yield: {dividend_yield:.2f}%, Status: {dividend_announcement}")
            return dividend_yield, dividend_announcement
            
        except Exception as e:
            logger.error(f"Error getting dividend info: {str(e)}")
            return 0.0, "Unknown"
    
    def _calculate_financial_ratios(self, stock: yf.Ticker) -> Dict[str, float]:
        """
        Calculate comprehensive financial ratios
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            Dictionary of financial ratios
        """
        try:
            ratios = {}
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            
            if balance_sheet is not None and not balance_sheet.empty:
                # Current Ratio
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0] if 'Total Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0] if 'Total Current Liabilities' in balance_sheet.index else 1
                ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities > 0 else 0
                
                # Debt to Equity
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 1
                ratios['debt_to_equity'] = total_debt / total_equity if total_equity > 0 else 0
                
                # ROE and ROA
                if income_stmt is not None and not income_stmt.empty:
                    net_income = income_stmt.loc['Net Income'].iloc[0] if 'Net Income' in income_stmt.index else 0
                    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
                    
                    ratios['roe'] = (net_income / total_equity * 100) if total_equity > 0 else 0
                    ratios['roa'] = (net_income / total_assets * 100) if total_assets > 0 else 0
            
            # Get market ratios
            info = stock.info
            ratios['pe_ratio'] = info.get('trailingPE', 0)
            ratios['pb_ratio'] = info.get('priceToBook', 0)
            
            logger.info(f"Financial ratios calculated: {list(ratios.keys())}")
            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {str(e)}")
            return {
                'current_ratio': 0.0,
                'debt_to_equity': 0.0,
                'roe': 0.0,
                'roa': 0.0,
                'pe_ratio': 0.0,
                'pb_ratio': 0.0
            }
    
    def _calculate_financial_health_score(self, eps: float, net_profit_margin: float, ratios: Dict[str, float]) -> float:
        """
        Calculate overall financial health score (0-100)
        
        Args:
            eps: Earnings per share
            net_profit_margin: Net profit margin percentage
            ratios: Dictionary of financial ratios
            
        Returns:
            Financial health score (0-100)
        """
        try:
            score = 0.0
            
            # EPS component (25 points)
            if eps > 0:
                score += min(25, eps * 5)  # Higher EPS = better score
            
            # Profit margin component (25 points)
            if net_profit_margin > 0:
                score += min(25, net_profit_margin * 2)  # Higher margin = better score
            
            # Current ratio component (15 points)
            current_ratio = ratios.get('current_ratio', 0)
            if current_ratio >= 1.5:
                score += 15
            elif current_ratio >= 1.0:
                score += 10
            elif current_ratio >= 0.5:
                score += 5
            
            # Debt to equity component (15 points)
            debt_to_equity = ratios.get('debt_to_equity', 0)
            if debt_to_equity <= 0.3:
                score += 15
            elif debt_to_equity <= 0.5:
                score += 10
            elif debt_to_equity <= 1.0:
                score += 5
            
            # ROE component (20 points)
            roe = ratios.get('roe', 0)
            if roe >= 15:
                score += 20
            elif roe >= 10:
                score += 15
            elif roe >= 5:
                score += 10
            elif roe > 0:
                score += 5
            
            logger.info(f"Financial health score: {score:.1f}/100")
            return min(100, max(0, score))
            
        except Exception as e:
            logger.error(f"Error calculating financial health score: {str(e)}")
            return 50.0  # Neutral score as fallback
    
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is still valid"""
        if ticker not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[ticker]
        return cache_age.total_seconds() < (self.cache_duration * 3600)
    
    def _cache_results(self, ticker: str, metrics: FundamentalMetrics):
        """Cache the fundamental metrics"""
        self.cache[ticker] = metrics
        self.cache_timestamps[ticker] = datetime.now()
    
    def _get_fallback_metrics(self, ticker: str) -> FundamentalMetrics:
        """Get fallback metrics when data is unavailable"""
        logger.warning(f"Using fallback metrics for {ticker}")
        return FundamentalMetrics(
            eps=0.0,
            eps_growth=0.0,
            net_profit_margin=0.0,
            revenue_growth=0.0,
            dividend_yield=0.0,
            dividend_announcement="Unknown",
            debt_to_equity=0.0,
            current_ratio=0.0,
            roe=0.0,
            roa=0.0,
            pe_ratio=0.0,
            pb_ratio=0.0,
            financial_health_score=50.0,
            last_updated=datetime.now()
        )
    
    def save_fundamental_data(self, ticker: str, output_dir: str = "data/fundamental"):
        """
        Save fundamental metrics to CSV file
        
        Args:
            ticker: Stock ticker
            output_dir: Output directory for data files
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            metrics = self.get_fundamental_metrics(ticker)
            
            # Convert to DataFrame
            data = {
                'metric': [
                    'eps', 'eps_growth', 'net_profit_margin', 'revenue_growth',
                    'dividend_yield', 'dividend_announcement', 'debt_to_equity',
                    'current_ratio', 'roe', 'roa', 'pe_ratio', 'pb_ratio',
                    'financial_health_score', 'last_updated'
                ],
                'value': [
                    metrics.eps, metrics.eps_growth, metrics.net_profit_margin,
                    metrics.revenue_growth, metrics.dividend_yield,
                    metrics.dividend_announcement, metrics.debt_to_equity,
                    metrics.current_ratio, metrics.roe, metrics.roa,
                    metrics.pe_ratio, metrics.pb_ratio, metrics.financial_health_score,
                    metrics.last_updated.strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            
            df = pd.DataFrame(data)
            filename = f"{output_dir}/{ticker}_fundamental_metrics.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Fundamental data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving fundamental data: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    analyzer = FundamentalAnalyzer()
    
    # Test with AAPL
    print("Testing Fundamental Analyzer with AAPL...")
    metrics = analyzer.get_fundamental_metrics("AAPL")
    
    print(f"\n📊 Fundamental Analysis Results for AAPL:")
    print(f"EPS: ${metrics.eps:.2f}")
    print(f"EPS Growth: {metrics.eps_growth:.2f}%")
    print(f"Net Profit Margin: {metrics.net_profit_margin:.2f}%")
    print(f"Revenue Growth: {metrics.revenue_growth:.2f}%")
    print(f"Dividend Yield: {metrics.dividend_yield:.2f}%")
    print(f"Dividend Status: {metrics.dividend_announcement}")
    print(f"Financial Health Score: {metrics.financial_health_score:.1f}/100")
    
    # Save data
    analyzer.save_fundamental_data("AAPL")
