#!/usr/bin/env python3
"""
Global Market Service Module
============================

This module provides comprehensive global market tracking including:
- Major global indices (Dow Jones, Nasdaq, FTSE, Nikkei)
- Cross-market correlation analysis
- Global market sentiment indicators
- Market volatility tracking

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
class GlobalMarketData:
    """Data class to store global market information"""
    dow_jones_change: float
    nasdaq_change: float
    ftse_change: float
    nikkei_change: float
    sp500_change: float
    global_volatility: float
    market_correlation: float
    risk_sentiment: str
    last_updated: datetime

class GlobalMarketService:
    """
    Global market tracking and analysis service
    
    Implements critical missing variables from Phase 1:
    - Dow Jones Daily Change
    - Nasdaq Daily Change  
    - FTSE 100 Change
    - Nikkei 225 Change
    - Global market correlation analysis
    """
    
    def __init__(self, cache_duration: int = 1):
        """
        Initialize the global market service
        
        Args:
            cache_duration: Cache duration in hours
        """
        self.cache_duration = cache_duration
        self.cache = {}
        self.cache_timestamps = {}
        
        # Define global indices
        self.global_indices = {
            '^DJI': 'Dow Jones Industrial Average',
            '^IXIC': 'NASDAQ Composite',
            '^FTSE': 'FTSE 100',
            '^N225': 'Nikkei 225',
            '^GSPC': 'S&P 500',
            '^VIX': 'CBOE Volatility Index'
        }
        
    def get_global_market_data(self) -> GlobalMarketData:
        """
        Get comprehensive global market data
        
        Returns:
            GlobalMarketData object with all global market metrics
        """
        try:
            # Check cache first
            if self._is_cache_valid('global'):
                logger.info("Using cached global market data")
                return self.cache['global']
            
            logger.info("Fetching fresh global market data")
            
            # Get data for all indices
            market_data = {}
            for symbol, name in self.global_indices.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    current_price = info.get('regularMarketPrice', 0)
                    previous_close = info.get('regularMarketPreviousClose', current_price)
                    
                    if previous_close > 0:
                        daily_change = ((current_price - previous_close) / previous_close) * 100
                    else:
                        daily_change = 0.0
                    
                    market_data[symbol] = {
                        'name': name,
                        'current_price': current_price,
                        'previous_close': previous_close,
                        'daily_change': daily_change,
                        'volume': info.get('volume', 0)
                    }
                    
                    logger.info(f"{name}: {daily_change:+.2f}%")
                    
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {str(e)}")
                    market_data[symbol] = {
                        'name': name,
                        'current_price': 0,
                        'previous_close': 0,
                        'daily_change': 0.0,
                        'volume': 0
                    }
            
            # Calculate global volatility
            global_volatility = self._calculate_global_volatility(market_data)
            
            # Calculate market correlation
            market_correlation = self._calculate_market_correlation(market_data)
            
            # Determine risk sentiment
            risk_sentiment = self._determine_risk_sentiment(market_data, global_volatility)
            
            # Create global market data object
            global_data = GlobalMarketData(
                dow_jones_change=market_data.get('^DJI', {}).get('daily_change', 0.0),
                nasdaq_change=market_data.get('^IXIC', {}).get('daily_change', 0.0),
                ftse_change=market_data.get('^FTSE', {}).get('daily_change', 0.0),
                nikkei_change=market_data.get('^N225', {}).get('daily_change', 0.0),
                sp500_change=market_data.get('^GSPC', {}).get('daily_change', 0.0),
                global_volatility=global_volatility,
                market_correlation=market_correlation,
                risk_sentiment=risk_sentiment,
                last_updated=datetime.now()
            )
            
            # Cache the results
            self._cache_results('global', global_data)
            
            return global_data
            
        except Exception as e:
            logger.error(f"Error fetching global market data: {str(e)}")
            return self._get_fallback_global_data()
    
    def get_historical_correlation(self, ticker: str, days: int = 30) -> Dict[str, float]:
        """
        Calculate historical correlation between a ticker and global indices
        
        Args:
            ticker: Stock ticker to analyze
            days: Number of days for correlation analysis
            
        Returns:
            Dictionary of correlations with global indices
        """
        try:
            correlations = {}
            
            # Get historical data for the ticker
            ticker_data = yf.Ticker(ticker)
            ticker_history = ticker_data.history(period=f"{days}d")
            
            if ticker_history.empty:
                logger.warning(f"No historical data available for {ticker}")
                return correlations
            
            # Calculate daily returns for ticker
            ticker_returns = ticker_history['Close'].pct_change().dropna()
            
            # Get historical data for global indices
            for symbol, name in self.global_indices.items():
                try:
                    index_data = yf.Ticker(symbol)
                    index_history = index_data.history(period=f"{days}d")
                    
                    if not index_history.empty:
                        index_returns = index_history['Close'].pct_change().dropna()
                        
                        # Align the data
                        aligned_data = pd.concat([ticker_returns, index_returns], axis=1).dropna()
                        
                        if len(aligned_data) > 10:  # Minimum data points for correlation
                            correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                            correlations[name] = correlation
                            logger.info(f"Correlation with {name}: {correlation:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Error calculating correlation with {symbol}: {str(e)}")
                    correlations[name] = 0.0
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error calculating historical correlation: {str(e)}")
            return {}
    
    def get_market_impact_score(self, ticker: str) -> float:
        """
        Calculate the potential impact of global markets on a specific ticker
        
        Args:
            ticker: Stock ticker to analyze
            
        Returns:
            Market impact score (0-100)
        """
        try:
            # Get global market data
            global_data = self.get_global_market_data()
            
            # Get correlations
            correlations = self.get_historical_correlation(ticker)
            
            # Calculate weighted impact score
            impact_score = 0.0
            total_weight = 0.0
            
            # Weight factors for different indices
            weights = {
                'S&P 500': 0.3,
                'NASDAQ Composite': 0.25,
                'Dow Jones Industrial Average': 0.2,
                'FTSE 100': 0.15,
                'Nikkei 225': 0.1
            }
            
            for index_name, correlation in correlations.items():
                if index_name in weights:
                    weight = weights[index_name]
                    total_weight += weight
                    
                    # Get the corresponding change
                    if index_name == 'S&P 500':
                        change = global_data.sp500_change
                    elif index_name == 'NASDAQ Composite':
                        change = global_data.nasdaq_change
                    elif index_name == 'Dow Jones Industrial Average':
                        change = global_data.dow_jones_change
                    elif index_name == 'FTSE 100':
                        change = global_data.ftse_change
                    elif index_name == 'Nikkei 225':
                        change = global_data.nikkei_change
                    else:
                        change = 0.0
                    
                    # Calculate impact (correlation * change * weight)
                    impact = abs(correlation * change * weight)
                    impact_score += impact
            
            # Normalize to 0-100 scale
            if total_weight > 0:
                impact_score = (impact_score / total_weight) * 10  # Scale factor
                impact_score = min(100, max(0, impact_score))
            
            logger.info(f"Market impact score for {ticker}: {impact_score:.1f}/100")
            return impact_score
            
        except Exception as e:
            logger.error(f"Error calculating market impact score: {str(e)}")
            return 50.0  # Neutral score as fallback
    
    def _calculate_global_volatility(self, market_data: Dict) -> float:
        """
        Calculate global market volatility based on index changes
        
        Args:
            market_data: Dictionary of market data for all indices
            
        Returns:
            Global volatility score (0-100)
        """
        try:
            changes = []
            for symbol, data in market_data.items():
                if symbol != '^VIX':  # Exclude VIX from volatility calculation
                    changes.append(abs(data.get('daily_change', 0)))
            
            if changes:
                # Calculate average absolute change
                avg_change = np.mean(changes)
                # Scale to 0-100 (2% average change = 50 volatility)
                volatility = min(100, avg_change * 25)
                return volatility
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating global volatility: {str(e)}")
            return 0.0
    
    def _calculate_market_correlation(self, market_data: Dict) -> float:
        """
        Calculate correlation between major markets
        
        Args:
            market_data: Dictionary of market data for all indices
            
        Returns:
            Average market correlation (-1 to 1)
        """
        try:
            # Get changes for major indices
            major_indices = ['^DJI', '^IXIC', '^GSPC', '^FTSE', '^N225']
            changes = []
            
            for symbol in major_indices:
                if symbol in market_data:
                    changes.append(market_data[symbol].get('daily_change', 0))
            
            if len(changes) >= 2:
                # Calculate correlation matrix
                df = pd.DataFrame(changes).T
                correlation_matrix = df.corr()
                
                # Get average correlation (excluding diagonal)
                avg_correlation = 0.0
                count = 0
                for i in range(len(correlation_matrix)):
                    for j in range(i+1, len(correlation_matrix)):
                        avg_correlation += correlation_matrix.iloc[i, j]
                        count += 1
                
                if count > 0:
                    return avg_correlation / count
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating market correlation: {str(e)}")
            return 0.0
    
    def _determine_risk_sentiment(self, market_data: Dict, volatility: float) -> str:
        """
        Determine overall market risk sentiment
        
        Args:
            market_data: Dictionary of market data
            volatility: Global volatility score
            
        Returns:
            Risk sentiment string
        """
        try:
            # Count positive vs negative changes
            positive_count = 0
            negative_count = 0
            
            for symbol, data in market_data.items():
                if symbol != '^VIX':
                    change = data.get('daily_change', 0)
                    if change > 0:
                        positive_count += 1
                    elif change < 0:
                        negative_count += 1
            
            # Determine sentiment based on volatility and direction
            if volatility > 70:
                if positive_count > negative_count:
                    return "High Risk - Bullish"
                else:
                    return "High Risk - Bearish"
            elif volatility > 40:
                if positive_count > negative_count:
                    return "Moderate Risk - Bullish"
                else:
                    return "Moderate Risk - Bearish"
            else:
                if positive_count > negative_count:
                    return "Low Risk - Bullish"
                else:
                    return "Low Risk - Bearish"
                    
        except Exception as e:
            logger.error(f"Error determining risk sentiment: {str(e)}")
            return "Unknown"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache_timestamps:
            return False
        
        cache_age = datetime.now() - self.cache_timestamps[key]
        return cache_age.total_seconds() < (self.cache_duration * 3600)
    
    def _cache_results(self, key: str, data: GlobalMarketData):
        """Cache the global market data"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
    
    def _get_fallback_global_data(self) -> GlobalMarketData:
        """Get fallback data when global market data is unavailable"""
        logger.warning("Using fallback global market data")
        return GlobalMarketData(
            dow_jones_change=0.0,
            nasdaq_change=0.0,
            ftse_change=0.0,
            nikkei_change=0.0,
            sp500_change=0.0,
            global_volatility=0.0,
            market_correlation=0.0,
            risk_sentiment="Unknown",
            last_updated=datetime.now()
        )
    
    def save_global_market_data(self, output_dir: str = "data/global"):
        """
        Save global market data to CSV file
        
        Args:
            output_dir: Output directory for data files
        """
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            global_data = self.get_global_market_data()
            
            # Convert to DataFrame
            data = {
                'index': ['Dow Jones', 'NASDAQ', 'FTSE 100', 'Nikkei 225', 'S&P 500'],
                'daily_change': [
                    global_data.dow_jones_change,
                    global_data.nasdaq_change,
                    global_data.ftse_change,
                    global_data.nikkei_change,
                    global_data.sp500_change
                ]
            }
            
            df = pd.DataFrame(data)
            filename = f"{output_dir}/global_indices_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Global market data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving global market data: {str(e)}")

# Example usage and testing
if __name__ == "__main__":
    service = GlobalMarketService()
    
    # Test global market data
    print("Testing Global Market Service...")
    global_data = service.get_global_market_data()
    
    print(f"\n🌍 Global Market Data:")
    print(f"Dow Jones: {global_data.dow_jones_change:+.2f}%")
    print(f"NASDAQ: {global_data.nasdaq_change:+.2f}%")
    print(f"FTSE 100: {global_data.ftse_change:+.2f}%")
    print(f"Nikkei 225: {global_data.nikkei_change:+.2f}%")
    print(f"S&P 500: {global_data.sp500_change:+.2f}%")
    print(f"Global Volatility: {global_data.global_volatility:.1f}/100")
    print(f"Market Correlation: {global_data.market_correlation:.3f}")
    print(f"Risk Sentiment: {global_data.risk_sentiment}")
    
    # Test correlation analysis
    print(f"\n📊 Market Impact Analysis for AAPL:")
    impact_score = service.get_market_impact_score("AAPL")
    print(f"Market Impact Score: {impact_score:.1f}/100")
    
    # Save data
    service.save_global_market_data()
