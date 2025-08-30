#!/usr/bin/env python3
"""
Core Strategy Service
Centralized service for all strategy-related functionality including sentiment analysis,
trading signals, market factors, economic indicators, balance sheet analysis, and backtesting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from typing import Dict, List, Optional, Tuple, Any
import requests
import json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

warnings.filterwarnings('ignore')

class StrategyService:
    """
    Core strategy service that provides all strategy-related functionality:
    - Sentiment analysis
    - Trading signals generation
    - Market factors analysis
    - Economic indicators
    - Balance sheet analysis
    - Company event impact
    - Backtesting
    - Sector trend analysis
    """
    
    def __init__(self):
        """Initialize the strategy service."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nltk_analyzer = NLTKSentimentAnalyzer()
        
    def analyze_sentiment(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze stock sentiment using multiple sources.
        
        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back for sentiment data
            
        Returns:
            DataFrame with sentiment analysis results
        """
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            
            # Get news headlines (simulated for now)
            headlines = self._get_sample_headlines(ticker, days_back)
            
            # Analyze sentiment for each headline
            sentiment_data = []
            for headline in headlines:
                # VADER sentiment
                vader_scores = self.sentiment_analyzer.polarity_scores(headline['headline'])
                
                # TextBlob sentiment
                blob = TextBlob(headline['headline'])
                textblob_polarity = blob.sentiment.polarity
                textblob_subjectivity = blob.sentiment.subjectivity
                
                sentiment_data.append({
                    'date': headline['date'],
                    'headline': headline['headline'],
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu'],
                    'textblob_polarity': textblob_polarity,
                    'textblob_subjectivity': textblob_subjectivity
                })
            
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Calculate aggregate sentiment metrics
                aggregate_sentiment = {
                    'ticker': ticker,
                    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_headlines': len(df),
                    'avg_vader_compound': df['vader_compound'].mean(),
                    'avg_textblob_polarity': df['textblob_polarity'].mean(),
                    'avg_textblob_subjectivity': df['textblob_subjectivity'].mean(),
                    'sentiment_score': (df['vader_compound'].mean() + df['textblob_polarity'].mean()) / 2,
                    'sentiment_label': self._get_sentiment_label(df['vader_compound'].mean())
                }
                
                return pd.DataFrame([aggregate_sentiment])
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return pd.DataFrame()
    
    def generate_trading_signals(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators and sentiment.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with price data and technical indicators
            
        Returns:
            DataFrame with trading signals
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            signals = []
            
            # Calculate technical signals
            for i in range(1, len(df)):
                signal = self._calculate_technical_signal(df, i)
                if signal:
                    signals.append(signal)
            
            if signals:
                signals_df = pd.DataFrame(signals)
                signals_df['date'] = pd.to_datetime(signals_df['date'])
                signals_df.set_index('date', inplace=True)
                return signals_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Trading signals generation error: {e}")
            return pd.DataFrame()
    
    def get_market_factors(self, ticker: str = None) -> Dict[str, Any]:
        """
        Get comprehensive market factors and economic indicators.
        
        Args:
            ticker: Stock ticker symbol (optional)
            
        Returns:
            Dictionary with market factors and economic indicators
        """
        try:
            market_data = {}
            
            # Get VIX (volatility index)
            try:
                vix = yf.Ticker('^VIX')
                vix_info = vix.info
                market_data['vix_current'] = vix_info.get('regularMarketPrice', 0)
                market_data['vix_change'] = vix_info.get('regularMarketChange', 0)
            except:
                market_data['vix_current'] = 0
                market_data['vix_change'] = 0
            
            # Get S&P 500 data
            try:
                sp500 = yf.Ticker('^GSPC')
                sp500_info = sp500.info
                market_data['sp500_current'] = sp500_info.get('regularMarketPrice', 0)
                market_data['sp500_change'] = sp500_info.get('regularMarketChange', 0)
            except:
                market_data['sp500_current'] = 0
                market_data['sp500_change'] = 0
            
            # Get economic indicators (simulated)
            market_data.update(self._get_simulated_economic_indicators())
            
            # Add timestamp
            market_data['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            market_data['ticker'] = ticker if ticker else 'GLOBAL'
            
            return market_data
            
        except Exception as e:
            print(f"Market factors error: {e}")
            return {}
    
    def analyze_balance_sheet(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze company balance sheet and financial ratios.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with financial analysis results
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial data
            balance_sheet = stock.balance_sheet
            income_stmt = stock.income_stmt
            cash_flow = stock.cashflow
            
            if balance_sheet is None or income_stmt is None:
                return self._get_simulated_financial_data(ticker)
            
            # Calculate financial ratios
            analysis = {}
            
            # Current ratio
            if 'Total Current Assets' in balance_sheet.index and 'Total Current Liabilities' in balance_sheet.index:
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                analysis['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else 0
            else:
                analysis['current_ratio'] = 1.5  # Default value
            
            # Debt-to-equity ratio
            if 'Total Debt' in balance_sheet.index and 'Total Stockholder Equity' in balance_sheet.index:
                total_debt = balance_sheet.loc['Total Debt'].iloc[0]
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                analysis['debt_to_equity'] = total_debt / total_equity if total_equity != 0 else 0
            else:
                analysis['debt_to_equity'] = 0.5  # Default value
            
            # ROA (Return on Assets)
            if 'Net Income' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                net_income = income_stmt.loc['Net Income'].iloc[0]
                total_assets = balance_sheet.loc['Total Assets'].iloc[0]
                analysis['roa'] = net_income / total_assets if total_assets != 0 else 0
            else:
                analysis['roa'] = 0.05  # Default value
            
            # Revenue growth
            if 'Total Revenue' in income_stmt.index:
                revenue = income_stmt.loc['Total Revenue']
                if len(revenue) >= 2:
                    current_revenue = revenue.iloc[0]
                    previous_revenue = revenue.iloc[1]
                    analysis['revenue_growth'] = (current_revenue - previous_revenue) / previous_revenue if previous_revenue != 0 else 0
                else:
                    analysis['revenue_growth'] = 0.1  # Default value
            else:
                analysis['revenue_growth'] = 0.1  # Default value
            
            # Calculate financial health score
            analysis['financial_health_score'] = self._calculate_financial_health_score(analysis)
            analysis['ticker'] = ticker
            analysis['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return analysis
            
        except Exception as e:
            print(f"Balance sheet analysis error: {e}")
            return self._get_simulated_financial_data(ticker)
    
    def analyze_company_events(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze company events and their potential impact.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with event analysis results
        """
        try:
            # Get sample events (in real implementation, this would fetch from news APIs)
            events = self._get_sample_company_events(ticker)
            
            if not events:
                return {}
            
            # Analyze event impact
            event_analysis = {
                'ticker': ticker,
                'total_events': len(events),
                'positive_events': len([e for e in events if e['sentiment'] == 'positive']),
                'negative_events': len([e for e in events if e['sentiment'] == 'negative']),
                'neutral_events': len([e for e in events if e['sentiment'] == 'neutral']),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Calculate event impact score
            positive_score = event_analysis['positive_events'] * 1
            negative_score = event_analysis['negative_events'] * -1
            event_analysis['event_impact_score'] = (positive_score + negative_score) / len(events) if events else 0
            
            return event_analysis
            
        except Exception as e:
            print(f"Company events analysis error: {e}")
            return {}
    
    def run_backtest(self, signals_df: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Run backtesting on trading signals.
        
        Args:
            signals_df: DataFrame with trading signals
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if signals_df.empty:
                return {}
            
            # Simulate trading based on signals
            capital = initial_capital
            trades = []
            
            for date, signal in signals_df.iterrows():
                if signal['signal'] == 'BUY' and capital > 0:
                    # Simulate buy trade
                    trade_amount = capital * 0.1  # Use 10% of capital per trade
                    capital -= trade_amount
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'amount': trade_amount,
                        'price': signal.get('entry_price', 100)
                    })
                elif signal['signal'] == 'SELL' and len(trades) > 0:
                    # Simulate sell trade
                    last_trade = trades[-1]
                    if last_trade['action'] == 'BUY':
                        profit = (signal.get('entry_price', 100) - last_trade['price']) / last_trade['price']
                        capital += last_trade['amount'] * (1 + profit)
                        trades.append({
                            'date': date,
                            'action': 'SELL',
                            'amount': last_trade['amount'],
                            'price': signal.get('entry_price', 100),
                            'profit': profit
                        })
            
            # Calculate performance metrics
            total_return = (capital - initial_capital) / initial_capital if initial_capital > 0 else 0
            winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
            total_trades = len([t for t in trades if t.get('profit') is not None])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': total_return,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': win_rate,
                'sharpe_ratio': self._calculate_sharpe_ratio(trades),
                'max_drawdown': self._calculate_max_drawdown(trades),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return results
            
        except Exception as e:
            print(f"Backtesting error: {e}")
            return {}
    
    def analyze_sector_trends(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze sector trends and compare with the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sector analysis results
        """
        try:
            # Get sector information (simulated)
            sector_data = self._get_simulated_sector_data(ticker)
            
            sector_analysis = {
                'ticker': ticker,
                'sector': sector_data.get('sector', 'Technology'),
                'sector_performance': sector_data.get('performance', 0.05),
                'sector_rank': sector_data.get('rank', 5),
                'sector_momentum': sector_data.get('momentum', 'positive'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return sector_analysis
            
        except Exception as e:
            print(f"Sector trends analysis error: {e}")
            return {}
    
    # Helper methods
    
    def _get_sample_headlines(self, ticker: str, days_back: int) -> List[Dict]:
        """Get sample headlines for sentiment analysis."""
        headlines = [
            {
                'date': datetime.now() - timedelta(days=1),
                'headline': f"{ticker} reports strong quarterly earnings"
            },
            {
                'date': datetime.now() - timedelta(days=2),
                'headline': f"{ticker} announces new product launch"
            },
            {
                'date': datetime.now() - timedelta(days=3),
                'headline': f"{ticker} faces regulatory challenges"
            },
            {
                'date': datetime.now() - timedelta(days=4),
                'headline': f"{ticker} expands into new markets"
            },
            {
                'date': datetime.now() - timedelta(days=5),
                'headline': f"{ticker} CEO announces retirement"
            }
        ]
        return headlines[:days_back]
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Get sentiment label based on compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_technical_signal(self, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """Calculate technical trading signal."""
        try:
            if index < 1:
                return None
            
            current_row = df.iloc[index]
            prev_row = df.iloc[index-1]
            
            signal = None
            
            # Simple moving average crossover
            if 'SMA_10' in df.columns and 'SMA_20' in df.columns:
                if (prev_row['SMA_10'] <= prev_row['SMA_20'] and 
                    current_row['SMA_10'] > current_row['SMA_20']):
                    signal = 'BUY'
                elif (prev_row['SMA_10'] >= prev_row['SMA_20'] and 
                      current_row['SMA_10'] < current_row['SMA_20']):
                    signal = 'SELL'
            
            # RSI signals
            if 'RSI_14' in df.columns:
                if current_row['RSI_14'] < 30:
                    signal = 'BUY'
                elif current_row['RSI_14'] > 70:
                    signal = 'SELL'
            
            if signal:
                return {
                    'date': df.index[index],
                    'signal': signal,
                    'entry_price': current_row['Close'],
                    'stop_loss': current_row['Close'] * 0.95 if signal == 'BUY' else current_row['Close'] * 1.05,
                    'take_profit': current_row['Close'] * 1.10 if signal == 'BUY' else current_row['Close'] * 0.90,
                    'position_size_pct': 0.02
                }
            
            return None
            
        except Exception as e:
            print(f"Technical signal calculation error: {e}")
            return None
    
    def _get_simulated_economic_indicators(self) -> Dict[str, Any]:
        """Get simulated economic indicators."""
        return {
            'gdp_growth': 0.025,
            'inflation_rate': 0.03,
            'unemployment_rate': 0.045,
            'interest_rate': 0.05,
            'consumer_confidence': 0.65,
            'manufacturing_pmi': 52.5,
            'retail_sales_growth': 0.04
        }
    
    def _get_simulated_financial_data(self, ticker: str) -> Dict[str, Any]:
        """Get simulated financial data when real data is unavailable."""
        return {
            'ticker': ticker,
            'current_ratio': 1.5,
            'debt_to_equity': 0.5,
            'roa': 0.05,
            'revenue_growth': 0.1,
            'financial_health_score': 75,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_financial_health_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate financial health score based on ratios."""
        score = 0
        
        # Current ratio scoring
        current_ratio = analysis.get('current_ratio', 1.5)
        if current_ratio >= 2.0:
            score += 25
        elif current_ratio >= 1.5:
            score += 20
        elif current_ratio >= 1.0:
            score += 15
        else:
            score += 5
        
        # Debt-to-equity scoring
        debt_to_equity = analysis.get('debt_to_equity', 0.5)
        if debt_to_equity <= 0.3:
            score += 25
        elif debt_to_equity <= 0.5:
            score += 20
        elif debt_to_equity <= 1.0:
            score += 15
        else:
            score += 5
        
        # ROA scoring
        roa = analysis.get('roa', 0.05)
        if roa >= 0.1:
            score += 25
        elif roa >= 0.05:
            score += 20
        elif roa >= 0.02:
            score += 15
        else:
            score += 5
        
        # Revenue growth scoring
        revenue_growth = analysis.get('revenue_growth', 0.1)
        if revenue_growth >= 0.15:
            score += 25
        elif revenue_growth >= 0.1:
            score += 20
        elif revenue_growth >= 0.05:
            score += 15
        else:
            score += 5
        
        return score
    
    def _get_sample_company_events(self, ticker: str) -> List[Dict]:
        """Get sample company events."""
        return [
            {
                'date': datetime.now() - timedelta(days=5),
                'event_type': 'earnings',
                'sentiment': 'positive',
                'impact': 'high'
            },
            {
                'date': datetime.now() - timedelta(days=2),
                'event_type': 'product_launch',
                'sentiment': 'positive',
                'impact': 'medium'
            },
            {
                'date': datetime.now() - timedelta(days=1),
                'event_type': 'regulatory',
                'sentiment': 'negative',
                'impact': 'low'
            }
        ]
    
    def _calculate_sharpe_ratio(self, trades: List[Dict]) -> float:
        """Calculate Sharpe ratio from trades."""
        if not trades:
            return 0
        
        profits = [t.get('profit', 0) for t in trades if t.get('profit') is not None]
        if not profits:
            return 0
        
        avg_return = np.mean(profits)
        std_return = np.std(profits)
        
        return avg_return / std_return if std_return != 0 else 0
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trades."""
        if not trades:
            return 0
        
        cumulative_returns = []
        cumulative = 1.0
        
        for trade in trades:
            if trade.get('profit') is not None:
                cumulative *= (1 + trade['profit'])
                cumulative_returns.append(cumulative)
        
        if not cumulative_returns:
            return 0
        
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _get_simulated_sector_data(self, ticker: str) -> Dict[str, Any]:
        """Get simulated sector data."""
        return {
            'sector': 'Technology',
            'performance': 0.05,
            'rank': 5,
            'momentum': 'positive'
        }
    
    def get_technical_indicators(self, ticker: str) -> Dict[str, Any]:
        """
        Get technical indicators for a stock ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing technical indicators
        """
        try:
            # Get stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="60d")
            
            if hist.empty:
                logger.warning(f"No historical data available for {ticker}")
                return self._get_fallback_technical_indicators(ticker)
            
            # Calculate technical indicators
            indicators = {}
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Moving averages
            indicators['sma_20'] = hist['Close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # MACD
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
            
            # Bollinger Bands
            sma_20 = hist['Close'].rolling(window=20).mean()
            std_20 = hist['Close'].rolling(window=20).std()
            indicators['bb_upper'] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
            indicators['bb_middle'] = sma_20.iloc[-1]
            indicators['bb_lower'] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)
            indicators['bb_position'] = (hist['Close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # Volume indicators
            indicators['volume_sma'] = hist['Volume'].rolling(window=20).mean().iloc[-1]
            indicators['volume_ratio'] = hist['Volume'].iloc[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1.0
            
            # Price momentum
            indicators['price_momentum'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100 if len(hist) >= 20 else 0
            
            # Volatility
            indicators['volatility'] = hist['Close'].pct_change().rolling(window=20).std().iloc[-1] * 100
            
            # VIX (simulated)
            indicators['vix_current'] = 20.0 + np.random.normal(0, 5)
            
            logger.info(f"Technical indicators calculated for {ticker}")
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
            return self._get_fallback_technical_indicators(ticker)
    
    def _get_fallback_technical_indicators(self, ticker: str) -> Dict[str, Any]:
        """Get fallback technical indicators when calculation fails."""
        return {
            'rsi': 50.0,
            'sma_20': 0.0,
            'sma_50': 0.0,
            'macd': 0.0,
            'macd_signal': 0.0,
            'macd_histogram': 0.0,
            'bb_upper': 0.0,
            'bb_middle': 0.0,
            'bb_lower': 0.0,
            'bb_position': 0.5,
            'volume_sma': 0.0,
            'volume_ratio': 1.0,
            'price_momentum': 0.0,
            'volatility': 0.0,
            'vix_current': 20.0
        }
