import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class EnhancedMarketFactors:
    """
    Enhanced market factors analyzer for comprehensive share price prediction.
    """
    
    def __init__(self):
        self.factors = {}
        
    def get_macroeconomic_factors(self, country='US'):
        """Get macroeconomic indicators."""
        factors = {}
        
        try:
            # Interest rates (10-year Treasury yield)
            treasury = yf.download('^TNX', period='1y')
            factors['interest_rate'] = treasury['Close'].iloc[-1] if not treasury.empty else None
            
            # Inflation (CPI data would need external API)
            # GDP growth (would need external API)
            # Unemployment rate (would need external API)
            
        except Exception as e:
            print(f"Error fetching macroeconomic data: {e}")
            
        return factors
    
    def get_valuation_metrics(self, ticker):
        """Get comprehensive valuation metrics."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics = {
                'pe_ratio': info.get('trailingPE'),
                'pb_ratio': info.get('priceToBook'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'dividend_yield': info.get('dividendYield'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio')
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching valuation metrics: {e}")
            return {}
    
    def get_institutional_activity(self, ticker):
        """Get institutional activity indicators."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get institutional holders
            institutional_holders = stock.institutional_holders
            major_holders = stock.major_holders
            
            activity = {
                'institutional_holders': institutional_holders,
                'major_holders': major_holders,
                'institutional_ownership_pct': None,
                'insider_ownership_pct': None
            }
            
            # Calculate ownership percentages
            if institutional_holders is not None and not institutional_holders.empty:
                activity['institutional_ownership_pct'] = institutional_holders['% Out'].sum()
            
            if major_holders is not None and not major_holders.empty:
                # Look for insider ownership
                insider_row = major_holders[major_holders[1].str.contains('insider', case=False, na=False)]
                if not insider_row.empty:
                    activity['insider_ownership_pct'] = insider_row.iloc[0]['% Out']
            
            return activity
            
        except Exception as e:
            print(f"Error fetching institutional data: {e}")
            return {}
    
    def get_market_microstructure(self, ticker, period='1mo'):
        """Get market microstructure data."""
        try:
            # Get intraday data for microstructure analysis
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval='1h')
            
            if hist.empty:
                return {}
            
            # Calculate microstructure metrics
            microstructure = {
                'avg_daily_volume': hist['Volume'].mean(),
                'volume_volatility': hist['Volume'].std() / hist['Volume'].mean(),
                'price_volatility': hist['Close'].pct_change().std(),
                'avg_daily_range': (hist['High'] - hist['Low']).mean(),
                'volume_price_correlation': hist['Volume'].corr(hist['Close'])
            }
            
            return microstructure
            
        except Exception as e:
            print(f"Error calculating microstructure metrics: {e}")
            return {}
    
    def get_sector_comparison(self, ticker):
        """Compare stock performance to sector."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Get sector ETF for comparison
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Industrials': 'XLI',
                'Energy': 'XLE',
                'Basic Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Consumer Defensive': 'XLP',
                'Utilities': 'XLU',
                'Communication Services': 'XLC'
            }
            
            sector_etf = sector_etfs.get(sector)
            if sector_etf:
                sector_data = yf.download(sector_etf, period='1y')
                stock_data = yf.download(ticker, period='1y')
                
                if not sector_data.empty and not stock_data.empty:
                    # Calculate relative performance
                    stock_return = (stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0]) - 1
                    sector_return = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0]) - 1
                    
                    comparison = {
                        'sector': sector,
                        'industry': industry,
                        'stock_return_1y': stock_return,
                        'sector_return_1y': sector_return,
                        'relative_performance': stock_return - sector_return,
                        'sector_etf': sector_etf
                    }
                    
                    return comparison
            
            return {'sector': sector, 'industry': industry}
            
        except Exception as e:
            print(f"Error in sector comparison: {e}")
            return {}
    
    def get_enhanced_financial_ratios(self, ticker):
        """Get enhanced financial ratios."""
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            balance_sheet = stock.balance_sheet
            income_stmt = stock.financials
            cash_flow = stock.cashflow
            
            ratios = {}
            
            if balance_sheet is not None and not balance_sheet.empty:
                # Liquidity ratios
                current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
                current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
                ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else None
                
                # Leverage ratios
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                ratios['debt_to_equity'] = total_debt / total_equity if total_equity != 0 else None
            
            if income_stmt is not None and not income_stmt.empty:
                # Profitability ratios
                net_income = income_stmt.loc['Net Income'].iloc[0]
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if balance_sheet is not None else None
                ratios['roa'] = net_income / total_assets if total_assets and total_assets != 0 else None
                
                # Growth ratios
                if len(income_stmt.columns) > 1:
                    revenue_current = income_stmt.loc['Total Revenue'].iloc[0]
                    revenue_previous = income_stmt.loc['Total Revenue'].iloc[1]
                    ratios['revenue_growth'] = (revenue_current - revenue_previous) / revenue_previous if revenue_previous != 0 else None
            
            return ratios
            
        except Exception as e:
            print(f"Error calculating financial ratios: {e}")
            return {}
    
    def get_comprehensive_factors(self, ticker):
        """Get all comprehensive factors for a stock."""
        print(f"🔍 Analyzing comprehensive factors for {ticker}...")
        
        factors = {
            'ticker': ticker,
            'timestamp': datetime.now(),
            'valuation': self.get_valuation_metrics(ticker),
            'financial_ratios': self.get_enhanced_financial_ratios(ticker),
            'institutional_activity': self.get_institutional_activity(ticker),
            'market_microstructure': self.get_market_microstructure(ticker),
            'sector_comparison': self.get_sector_comparison(ticker),
            'macroeconomic': self.get_macroeconomic_factors()
        }
        
        # Calculate composite score
        factors['composite_score'] = self._calculate_composite_score(factors)
        
        return factors
    
    def _calculate_composite_score(self, factors):
        """Calculate composite factor score."""
        score = 0.0
        weights = {
            'valuation': 0.25,
            'financial_ratios': 0.25,
            'sector_comparison': 0.20,
            'institutional_activity': 0.15,
            'market_microstructure': 0.10,
            'macroeconomic': 0.05
        }
        
        # Valuation score
        if factors['valuation']:
            pe_ratio = factors['valuation'].get('pe_ratio')
            if pe_ratio and pe_ratio > 0:
                # Lower P/E is better (inverse relationship)
                pe_score = max(0, 1 - (pe_ratio / 50))  # Normalize to 0-1
                score += pe_score * weights['valuation']
        
        # Financial ratios score
        if factors['financial_ratios']:
            current_ratio = factors['financial_ratios'].get('current_ratio')
            if current_ratio:
                # Higher current ratio is better
                cr_score = min(1, current_ratio / 3)  # Normalize to 0-1
                score += cr_score * weights['financial_ratios']
        
        # Sector comparison score
        if factors['sector_comparison']:
            rel_performance = factors['sector_comparison'].get('relative_performance')
            if rel_performance is not None:
                # Positive relative performance is better
                sector_score = max(0, min(1, (rel_performance + 0.5) / 1.0))  # Normalize to 0-1
                score += sector_score * weights['sector_comparison']
        
        return score

# Enhanced integration with existing system
def integrate_enhanced_factors(df, ticker):
    """Integrate enhanced factors into existing dataframe."""
    analyzer = EnhancedMarketFactors()
    factors = analyzer.get_comprehensive_factors(ticker)
    
    # Add factor columns to dataframe
    df['Valuation_Score'] = factors.get('composite_score', 0.5)
    df['PE_Ratio'] = factors.get('valuation', {}).get('pe_ratio', np.nan)
    df['Current_Ratio'] = factors.get('financial_ratios', {}).get('current_ratio', np.nan)
    df['Sector_Relative_Performance'] = factors.get('sector_comparison', {}).get('relative_performance', 0)
    df['Institutional_Ownership_Pct'] = factors.get('institutional_activity', {}).get('institutional_ownership_pct', np.nan)
    
    return df, factors

    def get_market_factors(self) -> Dict:
        """
        Get market factors for analysis.
        
        Returns:
            Dictionary with market factors
        """
        try:
            # Get VIX (volatility index)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="5d")
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0
            
            # Get S&P 500 for market performance
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="30d")
            spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0] - 1) * 100 if not spy_data.empty else 0.0
            
            # Get 10-year Treasury yield
            treasury = yf.Ticker("^TNX")
            treasury_data = treasury.history(period="5d")
            treasury_yield = treasury_data['Close'].iloc[-1] if not treasury_data.empty else 4.0
            
            # Get dollar index
            dollar = yf.Ticker("UUP")
            dollar_data = dollar.history(period="5d")
            dollar_index = dollar_data['Close'].iloc[-1] if not dollar_data.empty else 100.0
            
            # Get oil price
            oil = yf.Ticker("USO")
            oil_data = oil.history(period="5d")
            oil_price = oil_data['Close'].iloc[-1] if not oil_data.empty else 75.0
            
            # Get gold price
            gold = yf.Ticker("GLD")
            gold_data = gold.history(period="5d")
            gold_price = gold_data['Close'].iloc[-1] if not gold_data.empty else 195.0
            
            # Calculate market sentiment
            market_sentiment = 'bullish' if spy_return > 2 else 'bearish' if spy_return < -2 else 'neutral'
            
            # Calculate volatility regime
            volatility_regime = 'high' if current_vix > 25 else 'low' if current_vix < 15 else 'normal'
            
            return {
                'vix_level': current_vix,
                'spy_return_30d': spy_return,
                'treasury_10y_yield': treasury_yield,
                'dollar_index': dollar_index,
                'oil_price': oil_price,
                'gold_price': gold_price,
                'market_sentiment': market_sentiment,
                'volatility_regime': volatility_regime,
                'market_trend': 'up' if spy_return > 0 else 'down',
                'risk_appetite': 'high' if current_vix < 15 else 'low' if current_vix > 25 else 'normal',
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error fetching market factors: {e}")
            return {
                'vix_level': 20.0,
                'spy_return_30d': 0.0,
                'treasury_10y_yield': 4.0,
                'dollar_index': 100.0,
                'oil_price': 75.0,
                'gold_price': 195.0,
                'market_sentiment': 'neutral',
                'volatility_regime': 'normal',
                'market_trend': 'neutral',
                'risk_appetite': 'normal',
                'analysis_date': datetime.now().isoformat(),
                'error': str(e)
            }
