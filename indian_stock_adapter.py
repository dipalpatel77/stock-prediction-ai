#!/usr/bin/env python3
"""
Indian Stock Market Adapter
Helps the AI Stock Predictor work with Indian stocks (NSE/BSE)
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional

class IndianStockAdapter:
    """
    Adapter for Indian stock market data and sentiment analysis.
    """
    
    def __init__(self):
        """
        Initialize Indian stock adapter.
        """
        # Common Indian stock suffixes
        self.nse_suffix = ".NS"  # NSE suffix
        self.bse_suffix = ".BO"  # BSE suffix
        
        # Major Indian companies mapping
        self.indian_companies = {
            # NSE Symbols
            'RELIANCE': 'Reliance Industries Limited',
            'TCS': 'Tata Consultancy Services Limited',
            'HDFCBANK': 'HDFC Bank Limited',
            'INFY': 'Infosys Limited',
            'ICICIBANK': 'ICICI Bank Limited',
            'HINDUNILVR': 'Hindustan Unilever Limited',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel Limited',
            'KOTAKBANK': 'Kotak Mahindra Bank Limited',
            'AXISBANK': 'Axis Bank Limited',
            'ASIANPAINT': 'Asian Paints Limited',
            'MARUTI': 'Maruti Suzuki India Limited',
            'HCLTECH': 'HCL Technologies Limited',
            'WIPRO': 'Wipro Limited',
            'ULTRACEMCO': 'UltraTech Cement Limited',
            'SUNPHARMA': 'Sun Pharmaceutical Industries Limited',
            'TITAN': 'Titan Company Limited',
            'BAJFINANCE': 'Bajaj Finance Limited',
            'NESTLEIND': 'Nestle India Limited',
            
            # BSE Symbols (with .BO suffix)
            '500325': 'Reliance Industries Limited',  # RELIANCE
            '532540': 'Tata Consultancy Services Limited',  # TCS
            '500180': 'HDFC Bank Limited',  # HDFCBANK
            '500209': 'Infosys Limited',  # INFY
            '532174': 'ICICI Bank Limited',  # ICICIBANK
            '500696': 'Hindustan Unilever Limited',  # HINDUNILVR
            '500875': 'ITC Limited',  # ITC
            '500112': 'State Bank of India',  # SBIN
            '532454': 'Bharti Airtel Limited',  # BHARTIARTL
            '500247': 'Kotak Mahindra Bank Limited',  # KOTAKBANK
            '532215': 'Axis Bank Limited',  # AXISBANK
            '500820': 'Asian Paints Limited',  # ASIANPAINT
            '532500': 'Maruti Suzuki India Limited',  # MARUTI
            '532281': 'HCL Technologies Limited',  # HCLTECH
            '507685': 'Wipro Limited',  # WIPRO
            '532538': 'UltraTech Cement Limited',  # ULTRACEMCO
            '524715': 'Sun Pharmaceutical Industries Limited',  # SUNPHARMA
            '500114': 'Titan Company Limited',  # TITAN
            '500034': 'Bajaj Finance Limited',  # BAJFINANCE
            '500790': 'Nestle India Limited',  # NESTLEIND
        }
    
    def get_stock_symbol(self, symbol: str) -> str:
        """
        Get the correct Yahoo Finance symbol for Indian stocks.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', '500325')
            
        Returns:
            str: Yahoo Finance compatible symbol
        """
        # If it's already a BSE number, add .BO suffix
        if symbol.isdigit() and len(symbol) == 6:
            return f"{symbol}.BO"
        
        # If it's an NSE symbol, add .NS suffix
        if symbol in self.indian_companies:
            return f"{symbol}.NS"
        
        # If it already has a suffix, return as is
        if symbol.endswith(('.NS', '.BO')):
            return symbol
        
        # Default to NSE suffix
        return f"{symbol}.NS"
    
    def get_company_name(self, symbol: str) -> str:
        """
        Get company name for Indian stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            str: Company name
        """
        # Remove suffixes for lookup
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        return self.indian_companies.get(clean_symbol, clean_symbol)
    
    def get_indian_sentiment_keywords(self, symbol: str) -> List[str]:
        """
        Get Indian-specific sentiment keywords for better analysis.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List[str]: Keywords for sentiment analysis
        """
        company_name = self.get_company_name(symbol)
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        keywords = [
            clean_symbol,  # Stock symbol
            company_name,  # Full company name
            company_name.split()[0],  # First word of company name
        ]
        
        # Add sector-specific keywords
        sector_keywords = self._get_sector_keywords(clean_symbol)
        keywords.extend(sector_keywords)
        
        return keywords
    
    def _get_sector_keywords(self, symbol: str) -> List[str]:
        """
        Get sector-specific keywords for Indian stocks.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List[str]: Sector keywords
        """
        # Sector mappings for major Indian stocks
        sector_map = {
            # Banking & Finance
            'HDFCBANK': ['HDFC Bank', 'banking', 'finance', 'NPA', 'credit growth'],
            'ICICIBANK': ['ICICI Bank', 'banking', 'finance', 'NPA', 'credit growth'],
            'SBIN': ['SBI', 'State Bank', 'banking', 'PSU bank', 'government bank'],
            'KOTAKBANK': ['Kotak Bank', 'banking', 'private bank'],
            'AXISBANK': ['Axis Bank', 'banking', 'private bank'],
            'BAJFINANCE': ['Bajaj Finance', 'NBFC', 'consumer finance'],
            
            # IT & Technology
            'TCS': ['TCS', 'Tata Consultancy', 'IT services', 'digital transformation'],
            'INFY': ['Infosys', 'IT services', 'digital transformation'],
            'HCLTECH': ['HCL Tech', 'IT services', 'digital transformation'],
            'WIPRO': ['Wipro', 'IT services', 'digital transformation'],
            
            # Oil & Gas
            'RELIANCE': ['Reliance', 'Jio', 'oil', 'gas', 'petrochemicals', 'retail'],
            
            # Consumer Goods
            'HINDUNILVR': ['HUL', 'Hindustan Unilever', 'FMCG', 'consumer goods'],
            'ITC': ['ITC', 'cigarettes', 'FMCG', 'hotels', 'agri business'],
            'NESTLEIND': ['Nestle', 'FMCG', 'consumer goods', 'food'],
            'MARUTI': ['Maruti', 'Suzuki', 'automobile', 'cars', 'auto'],
            'TITAN': ['Titan', 'watches', 'jewelry', 'consumer goods'],
            
            # Telecom
            'BHARTIARTL': ['Bharti Airtel', 'telecom', 'mobile', '4G', '5G'],
            
            # Manufacturing
            'ASIANPAINT': ['Asian Paints', 'paints', 'manufacturing'],
            'ULTRACEMCO': ['UltraTech Cement', 'cement', 'construction'],
            'SUNPHARMA': ['Sun Pharma', 'pharmaceuticals', 'drugs', 'medicine'],
        }
        
        return sector_map.get(symbol, [])
    
    def get_indian_market_hours(self) -> Dict[str, str]:
        """
        Get Indian market trading hours.
        
        Returns:
            Dict[str, str]: Market hours information
        """
        return {
            'nse_pre_market': '09:00-09:08',
            'nse_trading': '09:15-15:30',
            'nse_post_market': '15:40-16:00',
            'bse_pre_market': '09:00-09:08',
            'bse_trading': '09:15-15:30',
            'bse_post_market': '15:40-16:00',
            'timezone': 'IST (UTC+5:30)'
        }
    
    def validate_indian_stock(self, symbol: str) -> bool:
        """
        Validate if the symbol is a valid Indian stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            bool: True if valid Indian stock
        """
        clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        # Check if it's in our known companies
        if clean_symbol in self.indian_companies:
            return True
        
        # Check if it's a 6-digit BSE code
        if clean_symbol.isdigit() and len(clean_symbol) == 6:
            return True
        
        # Try to fetch data to validate
        try:
            yf_symbol = self.get_stock_symbol(symbol)
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            return 'regularMarketPrice' in info and info['regularMarketPrice'] is not None
        except:
            return False
    
    def get_indian_stock_info(self, symbol: str) -> Dict:
        """
        Get comprehensive information about an Indian stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict: Stock information
        """
        try:
            yf_symbol = self.get_stock_symbol(symbol)
            stock = yf.Ticker(yf_symbol)
            info = stock.info
            
            return {
                'symbol': symbol,
                'yf_symbol': yf_symbol,
                'company_name': self.get_company_name(symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('regularMarketPrice', 0),
                'volume': info.get('volume', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'exchange': info.get('exchange', 'Unknown'),
                'sentiment_keywords': self.get_indian_sentiment_keywords(symbol)
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'company_name': self.get_company_name(symbol),
                'sentiment_keywords': self.get_indian_sentiment_keywords(symbol)
            }


# Example usage
if __name__ == "__main__":
    adapter = IndianStockAdapter()
    
    # Test with Indian stocks
    test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', '500325', '532540']
    
    print("🇮🇳 Indian Stock Market Adapter Test")
    print("=" * 50)
    
    for stock in test_stocks:
        print(f"\n📊 Testing: {stock}")
        
        if adapter.validate_indian_stock(stock):
            info = adapter.get_indian_stock_info(stock)
            print(f"   ✅ Valid Indian Stock")
            print(f"   📈 Symbol: {info['yf_symbol']}")
            print(f"   🏢 Company: {info['company_name']}")
            print(f"   🏭 Sector: {info.get('sector', 'Unknown')}")
            print(f"   💰 Current Price: ₹{info.get('current_price', 0):.2f}")
            print(f"   🔍 Keywords: {info['sentiment_keywords'][:3]}...")
        else:
            print(f"   ❌ Invalid or not found")
    
    print(f"\n⏰ Indian Market Hours:")
    hours = adapter.get_indian_market_hours()
    for key, value in hours.items():
        print(f"   {key}: {value}")
