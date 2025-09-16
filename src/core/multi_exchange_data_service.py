#!/usr/bin/env python3
"""
Multi-Exchange Data Service
Enhanced data service that fuses BSE and NSE data for better predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class MultiExchangeDataService:
    """
    Enhanced data service that combines BSE and NSE data for improved predictions.
    """
    
    def __init__(self, use_database: bool = True):
        """Initialize multi-exchange data service."""
        self.use_database = use_database
        self.data_service = None
        self.angel_downloader = None
        
        # Initialize services
        self._initialize_services()
        
        # Fusion parameters
        self.fusion_methods = {
            'price': 'volume_weighted',  # volume_weighted, liquidity_weighted, vwap
            'volume': 'sum',  # sum, max, average
            'technical': 'average'  # average, weighted_average, best_signal
        }
        
        # Arbitrage detection parameters
        self.arbitrage_threshold = 0.5  # 0.5% price difference threshold
        self.min_volume_threshold = 1000  # Minimum volume for arbitrage detection
    
    def _initialize_services(self):
        """Initialize required services."""
        try:
            from .data_service import DataService
            from ..utils.angel_one_data_downloader import AngelOneDataDownloader
            
            self.data_service = DataService(use_database=self.use_database)
            self.angel_downloader = AngelOneDataDownloader()
            
        except Exception as e:
            print(f"⚠️ Error initializing services: {e}")
    
    def load_multi_exchange_data(self, ticker: str, period: str = "1y", 
                                interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Load and fuse data from both BSE and NSE exchanges.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period
            interval: Data interval
            
        Returns:
            Fused DataFrame with multi-exchange data
        """
        try:
            print(f"🔄 Loading multi-exchange data for {ticker}...")
            
            # Get data from both exchanges
            bse_data = self._load_exchange_data(ticker, 'BSE', period, interval)
            nse_data = self._load_exchange_data(ticker, 'NSE', period, interval)
            
            if bse_data is None and nse_data is None:
                print(f"❌ No data available for {ticker} on either exchange")
                return None
            
            # Fuse the data
            if bse_data is not None and nse_data is not None:
                fused_data = self._fuse_exchange_data(bse_data, nse_data, ticker)
                print(f"✅ Fused data from both exchanges: {len(fused_data)} records")
            elif bse_data is not None:
                fused_data = self._enhance_single_exchange_data(bse_data, 'BSE', ticker)
                print(f"✅ Enhanced BSE data: {len(fused_data)} records")
            else:
                fused_data = self._enhance_single_exchange_data(nse_data, 'NSE', ticker)
                print(f"✅ Enhanced NSE data: {len(fused_data)} records")
            
            return fused_data
            
        except Exception as e:
            print(f"❌ Error loading multi-exchange data for {ticker}: {e}")
            return None
    
    def _load_exchange_data(self, ticker: str, exchange: str, 
                           period: str, interval: str) -> Optional[pd.DataFrame]:
        """Load data from specific exchange."""
        try:
            if not self.angel_downloader.authenticate():
                print(f"⚠️ Angel One authentication failed for {exchange}")
                return None
            
            # Get symbol info for the exchange
            symbol_info = self._get_symbol_info_for_exchange(ticker, exchange)
            if not symbol_info:
                print(f"⚠️ Symbol {ticker} not found on {exchange}")
                return None
            
            # Download data
            data = self.angel_downloader.get_historical_data(
                symbol_name=symbol_info['symbol'],
                exchange=exchange,
                interval=self._map_interval(interval),
                days_back=self._get_days_back(period)
            )
            
            if data is not None and not data.empty:
                # Add exchange information
                data['exchange'] = exchange
                data['symbol_token'] = symbol_info['token']
                print(f"✅ Loaded {len(data)} records from {exchange}")
                return data
            else:
                print(f"⚠️ No data received from {exchange}")
                return None
                
        except Exception as e:
            print(f"❌ Error loading {exchange} data for {ticker}: {e}")
            return None
    
    def _get_symbol_info_for_exchange(self, ticker: str, exchange: str) -> Optional[Dict]:
        """Get symbol information for specific exchange."""
        try:
            from ..utils.indian_stock_mapper import load_angel_master, get_symbol_info
            
            master_data = load_angel_master()
            symbol_info = get_symbol_info(ticker, master_data)
            
            if symbol_info and symbol_info['exchange'] == exchange:
                return symbol_info
            else:
                # Try to find the symbol on the specific exchange
                matches = master_data[
                    (master_data['symbol'] == ticker.upper()) & 
                    (master_data['exch_seg'] == exchange)
                ]
                
                if not matches.empty:
                    row = matches.iloc[0]
                    return {
                        'symbol': row['symbol'],
                        'token': row['token'],
                        'exchange': row['exch_seg'],
                        'name': row['name']
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting symbol info for {ticker} on {exchange}: {e}")
            return None
    
    def _fuse_exchange_data(self, bse_data: pd.DataFrame, nse_data: pd.DataFrame, 
                           ticker: str) -> pd.DataFrame:
        """Fuse data from both exchanges."""
        try:
            # Align data by date
            bse_data = bse_data.copy()
            nse_data = nse_data.copy()
            
            # Ensure both have Date index
            if 'Date' not in bse_data.columns:
                bse_data = bse_data.reset_index()
            if 'Date' not in nse_data.columns:
                nse_data = nse_data.reset_index()
            
            # Merge on date
            merged_data = pd.merge(
                bse_data, nse_data, 
                on='Date', 
                suffixes=('_BSE', '_NSE'),
                how='outer'
            )
            
            # Sort by date
            merged_data = merged_data.sort_values('Date').reset_index(drop=True)
            
            # Create fused features
            fused_data = self._create_fused_features(merged_data)
            
            # Add multi-exchange specific features
            fused_data = self._add_arbitrage_features(fused_data)
            fused_data = self._add_market_microstructure_features(fused_data)
            fused_data = self._add_cross_exchange_technical_indicators(fused_data)
            
            # Add metadata
            fused_data['ticker'] = ticker
            fused_data['data_source'] = 'multi_exchange_fusion'
            fused_data['fusion_method'] = self.fusion_methods['price']
            
            return fused_data
            
        except Exception as e:
            print(f"❌ Error fusing exchange data: {e}")
            return bse_data  # Return BSE data as fallback
    
    def _create_fused_features(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """Create fused price and volume features."""
        try:
            df = merged_data.copy()
            
            # Fuse prices using volume-weighted average
            if 'Close_BSE' in df.columns and 'Close_NSE' in df.columns:
                # Handle missing values
                bse_close = df['Close_BSE'].fillna(method='ffill')
                nse_close = df['Close_NSE'].fillna(method='ffill')
                bse_volume = df['Volume_BSE'].fillna(0)
                nse_volume = df['Volume_NSE'].fillna(0)
                
                # Volume-weighted average price
                total_volume = bse_volume + nse_volume
                df['Close'] = np.where(
                    total_volume > 0,
                    (bse_close * bse_volume + nse_close * nse_volume) / total_volume,
                    (bse_close + nse_close) / 2  # Simple average if no volume
                )
                
                # Individual exchange prices
                df['Close_BSE'] = bse_close
                df['Close_NSE'] = nse_close
                
                # Price spread
                df['Price_Spread'] = abs(bse_close - nse_close)
                df['Price_Spread_Pct'] = (df['Price_Spread'] / df['Close']) * 100
            
            # Fuse volumes
            if 'Volume_BSE' in df.columns and 'Volume_NSE' in df.columns:
                df['Volume'] = df['Volume_BSE'].fillna(0) + df['Volume_NSE'].fillna(0)
                df['Volume_BSE'] = df['Volume_BSE'].fillna(0)
                df['Volume_NSE'] = df['Volume_NSE'].fillna(0)
                
                # Volume ratio
                df['Volume_Ratio'] = np.where(
                    df['Volume_NSE'] > 0,
                    df['Volume_BSE'] / df['Volume_NSE'],
                    0
                )
                
                # Dominant exchange
                df['Dominant_Exchange'] = np.where(
                    df['Volume_BSE'] > df['Volume_NSE'],
                    'BSE',
                    'NSE'
                )
            
            # Fuse OHLC data
            for col in ['Open', 'High', 'Low']:
                bse_col = f'{col}_BSE'
                nse_col = f'{col}_NSE'
                
                if bse_col in df.columns and nse_col in df.columns:
                    # Use the fused close price as reference for OHLC
                    if col == 'Open':
                        df[col] = (df[bse_col].fillna(method='ffill') + df[nse_col].fillna(method='ffill')) / 2
                    elif col == 'High':
                        df[col] = df[[bse_col, nse_col]].max(axis=1)
                    elif col == 'Low':
                        df[col] = df[[bse_col, nse_col]].min(axis=1)
            
            return df
            
        except Exception as e:
            print(f"❌ Error creating fused features: {e}")
            return merged_data
    
    def _add_arbitrage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add arbitrage-related features."""
        try:
            if 'Close_BSE' in df.columns and 'Close_NSE' in df.columns:
                # Arbitrage opportunity detection
                df['Arbitrage_Opportunity'] = (
                    df['Price_Spread_Pct'] > self.arbitrage_threshold
                ).astype(int)
                
                # Arbitrage direction
                df['Arbitrage_Direction'] = np.where(
                    df['Close_BSE'] > df['Close_NSE'],
                    1,  # BSE higher
                    -1   # NSE higher
                )
                
                # Arbitrage magnitude
                df['Arbitrage_Magnitude'] = df['Price_Spread_Pct']
                
                # Volume-weighted arbitrage
                if 'Volume_BSE' in df.columns and 'Volume_NSE' in df.columns:
                    total_volume = df['Volume_BSE'] + df['Volume_NSE']
                    df['Volume_Weighted_Arbitrage'] = np.where(
                        total_volume > 0,
                        df['Price_Spread'] * total_volume,
                        0
                    )
            
            return df
            
        except Exception as e:
            print(f"❌ Error adding arbitrage features: {e}")
            return df
    
    def _add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        try:
            if 'Volume_BSE' in df.columns and 'Volume_NSE' in df.columns:
                # Liquidity imbalance
                df['Liquidity_Imbalance'] = abs(df['Volume_BSE'] - df['Volume_NSE'])
                df['Liquidity_Imbalance_Pct'] = (
                    df['Liquidity_Imbalance'] / (df['Volume_BSE'] + df['Volume_NSE'])
                ) * 100
                
                # Market share
                total_volume = df['Volume_BSE'] + df['Volume_NSE']
                df['BSE_Market_Share'] = np.where(
                    total_volume > 0,
                    df['Volume_BSE'] / total_volume,
                    0.5
                )
                df['NSE_Market_Share'] = 1 - df['BSE_Market_Share']
                
                # Volume volatility
                df['Volume_Volatility'] = df['Volume'].rolling(20).std()
                df['BSE_Volume_Volatility'] = df['Volume_BSE'].rolling(20).std()
                df['NSE_Volume_Volatility'] = df['Volume_NSE'].rolling(20).std()
            
            return df
            
        except Exception as e:
            print(f"❌ Error adding microstructure features: {e}")
            return df
    
    def _add_cross_exchange_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-exchange technical indicators."""
        try:
            if 'Close_BSE' in df.columns and 'Close_NSE' in df.columns:
                # Cross-exchange momentum
                df['BSE_Momentum'] = df['Close_BSE'].pct_change(5)
                df['NSE_Momentum'] = df['Close_NSE'].pct_change(5)
                df['Cross_Exchange_Momentum'] = df['BSE_Momentum'] - df['NSE_Momentum']
                
                # Cross-exchange RSI
                df['BSE_RSI'] = self._calculate_rsi(df['Close_BSE'])
                df['NSE_RSI'] = self._calculate_rsi(df['Close_NSE'])
                df['Cross_Exchange_RSI_Diff'] = df['BSE_RSI'] - df['NSE_RSI']
                
                # Cross-exchange moving averages
                df['BSE_SMA_20'] = df['Close_BSE'].rolling(20).mean()
                df['NSE_SMA_20'] = df['Close_NSE'].rolling(20).mean()
                df['Cross_Exchange_SMA_Diff'] = df['BSE_SMA_20'] - df['NSE_SMA_20']
                
                # Convergence/Divergence signals
                df['Price_Convergence'] = (
                    df['Price_Spread_Pct'].rolling(5).mean() < 
                    df['Price_Spread_Pct'].rolling(20).mean()
                ).astype(int)
            
            return df
            
        except Exception as e:
            print(f"❌ Error adding cross-exchange technical indicators: {e}")
            return df
    
    def _enhance_single_exchange_data(self, data: pd.DataFrame, exchange: str, 
                                     ticker: str) -> pd.DataFrame:
        """Enhance single exchange data with multi-exchange features."""
        try:
            df = data.copy()
            
            # Add exchange information
            df['exchange'] = exchange
            df['ticker'] = ticker
            df['data_source'] = f'single_exchange_{exchange}'
            
            # Add placeholder columns for multi-exchange features
            df['Price_Spread'] = 0
            df['Price_Spread_Pct'] = 0
            df['Arbitrage_Opportunity'] = 0
            df['Cross_Exchange_Momentum'] = 0
            df['Liquidity_Imbalance'] = 0
            
            return df
            
        except Exception as e:
            print(f"❌ Error enhancing single exchange data: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _map_interval(self, interval: str) -> str:
        """Map standard interval to Angel One format."""
        mapping = {
            '1d': 'ONE_DAY',
            '1h': 'ONE_HOUR',
            '30m': 'THIRTY_MINUTE',
            '15m': 'FIFTEEN_MINUTE',
            '5m': 'FIVE_MINUTE',
            '1m': 'ONE_MINUTE'
        }
        return mapping.get(interval, 'ONE_DAY')
    
    def _get_days_back(self, period: str) -> int:
        """Get days back from period string."""
        mapping = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730
        }
        return mapping.get(period, 365)
    
    def get_fusion_quality_metrics(self, df: pd.DataFrame) -> Dict:
        """Get quality metrics for fused data."""
        try:
            metrics = {
                'total_records': len(df),
                'bse_records': len(df[df['exchange'] == 'BSE']) if 'exchange' in df.columns else 0,
                'nse_records': len(df[df['exchange'] == 'NSE']) if 'exchange' in df.columns else 0,
                'fusion_coverage': 0,
                'avg_price_spread': 0,
                'arbitrage_opportunities': 0
            }
            
            if 'Price_Spread_Pct' in df.columns:
                metrics['avg_price_spread'] = df['Price_Spread_Pct'].mean()
                metrics['arbitrage_opportunities'] = (df['Price_Spread_Pct'] > self.arbitrage_threshold).sum()
            
            if 'Close_BSE' in df.columns and 'Close_NSE' in df.columns:
                both_exchanges = df[df['Close_BSE'].notna() & df['Close_NSE'].notna()]
                metrics['fusion_coverage'] = len(both_exchanges) / len(df) * 100
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating fusion quality metrics: {e}")
            return {}
