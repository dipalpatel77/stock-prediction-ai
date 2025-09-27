#!/usr/bin/env python3
"""
Technical Indicators Service
Service for calculating technical indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class TechnicalIndicatorsService:
    """Service for calculating technical indicators"""
    
    def __init__(self):
        logger.info("Technical Indicators Service initialized")
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            data: Stock data with OHLCV columns
            
        Returns:
            DataFrame with all technical indicators
        """
        try:
            result = data.copy()
            
            # Moving Averages
            result = self._add_moving_averages(result)
            
            # MACD
            result = self._add_macd(result)
            
            # RSI
            result = self._add_rsi(result)
            
            # Bollinger Bands
            result = self._add_bollinger_bands(result)
            
            # Stochastic Oscillator
            result = self._add_stochastic(result)
            
            # Williams %R
            result = self._add_williams_r(result)
            
            # Average True Range
            result = self._add_atr(result)
            
            # Commodity Channel Index
            result = self._add_cci(result)
            
            # Volume indicators
            result = self._add_volume_indicators(result)
            
            logger.info(f"Calculated technical indicators: {len(result.columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Technical indicators calculation failed: {e}")
            return data
    
    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        try:
            # Simple Moving Averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_10'] = data['Close'].rolling(window=10).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_100'] = data['Close'].rolling(window=100).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            data['EMA_5'] = data['Close'].ewm(span=5).mean()
            data['EMA_10'] = data['Close'].ewm(span=10).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_20'] = data['Close'].ewm(span=20).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            data['EMA_50'] = data['Close'].ewm(span=50).mean()
            
            # Weighted Moving Average
            data['WMA_20'] = data['Close'].rolling(window=20).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1))
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Moving averages calculation failed: {e}")
            return data
    
    def _add_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicators"""
        try:
            # MACD Line
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            
            # Signal Line
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            # Histogram
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            return data
            
        except Exception as e:
            logger.error(f"MACD calculation failed: {e}")
            return data
    
    def _add_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator"""
        try:
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            return data
            
        except Exception as e:
            logger.error(f"RSI calculation failed: {e}")
            return data
    
    def _add_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        try:
            # Middle Band (SMA)
            data['BB_Middle'] = data['Close'].rolling(window=period).mean()
            
            # Standard Deviation
            bb_std = data['Close'].rolling(window=period).std()
            
            # Upper and Lower Bands
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * std_dev)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * std_dev)
            
            # Band Width
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            
            # %B
            data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            return data
            
        except Exception as e:
            logger.error(f"Bollinger Bands calculation failed: {e}")
            return data
    
    def _add_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        try:
            # %K
            lowest_low = data['Low'].rolling(window=k_period).min()
            highest_high = data['High'].rolling(window=k_period).max()
            data['Stoch_K'] = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
            
            # %D
            data['Stoch_D'] = data['Stoch_K'].rolling(window=d_period).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"Stochastic calculation failed: {e}")
            return data
    
    def _add_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R"""
        try:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            data['Williams_R'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
            
            return data
            
        except Exception as e:
            logger.error(f"Williams %R calculation failed: {e}")
            return data
    
    def _add_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range"""
        try:
            # True Range
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data['ATR'] = true_range.rolling(window=period).mean()
            
            return data
            
        except Exception as e:
            logger.error(f"ATR calculation failed: {e}")
            return data
    
    def _add_cci(self, data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        try:
            # Typical Price
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            
            # Simple Moving Average of Typical Price
            sma_tp = typical_price.rolling(window=period).mean()
            
            # Mean Deviation
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            # CCI
            data['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            return data
            
        except Exception as e:
            logger.error(f"CCI calculation failed: {e}")
            return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators"""
        try:
            # Volume Moving Average
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            
            # Volume Ratio
            data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            
            # On-Balance Volume
            data['OBV'] = (data['Volume'] * np.sign(data['Close'].diff())).cumsum()
            
            # Volume Price Trend
            data['VPT'] = (data['Volume'] * data['Close'].pct_change()).cumsum()
            
            # Accumulation/Distribution Line
            clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            clv = clv.fillna(0)
            data['ADL'] = (clv * data['Volume']).cumsum()
            
            return data
            
        except Exception as e:
            logger.error(f"Volume indicators calculation failed: {e}")
            return data
    
    def get_indicator_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get trading signals from technical indicators
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signals = {}
            
            # Moving Average Signals
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                signals['sma_cross'] = (data['SMA_20'] > data['SMA_50']).astype(int)
                signals['sma_cross_signal'] = data['SMA_20'].diff() > 0
            
            # MACD Signals
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                signals['macd_bullish'] = (data['MACD'] > data['MACD_Signal']).astype(int)
                signals['macd_cross'] = (data['MACD'].diff() > 0).astype(int)
            
            # RSI Signals
            if 'RSI' in data.columns:
                signals['rsi_oversold'] = (data['RSI'] < 30).astype(int)
                signals['rsi_overbought'] = (data['RSI'] > 70).astype(int)
                signals['rsi_neutral'] = ((data['RSI'] >= 30) & (data['RSI'] <= 70)).astype(int)
            
            # Bollinger Bands Signals
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                signals['bb_squeeze'] = (data['BB_Width'] < data['BB_Width'].rolling(20).mean()).astype(int)
                signals['bb_breakout'] = (data['Close'] > data['BB_Upper']).astype(int)
                signals['bb_breakdown'] = (data['Close'] < data['BB_Lower']).astype(int)
            
            # Stochastic Signals
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                signals['stoch_oversold'] = (data['Stoch_K'] < 20).astype(int)
                signals['stoch_overbought'] = (data['Stoch_K'] > 80).astype(int)
                signals['stoch_cross'] = (data['Stoch_K'] > data['Stoch_D']).astype(int)
            
            # Williams %R Signals
            if 'Williams_R' in data.columns:
                signals['williams_oversold'] = (data['Williams_R'] < -80).astype(int)
                signals['williams_overbought'] = (data['Williams_R'] > -20).astype(int)
            
            # CCI Signals
            if 'CCI' in data.columns:
                signals['cci_oversold'] = (data['CCI'] < -100).astype(int)
                signals['cci_overbought'] = (data['CCI'] > 100).astype(int)
            
            # Volume Signals
            if 'Volume_Ratio' in data.columns:
                signals['volume_spike'] = (data['Volume_Ratio'] > 2).astype(int)
                signals['volume_above_avg'] = (data['Volume_Ratio'] > 1).astype(int)
            
            logger.info(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return {}
    
    def calculate_custom_indicator(self, data: pd.DataFrame, indicator_name: str, **params) -> pd.DataFrame:
        """
        Calculate a custom technical indicator
        
        Args:
            data: Stock data
            indicator_name: Name of the indicator
            **params: Indicator parameters
            
        Returns:
            DataFrame with custom indicator
        """
        try:
            if indicator_name == 'custom_sma':
                period = params.get('period', 20)
                data[f'Custom_SMA_{period}'] = data['Close'].rolling(window=period).mean()
            
            elif indicator_name == 'custom_ema':
                period = params.get('period', 20)
                data[f'Custom_EMA_{period}'] = data['Close'].ewm(span=period).mean()
            
            elif indicator_name == 'custom_rsi':
                period = params.get('period', 14)
                data = self._add_rsi(data, period)
            
            elif indicator_name == 'custom_bollinger':
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2)
                data = self._add_bollinger_bands(data, period, std_dev)
            
            else:
                logger.warning(f"Unknown custom indicator: {indicator_name}")
            
            return data
            
        except Exception as e:
            logger.error(f"Custom indicator calculation failed: {e}")
            return data
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary of all technical indicators
        
        Args:
            data: DataFrame with technical indicators
            
        Returns:
            Dictionary with indicator summary
        """
        try:
            summary = {
                'total_indicators': len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]),
                'moving_averages': len([col for col in data.columns if 'SMA' in col or 'EMA' in col or 'WMA' in col]),
                'oscillators': len([col for col in data.columns if any(x in col for x in ['RSI', 'Stoch', 'Williams', 'CCI'])]),
                'volume_indicators': len([col for col in data.columns if 'Volume' in col or 'OBV' in col or 'VPT' in col or 'ADL' in col]),
                'trend_indicators': len([col for col in data.columns if any(x in col for x in ['MACD', 'BB', 'ATR'])]),
                'latest_values': {}
            }
            
            # Get latest values for key indicators
            key_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Stoch_K', 'Williams_R', 'CCI']
            for indicator in key_indicators:
                if indicator in data.columns:
                    summary['latest_values'][indicator] = data[indicator].iloc[-1] if not data.empty else None
            
            return summary
            
        except Exception as e:
            logger.error(f"Indicator summary failed: {e}")
            return {}
