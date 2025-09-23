#!/usr/bin/env python3
"""
Feature Engineering Service
Service for engineering features from stock data
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeatureEngineeringService:
    """Service for engineering features from stock data"""
    
    def __init__(self):
        logger.info("Feature Engineering Service initialized")
    
    def engineer_all_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Engineer all types of features
        
        Args:
            data: Stock data with technical indicators
            
        Returns:
            Dictionary with all engineered features
        """
        try:
            features = {}
            
            # Price-based features
            features['price_features'] = self._create_price_features(data)
            
            # Technical indicator features
            features['technical_features'] = self._create_technical_features(data)
            
            # Volume features
            features['volume_features'] = self._create_volume_features(data)
            
            # Time-based features
            features['time_features'] = self._create_time_features(data)
            
            # Statistical features
            features['statistical_features'] = self._create_statistical_features(data)
            
            # Momentum features
            features['momentum_features'] = self._create_momentum_features(data)
            
            # Volatility features
            features['volatility_features'] = self._create_volatility_features(data)
            
            # Pattern features
            features['pattern_features'] = self._create_pattern_features(data)
            
            # Market regime features
            features['regime_features'] = self._create_regime_features(data)
            
            logger.info(f"Engineered {len(features)} feature groups")
            return features
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return {}
    
    def _create_price_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create price-based features"""
        try:
            features = {}
            
            # Basic price changes
            features['price_change'] = data['Close'].pct_change()
            features['price_change_abs'] = data['Close'].diff()
            features['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Price ranges
            features['daily_range'] = data['High'] - data['Low']
            features['daily_range_pct'] = features['daily_range'] / data['Close']
            features['body_size'] = abs(data['Close'] - data['Open'])
            features['body_size_pct'] = features['body_size'] / data['Close']
            
            # Price positions
            features['close_to_high'] = data['Close'] / data['High']
            features['close_to_low'] = data['Close'] / data['Low']
            features['close_to_open'] = data['Close'] / data['Open']
            features['open_to_high'] = data['Open'] / data['High']
            features['open_to_low'] = data['Open'] / data['Low']
            
            # Gap features
            features['gap_up'] = (data['Open'] > data['Close'].shift(1)).astype(int)
            features['gap_down'] = (data['Open'] < data['Close'].shift(1)).astype(int)
            features['gap_size'] = data['Open'] - data['Close'].shift(1)
            features['gap_size_pct'] = features['gap_size'] / data['Close'].shift(1)
            
            # Price levels
            features['price_above_sma20'] = (data['Close'] > data.get('SMA_20', data['Close'])).astype(int)
            features['price_above_sma50'] = (data['Close'] > data.get('SMA_50', data['Close'])).astype(int)
            features['price_above_sma200'] = (data['Close'] > data.get('SMA_200', data['Close'])).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Price features failed: {e}")
            return {}
    
    def _create_technical_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create technical indicator features"""
        try:
            features = {}
            
            # Moving average features
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                features['sma_20_50_ratio'] = data['SMA_20'] / data['SMA_50']
                features['sma_20_50_diff'] = data['SMA_20'] - data['SMA_50']
                features['sma_20_50_signal'] = (data['SMA_20'] > data['SMA_50']).astype(int)
            
            if 'EMA_12' in data.columns and 'EMA_26' in data.columns:
                features['ema_12_26_ratio'] = data['EMA_12'] / data['EMA_26']
                features['ema_12_26_diff'] = data['EMA_12'] - data['EMA_26']
                features['ema_12_26_signal'] = (data['EMA_12'] > data['EMA_26']).astype(int)
            
            # MACD features
            if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                features['macd_signal'] = (data['MACD'] > data['MACD_Signal']).astype(int)
                features['macd_histogram'] = data['MACD_Histogram']
                features['macd_histogram_change'] = data['MACD_Histogram'].diff()
                features['macd_histogram_signal'] = (data['MACD_Histogram'] > 0).astype(int)
            
            # RSI features
            if 'RSI' in data.columns:
                features['rsi_oversold'] = (data['RSI'] < 30).astype(int)
                features['rsi_overbought'] = (data['RSI'] > 70).astype(int)
                features['rsi_neutral'] = ((data['RSI'] >= 30) & (data['RSI'] <= 70)).astype(int)
                features['rsi_momentum'] = data['RSI'].diff()
                features['rsi_divergence'] = self._calculate_rsi_divergence(data)
            
            # Bollinger Bands features
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                features['bb_position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                features['bb_squeeze'] = (data['BB_Width'] < data['BB_Width'].rolling(20).mean()).astype(int)
                features['bb_breakout'] = (data['Close'] > data['BB_Upper']).astype(int)
                features['bb_breakdown'] = (data['Close'] < data['BB_Lower']).astype(int)
                features['bb_width'] = data['BB_Width']
                features['bb_width_change'] = data['BB_Width'].diff()
            
            # Stochastic features
            if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
                features['stoch_oversold'] = (data['Stoch_K'] < 20).astype(int)
                features['stoch_overbought'] = (data['Stoch_K'] > 80).astype(int)
                features['stoch_cross'] = (data['Stoch_K'] > data['Stoch_D']).astype(int)
                features['stoch_divergence'] = self._calculate_stoch_divergence(data)
            
            return features
            
        except Exception as e:
            logger.error(f"Technical features failed: {e}")
            return {}
    
    def _create_volume_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create volume-based features"""
        try:
            features = {}
            
            # Volume changes
            features['volume_change'] = data['Volume'].pct_change()
            features['volume_change_abs'] = data['Volume'].diff()
            features['log_volume'] = np.log(data['Volume'] + 1)
            features['log_volume_change'] = features['log_volume'].diff()
            
            # Volume ratios
            if 'Volume_SMA' in data.columns:
                features['volume_ratio'] = data['Volume'] / data['Volume_SMA']
                features['volume_above_avg'] = (data['Volume'] > data['Volume_SMA']).astype(int)
                features['volume_spike'] = (features['volume_ratio'] > 2).astype(int)
                features['volume_dry'] = (features['volume_ratio'] < 0.5).astype(int)
            
            # Volume-price relationship
            features['volume_price_trend'] = (data['Volume'] * data['Close'].pct_change()).cumsum()
            features['volume_price_correlation'] = data['Volume'].rolling(20).corr(data['Close'].pct_change())
            
            # On-Balance Volume features
            if 'OBV' in data.columns:
                features['obv_change'] = data['OBV'].pct_change()
                features['obv_sma'] = data['OBV'].rolling(20).mean()
                features['obv_signal'] = (data['OBV'] > features['obv_sma']).astype(int)
            
            # Accumulation/Distribution features
            if 'ADL' in data.columns:
                features['adl_change'] = data['ADL'].pct_change()
                features['adl_sma'] = data['ADL'].rolling(20).mean()
                features['adl_signal'] = (data['ADL'] > features['adl_sma']).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Volume features failed: {e}")
            return {}
    
    def _create_time_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create time-based features"""
        try:
            features = {}
            
            # Day of week features
            features['day_of_week'] = data.index.dayofweek
            features['is_monday'] = (data.index.dayofweek == 0).astype(int)
            features['is_tuesday'] = (data.index.dayofweek == 1).astype(int)
            features['is_wednesday'] = (data.index.dayofweek == 2).astype(int)
            features['is_thursday'] = (data.index.dayofweek == 3).astype(int)
            features['is_friday'] = (data.index.dayofweek == 4).astype(int)
            features['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            
            # Month features
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            features['is_january'] = (data.index.month == 1).astype(int)
            features['is_december'] = (data.index.month == 12).astype(int)
            features['is_q1'] = (data.index.quarter == 1).astype(int)
            features['is_q4'] = (data.index.quarter == 4).astype(int)
            
            # Year features
            features['year'] = data.index.year
            features['year_progress'] = data.index.dayofyear / 365.25
            
            # Cyclical features
            features['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            features['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)
            features['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
            
            # Trading session features
            features['is_first_hour'] = (data.index.hour == 9).astype(int) if hasattr(data.index, 'hour') else 0
            features['is_last_hour'] = (data.index.hour == 15).astype(int) if hasattr(data.index, 'hour') else 0
            
            return features
            
        except Exception as e:
            logger.error(f"Time features failed: {e}")
            return {}
    
    def _create_statistical_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create statistical features"""
        try:
            features = {}
            
            # Rolling statistics
            window = 20
            features['price_std'] = data['Close'].rolling(window=window).std()
            features['price_mean'] = data['Close'].rolling(window=window).mean()
            features['price_skew'] = data['Close'].rolling(window=window).skew()
            features['price_kurt'] = data['Close'].rolling(window=window).kurt()
            features['price_median'] = data['Close'].rolling(window=window).median()
            features['price_min'] = data['Close'].rolling(window=window).min()
            features['price_max'] = data['Close'].rolling(window=window).max()
            
            # Z-scores
            features['price_zscore'] = (data['Close'] - features['price_mean']) / features['price_std']
            features['volume_zscore'] = (data['Volume'] - data['Volume'].rolling(window=window).mean()) / data['Volume'].rolling(window=window).std()
            
            # Percentile ranks
            features['price_percentile'] = data['Close'].rolling(window=window).rank(pct=True)
            features['volume_percentile'] = data['Volume'].rolling(window=window).rank(pct=True)
            
            # Autocorrelation
            features['price_autocorr_1'] = data['Close'].rolling(window=window).apply(lambda x: x.autocorr(lag=1))
            features['price_autocorr_5'] = data['Close'].rolling(window=window).apply(lambda x: x.autocorr(lag=5))
            
            # Rolling correlations
            features['price_volume_corr'] = data['Close'].rolling(window=window).corr(data['Volume'])
            
            return features
            
        except Exception as e:
            logger.error(f"Statistical features failed: {e}")
            return {}
    
    def _create_momentum_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create momentum features"""
        try:
            features = {}
            
            # Price momentum
            features['momentum_1'] = data['Close'] / data['Close'].shift(1) - 1
            features['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
            features['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            features['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
            
            # Rate of change
            features['roc_1'] = data['Close'].pct_change(1)
            features['roc_5'] = data['Close'].pct_change(5)
            features['roc_10'] = data['Close'].pct_change(10)
            features['roc_20'] = data['Close'].pct_change(20)
            
            # Momentum oscillators
            features['momentum_oscillator'] = data['Close'] - data['Close'].shift(10)
            features['momentum_oscillator_sma'] = features['momentum_oscillator'].rolling(10).mean()
            features['momentum_oscillator_signal'] = (features['momentum_oscillator'] > features['momentum_oscillator_sma']).astype(int)
            
            # Acceleration
            features['acceleration'] = data['Close'].diff().diff()
            features['acceleration_sma'] = features['acceleration'].rolling(5).mean()
            features['acceleration_signal'] = (features['acceleration'] > features['acceleration_sma']).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Momentum features failed: {e}")
            return {}
    
    def _create_volatility_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create volatility features"""
        try:
            features = {}
            
            # Historical volatility
            features['volatility_5'] = data['Close'].pct_change().rolling(5).std() * np.sqrt(252)
            features['volatility_10'] = data['Close'].pct_change().rolling(10).std() * np.sqrt(252)
            features['volatility_20'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            features['volatility_30'] = data['Close'].pct_change().rolling(30).std() * np.sqrt(252)
            
            # Volatility ratios
            features['vol_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
            features['vol_ratio_10_20'] = features['volatility_10'] / features['volatility_20']
            
            # ATR-based volatility
            if 'ATR' in data.columns:
                features['atr_volatility'] = data['ATR'] / data['Close']
                features['atr_volatility_sma'] = features['atr_volatility'].rolling(20).mean()
                features['atr_volatility_signal'] = (features['atr_volatility'] > features['atr_volatility_sma']).astype(int)
            
            # GARCH-like features
            features['squared_returns'] = data['Close'].pct_change() ** 2
            features['volatility_clustering'] = features['squared_returns'].rolling(20).mean()
            
            # Volatility regime
            features['high_volatility'] = (features['volatility_20'] > features['volatility_20'].rolling(50).quantile(0.8)).astype(int)
            features['low_volatility'] = (features['volatility_20'] < features['volatility_20'].rolling(50).quantile(0.2)).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Volatility features failed: {e}")
            return {}
    
    def _create_pattern_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create pattern recognition features"""
        try:
            features = {}
            
            # Candlestick patterns
            features['doji'] = (abs(data['Open'] - data['Close']) < (data['High'] - data['Low']) * 0.1).astype(int)
            features['hammer'] = self._detect_hammer(data)
            features['shooting_star'] = self._detect_shooting_star(data)
            features['engulfing'] = self._detect_engulfing(data)
            
            # Support and resistance
            features['near_support'] = self._detect_near_support(data)
            features['near_resistance'] = self._detect_near_resistance(data)
            
            # Trend patterns
            features['higher_highs'] = self._detect_higher_highs(data)
            features['lower_lows'] = self._detect_lower_lows(data)
            features['double_top'] = self._detect_double_top(data)
            features['double_bottom'] = self._detect_double_bottom(data)
            
            return features
            
        except Exception as e:
            logger.error(f"Pattern features failed: {e}")
            return {}
    
    def _create_regime_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create market regime features"""
        try:
            features = {}
            
            # Trend regime
            if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                features['trend_regime'] = self._classify_trend_regime(data)
                features['trend_strength'] = self._calculate_trend_strength(data)
            
            # Volatility regime
            features['volatility_regime'] = self._classify_volatility_regime(data)
            
            # Volume regime
            features['volume_regime'] = self._classify_volume_regime(data)
            
            # Market regime
            features['market_regime'] = self._classify_market_regime(data)
            
            return features
            
        except Exception as e:
            logger.error(f"Regime features failed: {e}")
            return {}
    
    def _calculate_rsi_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate RSI divergence"""
        try:
            if 'RSI' not in data.columns:
                return pd.Series(0, index=data.index)
            
            # Simple divergence detection
            price_peaks = data['Close'].rolling(5, center=True).max() == data['Close']
            rsi_peaks = data['RSI'].rolling(5, center=True).max() == data['RSI']
            
            divergence = pd.Series(0, index=data.index)
            divergence[price_peaks & rsi_peaks] = 1
            
            return divergence
            
        except Exception as e:
            logger.error(f"RSI divergence calculation failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_stoch_divergence(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Stochastic divergence"""
        try:
            if 'Stoch_K' not in data.columns:
                return pd.Series(0, index=data.index)
            
            # Simple divergence detection
            price_peaks = data['Close'].rolling(5, center=True).max() == data['Close']
            stoch_peaks = data['Stoch_K'].rolling(5, center=True).max() == data['Stoch_K']
            
            divergence = pd.Series(0, index=data.index)
            divergence[price_peaks & stoch_peaks] = 1
            
            return divergence
            
        except Exception as e:
            logger.error(f"Stochastic divergence calculation failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_hammer(self, data: pd.DataFrame) -> pd.Series:
        """Detect hammer candlestick pattern"""
        try:
            body = abs(data['Close'] - data['Open'])
            lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
            upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
            
            hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
            return hammer.astype(int)
            
        except Exception as e:
            logger.error(f"Hammer detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_shooting_star(self, data: pd.DataFrame) -> pd.Series:
        """Detect shooting star candlestick pattern"""
        try:
            body = abs(data['Close'] - data['Open'])
            lower_shadow = data[['Open', 'Close']].min(axis=1) - data['Low']
            upper_shadow = data['High'] - data[['Open', 'Close']].max(axis=1)
            
            shooting_star = (upper_shadow > 2 * body) & (lower_shadow < body)
            return shooting_star.astype(int)
            
        except Exception as e:
            logger.error(f"Shooting star detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_engulfing(self, data: pd.DataFrame) -> pd.Series:
        """Detect engulfing candlestick pattern"""
        try:
            prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
            curr_body = abs(data['Close'] - data['Open'])
            
            bullish_engulfing = (data['Close'] > data['Open']) & (data['Close'].shift(1) < data['Open'].shift(1)) & (curr_body > prev_body)
            bearish_engulfing = (data['Close'] < data['Open']) & (data['Close'].shift(1) > data['Open'].shift(1)) & (curr_body > prev_body)
            
            engulfing = bullish_engulfing | bearish_engulfing
            return engulfing.astype(int)
            
        except Exception as e:
            logger.error(f"Engulfing detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_near_support(self, data: pd.DataFrame) -> pd.Series:
        """Detect if price is near support level"""
        try:
            support = data['Low'].rolling(20).min()
            near_support = (data['Close'] <= support * 1.02) & (data['Close'] >= support * 0.98)
            return near_support.astype(int)
            
        except Exception as e:
            logger.error(f"Support detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_near_resistance(self, data: pd.DataFrame) -> pd.Series:
        """Detect if price is near resistance level"""
        try:
            resistance = data['High'].rolling(20).max()
            near_resistance = (data['Close'] >= resistance * 0.98) & (data['Close'] <= resistance * 1.02)
            return near_resistance.astype(int)
            
        except Exception as e:
            logger.error(f"Resistance detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_higher_highs(self, data: pd.DataFrame) -> pd.Series:
        """Detect higher highs pattern"""
        try:
            highs = data['High'].rolling(5, center=True).max() == data['High']
            higher_highs = (data['High'] > data['High'].shift(5)) & highs
            return higher_highs.astype(int)
            
        except Exception as e:
            logger.error(f"Higher highs detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_lower_lows(self, data: pd.DataFrame) -> pd.Series:
        """Detect lower lows pattern"""
        try:
            lows = data['Low'].rolling(5, center=True).min() == data['Low']
            lower_lows = (data['Low'] < data['Low'].shift(5)) & lows
            return lower_lows.astype(int)
            
        except Exception as e:
            logger.error(f"Lower lows detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_double_top(self, data: pd.DataFrame) -> pd.Series:
        """Detect double top pattern"""
        try:
            highs = data['High'].rolling(5, center=True).max() == data['High']
            double_top = highs & (data['High'].shift(10) > data['High'].shift(5)) & (data['High'].shift(5) > data['High'])
            return double_top.astype(int)
            
        except Exception as e:
            logger.error(f"Double top detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> pd.Series:
        """Detect double bottom pattern"""
        try:
            lows = data['Low'].rolling(5, center=True).min() == data['Low']
            double_bottom = lows & (data['Low'].shift(10) < data['Low'].shift(5)) & (data['Low'].shift(5) < data['Low'])
            return double_bottom.astype(int)
            
        except Exception as e:
            logger.error(f"Double bottom detection failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _classify_trend_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify trend regime"""
        try:
            if 'SMA_20' not in data.columns or 'SMA_50' not in data.columns:
                return pd.Series(0, index=data.index)
            
            # 0: Sideways, 1: Uptrend, -1: Downtrend
            trend = pd.Series(0, index=data.index)
            trend[(data['SMA_20'] > data['SMA_50']) & (data['Close'] > data['SMA_20'])] = 1
            trend[(data['SMA_20'] < data['SMA_50']) & (data['Close'] < data['SMA_20'])] = -1
            
            return trend
            
        except Exception as e:
            logger.error(f"Trend regime classification failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength"""
        try:
            if 'SMA_20' not in data.columns or 'SMA_50' not in data.columns:
                return pd.Series(0, index=data.index)
            
            trend_strength = abs(data['SMA_20'] - data['SMA_50']) / data['SMA_50']
            return trend_strength
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _classify_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify volatility regime"""
        try:
            volatility = data['Close'].pct_change().rolling(20).std()
            vol_median = volatility.rolling(50).median()
            
            # 0: Normal, 1: High, -1: Low
            vol_regime = pd.Series(0, index=data.index)
            vol_regime[volatility > vol_median * 1.5] = 1
            vol_regime[volatility < vol_median * 0.5] = -1
            
            return vol_regime
            
        except Exception as e:
            logger.error(f"Volatility regime classification failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _classify_volume_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify volume regime"""
        try:
            if 'Volume_SMA' not in data.columns:
                return pd.Series(0, index=data.index)
            
            volume_ratio = data['Volume'] / data['Volume_SMA']
            vol_median = volume_ratio.rolling(50).median()
            
            # 0: Normal, 1: High, -1: Low
            vol_regime = pd.Series(0, index=data.index)
            vol_regime[volume_ratio > vol_median * 1.5] = 1
            vol_regime[volume_ratio < vol_median * 0.5] = -1
            
            return vol_regime
            
        except Exception as e:
            logger.error(f"Volume regime classification failed: {e}")
            return pd.Series(0, index=data.index)
    
    def _classify_market_regime(self, data: pd.DataFrame) -> pd.Series:
        """Classify overall market regime"""
        try:
            # Combine trend, volatility, and volume regimes
            trend_regime = self._classify_trend_regime(data)
            vol_regime = self._classify_volatility_regime(data)
            volume_regime = self._classify_volume_regime(data)
            
            # Simple combination
            market_regime = trend_regime + vol_regime + volume_regime
            
            # Normalize to -1, 0, 1
            market_regime = np.where(market_regime > 1, 1, np.where(market_regime < -1, -1, 0))
            
            return pd.Series(market_regime, index=data.index)
            
        except Exception as e:
            logger.error(f"Market regime classification failed: {e}")
            return pd.Series(0, index=data.index)
    
    def get_feature_summary(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of engineered features
        
        Args:
            features: Dictionary with engineered features
            
        Returns:
            Dictionary with feature summary
        """
        try:
            summary = {
                'total_feature_groups': len(features),
                'feature_groups': list(features.keys()),
                'total_features': sum(len(feature_group) for feature_group in features.values() if isinstance(feature_group, dict)),
                'feature_counts': {group: len(feature_group) for group, feature_group in features.items() if isinstance(feature_group, dict)}
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Feature summary failed: {e}")
            return {}
