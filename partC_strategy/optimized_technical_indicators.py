#!/usr/bin/env python3
"""
Optimized Technical Indicators Module
Consolidates all technical analysis functionality from duplicate files
"""

import pandas as pd
import numpy as np
import talib
from scipy import stats
from scipy.signal import savgol_filter
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class OptimizedTechnicalIndicators:
    """
    Optimized technical indicators that consolidates functionality from:
    - technicals.py
    - enhanced_technicals.py
    - advanced_technicals.py
    """
    
    def __init__(self):
        self.indicators = {}
        self.signal_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'williams_oversold': -80,
            'williams_overbought': -20
        }
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators including basic, enhanced, and advanced.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators
        """
        print("📊 Adding comprehensive technical indicators...")
        
        df = df.copy()
        
        # Basic indicators
        df = self._add_basic_indicators(df)
        
        # Enhanced momentum indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility indicators
        df = self._add_volatility_indicators(df)
        
        # Volume indicators
        df = self._add_volume_indicators(df)
        
        # Trend indicators
        df = self._add_trend_indicators(df)
        
        # Advanced indicators
        df = self._add_advanced_indicators(df)
        
        # Market microstructure indicators
        df = self._add_microstructure_indicators(df)
        
        # Support/Resistance levels
        df = self._add_support_resistance(df)
        
        # Market regime indicators
        df = self._add_market_regime_indicators(df)
        
        # Fill NaN values instead of dropping them
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaN values with mean, but only for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Only drop rows that have NaN values in essential columns (Close, Open, High, Low)
        essential_columns = ['Close', 'Open', 'High', 'Low']
        essential_cols_present = [col for col in essential_columns if col in df.columns]
        if essential_cols_present:
            df = df.dropna(subset=essential_cols_present)
        else:
            # If essential columns are not present, drop rows with any NaN values
            df = df.dropna()
        
        print(f"✅ Added {len([col for col in df.columns if any(x in col for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR', 'Stoch', 'Williams', 'CCI', 'ROC', 'MFI', 'OBV', 'VWAP', 'Support', 'Resistance'])])} technical indicators")
        print(f"📈 Final shape: {df.shape}")
        
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
            
            # Price position relative to moving averages
            df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
            df['Price_vs_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
            df['Price_vs_SMA200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
            
            # MACD - ensure EMAs exist first
            if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            else:
                # Create EMAs specifically for MACD if they don't exist
                df['EMA_12'] = df['Close'].ewm(span=12).mean()
                df['EMA_26'] = df['Close'].ewm(span=26).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
                df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            
        except Exception as e:
            print(f"Error adding basic indicators: {e}")
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        try:
            # RSI
            df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
            df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
            
            # Stochastic Oscillator
            df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
            
            # Stochastic RSI
            df['StochRSI_K'], df['StochRSI_D'] = talib.STOCHRSI(df['Close'])
            
            # Williams %R
            df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # Rate of Change
            df['ROC_10'] = talib.ROC(df['Close'], timeperiod=10)
            df['ROC_20'] = talib.ROC(df['Close'], timeperiod=20)
            
            # Momentum
            df['MOM_10'] = talib.MOM(df['Close'], timeperiod=10)
            
            # Commodity Channel Index
            df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
        except Exception as e:
            print(f"Error adding momentum indicators: {e}")
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        try:
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
                df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Average True Range
            df['ATR_14'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['ATR_21'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=21)
            
            # Normalized ATR
            df['ATR_Ratio'] = df['ATR_14'] / df['Close']
            
            # Historical Volatility
            df['Hist_Vol_20'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            
        except Exception as e:
            print(f"Error adding volatility indicators: {e}")
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        try:
            # Volume Rate of Change
            df['Volume_ROC'] = df['Volume'].pct_change(periods=5)
            
            # Volume Weighted Average Price (VWAP)
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # Volume Price Trend (VPT)
            df['VPT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
            df['VPT'] = df['VPT'].cumsum()
            
            # Money Flow Index (MFI)
            df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
            
            # On Balance Volume (OBV)
            df['OBV'] = talib.OBV(df['Close'], df['Volume'])
            
            # Accumulation/Distribution Line
            df['ADL'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Chaikin Money Flow
            df['CMF'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'], fastperiod=3, slowperiod=10)
            
            # Volume Price Confirmation
            df['Volume_Price_Confirmation'] = np.where(
                (df['Close'] > df['Close'].shift(1)) & (df['Volume'] > df['Volume'].rolling(20).mean()),
                1,  # Bullish confirmation
                np.where(
                    (df['Close'] < df['Close'].shift(1)) & (df['Volume'] > df['Volume'].rolling(20).mean()),
                    -1,  # Bearish confirmation
                    0  # Neutral
                )
            )
            
            # Volume SMA
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
        except Exception as e:
            print(f"Error adding volume indicators: {e}")
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators."""
        try:
            # Parabolic SAR
            df['SAR'] = talib.SAR(df['High'], df['Low'])
            
            # Directional Movement Index
            df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['DI_Plus'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            df['DI_Minus'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # Aroon Oscillator
            df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['High'], df['Low'], timeperiod=14)
            df['Aroon_Osc'] = df['Aroon_Up'] - df['Aroon_Down']
            
        except Exception as e:
            print(f"Error adding trend indicators: {e}")
        
        return df
    
    def _add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators."""
        try:
            # Ichimoku Cloud
            high_9 = df['High'].rolling(window=9).max()
            low_9 = df['Low'].rolling(window=9).min()
            df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            high_26 = df['High'].rolling(window=26).max()
            low_26 = df['Low'].rolling(window=26).min()
            df['Ichimoku_Base'] = (high_26 + low_26) / 2
            
            df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            
            high_52 = df['High'].rolling(window=52).max()
            low_52 = df['Low'].rolling(window=52).min()
            df['Ichimoku_SpanB'] = ((high_52 + low_52) / 2).shift(26)
            
            # Fibonacci Retracement levels
            high_20 = df['High'].rolling(window=20).max()
            low_20 = df['Low'].rolling(window=20).min()
            diff = high_20 - low_20
            
            df['Fib_236'] = high_20 - (diff * 0.236)
            df['Fib_382'] = high_20 - (diff * 0.382)
            df['Fib_500'] = high_20 - (diff * 0.500)
            df['Fib_618'] = high_20 - (diff * 0.618)
            
        except Exception as e:
            print(f"Error adding advanced indicators: {e}")
        
        return df
    
    def _add_microstructure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure indicators."""
        try:
            # Price efficiency ratio
            df['Price_Efficiency'] = abs(df['Close'] - df['Close'].shift(20)) / df['Close'].rolling(20).apply(lambda x: sum(abs(x.diff().dropna())))
            
            # Volume efficiency
            df['Volume_Efficiency'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
            
            # Price momentum efficiency
            df['Momentum_Efficiency'] = df['Close'].pct_change().rolling(20).apply(lambda x: x.corr(pd.Series(range(len(x)))))
            
        except Exception as e:
            print(f"Error adding microstructure indicators: {e}")
        
        return df
    
    def _add_support_resistance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels."""
        try:
            # Dynamic support and resistance using rolling min/max
            df['Support_20'] = df['Low'].rolling(window=20).min()
            df['Resistance_20'] = df['High'].rolling(window=20).max()
            
            # Pivot points
            df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot'] - df['Low']
            df['S1'] = 2 * df['Pivot'] - df['High']
            df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
            
        except Exception as e:
            print(f"Error adding support/resistance: {e}")
        
        return df
    
    def _add_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators."""
        try:
            # Trend strength
            df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
            
            # Volatility regime
            df['Volatility_Regime'] = np.where(df['Hist_Vol_20'] > df['Hist_Vol_20'].rolling(50).mean(), 'High', 'Low')
            
            # Market efficiency ratio
            df['Market_Efficiency'] = df['Close'].pct_change().rolling(20).apply(lambda x: x.corr(pd.Series(range(len(x)))))
            
        except Exception as e:
            print(f"Error adding market regime indicators: {e}")
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with trading signals
        """
        df = df.copy()
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Signal_Reason'] = ''
        
        for i in range(20, len(df)):  # Start from 20 to ensure indicators are available
            buy_conditions = []
            sell_conditions = []
            signal_strength = 0.0
            
            # RSI conditions
            if 'RSI_14' in df.columns and not pd.isna(df['RSI_14'].iloc[i]):
                rsi = df['RSI_14'].iloc[i]
                if rsi < self.signal_thresholds['rsi_oversold']:
                    buy_conditions.append('RSI_oversold')
                    signal_strength += 0.3
                elif rsi > self.signal_thresholds['rsi_overbought']:
                    sell_conditions.append('RSI_overbought')
                    signal_strength -= 0.3
            
            # MACD conditions
            if all(col in df.columns for col in ['MACD', 'Signal_Line']):
                macd = df['MACD'].iloc[i]
                signal = df['Signal_Line'].iloc[i]
                macd_prev = df['MACD'].iloc[i-1]
                signal_prev = df['Signal_Line'].iloc[i-1]
                
                if macd > signal and macd_prev <= signal_prev:
                    buy_conditions.append('MACD_bull_cross')
                    signal_strength += 0.4
                elif macd < signal and macd_prev >= signal_prev:
                    sell_conditions.append('MACD_bear_cross')
                    signal_strength -= 0.4
            
            # Moving average conditions
            if all(col in df.columns for col in ['SMA_20', 'SMA_50']):
                close = df['Close'].iloc[i]
                sma20 = df['SMA_20'].iloc[i]
                sma50 = df['SMA_50'].iloc[i]
                
                if close > sma20 and close > sma50:
                    buy_conditions.append('Price_above_MAs')
                    signal_strength += 0.2
                elif close < sma20 and close < sma50:
                    sell_conditions.append('Price_below_MAs')
                    signal_strength -= 0.2
            
            # Stochastic conditions
            if 'Stoch_K' in df.columns and not pd.isna(df['Stoch_K'].iloc[i]):
                stoch_k = df['Stoch_K'].iloc[i]
                if stoch_k < self.signal_thresholds['stoch_oversold']:
                    buy_conditions.append('Stoch_oversold')
                    signal_strength += 0.2
                elif stoch_k > self.signal_thresholds['stoch_overbought']:
                    sell_conditions.append('Stoch_overbought')
                    signal_strength -= 0.2
            
            # Williams %R conditions
            if 'Williams_R' in df.columns and not pd.isna(df['Williams_R'].iloc[i]):
                williams_r = df['Williams_R'].iloc[i]
                if williams_r < self.signal_thresholds['williams_oversold']:
                    buy_conditions.append('Williams_oversold')
                    signal_strength += 0.2
                elif williams_r > self.signal_thresholds['williams_overbought']:
                    sell_conditions.append('Williams_overbought')
                    signal_strength -= 0.2
            
            # Determine final signal
            if signal_strength > 0.3:
                df.loc[df.index[i], 'Signal'] = 'BUY'
                df.loc[df.index[i], 'Signal_Strength'] = signal_strength
                df.loc[df.index[i], 'Signal_Reason'] = ', '.join(buy_conditions)
            elif signal_strength < -0.3:
                df.loc[df.index[i], 'Signal'] = 'SELL'
                df.loc[df.index[i], 'Signal_Strength'] = abs(signal_strength)
                df.loc[df.index[i], 'Signal_Reason'] = ', '.join(sell_conditions)
        
        return df

# Convenience function for backward compatibility
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility function."""
    analyzer = OptimizedTechnicalIndicators()
    return analyzer.add_all_indicators(df)

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility function."""
    analyzer = OptimizedTechnicalIndicators()
    return analyzer.generate_signals(df)
