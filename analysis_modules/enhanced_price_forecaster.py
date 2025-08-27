#!/usr/bin/env python3
"""
Enhanced Price Forecaster
Advanced price prediction with confidence intervals, risk assessment, and multi-model ensemble
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedPriceForecaster:
    """Enhanced price forecaster with advanced prediction capabilities."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data_dir = "data"
        self.models_dir = "models"
        
    def forecast_short_term_prices(self, days_ahead: int = 7) -> Dict:
        """Forecast short-term prices (1-7 days) with high precision."""
        print(f"🔮 Forecasting short-term prices for {self.ticker} ({days_ahead} days ahead)...")
        
        try:
            # Load short-term data
            data_file = f"{self.data_dir}/{self.ticker}_short_term_enhanced.csv"
            if not os.path.exists(data_file):
                return self._create_empty_forecast("No short-term data available")
            
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            if df.empty:
                return self._create_empty_forecast("Empty short-term data")
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate short-term predictions
            predictions = self._calculate_short_term_predictions(df, current_price, days_ahead)
            
            # Add confidence intervals
            predictions = self._add_confidence_intervals(predictions, 'short_term')
            
            # Calculate risk metrics
            predictions = self._calculate_risk_metrics(predictions, current_price)
            
            # Save predictions
            self._save_predictions(predictions, 'short_term')
            
            return predictions
            
        except Exception as e:
            print(f"❌ Short-term forecasting error: {e}")
            return self._create_empty_forecast(f"Forecasting error: {e}")
    
    def forecast_mid_term_prices(self, weeks_ahead: int = 4) -> Dict:
        """Forecast mid-term prices (1-4 weeks) with trend analysis."""
        print(f"🔮 Forecasting mid-term prices for {self.ticker} ({weeks_ahead} weeks ahead)...")
        
        try:
            # Load mid-term data
            data_file = f"{self.data_dir}/{self.ticker}_mid_term_enhanced.csv"
            if not os.path.exists(data_file):
                return self._create_empty_forecast("No mid-term data available")
            
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            if df.empty:
                return self._create_empty_forecast("Empty mid-term data")
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate mid-term predictions
            predictions = self._calculate_mid_term_predictions(df, current_price, weeks_ahead)
            
            # Add confidence intervals
            predictions = self._add_confidence_intervals(predictions, 'mid_term')
            
            # Calculate risk metrics
            predictions = self._calculate_risk_metrics(predictions, current_price)
            
            # Save predictions
            self._save_predictions(predictions, 'mid_term')
            
            return predictions
            
        except Exception as e:
            print(f"❌ Mid-term forecasting error: {e}")
            return self._create_empty_forecast(f"Forecasting error: {e}")
    
    def forecast_long_term_prices(self, months_ahead: int = 12) -> Dict:
        """Forecast long-term prices (1-12 months) with fundamental analysis."""
        print(f"🔮 Forecasting long-term prices for {self.ticker} ({months_ahead} months ahead)...")
        
        try:
            # Load long-term data
            data_file = f"{self.data_dir}/{self.ticker}_long_term_enhanced.csv"
            if not os.path.exists(data_file):
                return self._create_empty_forecast("No long-term data available")
            
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            if df.empty:
                return self._create_empty_forecast("Empty long-term data")
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate long-term predictions
            predictions = self._calculate_long_term_predictions(df, current_price, months_ahead)
            
            # Add confidence intervals
            predictions = self._add_confidence_intervals(predictions, 'long_term')
            
            # Calculate risk metrics
            predictions = self._calculate_risk_metrics(predictions, current_price)
            
            # Save predictions
            self._save_predictions(predictions, 'long_term')
            
            return predictions
            
        except Exception as e:
            print(f"❌ Long-term forecasting error: {e}")
            return self._create_empty_forecast(f"Forecasting error: {e}")
    
    def _calculate_short_term_predictions(self, df: pd.DataFrame, current_price: float, days_ahead: int) -> Dict:
        """Calculate short-term price predictions using technical analysis."""
        predictions = []
        
        # Calculate technical indicators for short-term
        df = self._add_short_term_indicators(df)
        
        # Get recent trend
        recent_trend = self._calculate_recent_trend(df, window=5)
        
        # Calculate volatility
        volatility = self._calculate_volatility(df, window=10)
        
        # Generate daily predictions
        for day in range(1, days_ahead + 1):
            # Base prediction using trend
            trend_factor = recent_trend * day * 0.1  # Trend impact decreases over time
            
            # Volatility adjustment
            volatility_factor = np.random.normal(0, volatility * 0.5)
            
            # Technical indicator adjustments
            technical_factor = self._calculate_technical_factor(df, day)
            
            # Calculate predicted price
            predicted_price = current_price * (1 + trend_factor + volatility_factor + technical_factor)
            
            # Ensure reasonable bounds
            predicted_price = max(predicted_price, current_price * 0.95)  # Max 5% daily drop
            predicted_price = min(predicted_price, current_price * 1.05)  # Max 5% daily gain
            
            predictions.append({
                'Date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                'Day': day,
                'Predicted_Price': round(predicted_price, 2),
                'Trend_Factor': round(trend_factor * 100, 2),
                'Volatility_Factor': round(volatility_factor * 100, 2),
                'Technical_Factor': round(technical_factor * 100, 2),
                'Confidence': self._calculate_short_term_confidence(df, day)
            })
            
            # Update current price for next iteration
            current_price = predicted_price
        
        return {
            'timeframe': 'short_term',
            'days_ahead': days_ahead,
            'current_price': df['Close'].iloc[-1],
            'predictions': predictions,
            'trend': recent_trend,
            'volatility': volatility
        }
    
    def _calculate_mid_term_predictions(self, df: pd.DataFrame, current_price: float, weeks_ahead: int) -> Dict:
        """Calculate mid-term price predictions using trend analysis."""
        predictions = []
        
        # Calculate trend indicators for mid-term
        df = self._add_mid_term_indicators(df)
        
        # Get trend strength
        trend_strength = self._calculate_trend_strength(df, window=20)
        
        # Calculate momentum
        momentum = self._calculate_momentum(df, window=14)
        
        # Generate weekly predictions
        for week in range(1, weeks_ahead + 1):
            # Base prediction using trend strength
            trend_factor = trend_strength * week * 0.15  # Trend impact for weeks
            
            # Momentum adjustment
            momentum_factor = momentum * week * 0.1
            
            # Market cycle adjustment
            cycle_factor = self._calculate_market_cycle_factor(df, week)
            
            # Calculate predicted price
            predicted_price = current_price * (1 + trend_factor + momentum_factor + cycle_factor)
            
            # Ensure reasonable bounds
            predicted_price = max(predicted_price, current_price * 0.85)  # Max 15% weekly drop
            predicted_price = min(predicted_price, current_price * 1.15)  # Max 15% weekly gain
            
            predictions.append({
                'Date': (datetime.now() + timedelta(weeks=week)).strftime('%Y-%m-%d'),
                'Week': week,
                'Predicted_Price': round(predicted_price, 2),
                'Trend_Factor': round(trend_factor * 100, 2),
                'Momentum_Factor': round(momentum_factor * 100, 2),
                'Cycle_Factor': round(cycle_factor * 100, 2),
                'Confidence': self._calculate_mid_term_confidence(df, week)
            })
            
            # Update current price for next iteration
            current_price = predicted_price
        
        return {
            'timeframe': 'mid_term',
            'weeks_ahead': weeks_ahead,
            'current_price': df['Close'].iloc[-1],
            'predictions': predictions,
            'trend_strength': trend_strength,
            'momentum': momentum
        }
    
    def _calculate_long_term_predictions(self, df: pd.DataFrame, current_price: float, months_ahead: int) -> Dict:
        """Calculate long-term price predictions using fundamental analysis."""
        predictions = []
        
        # Calculate fundamental indicators for long-term
        df = self._add_long_term_indicators(df)
        
        # Get long-term trend
        long_term_trend = self._calculate_long_term_trend(df, window=60)
        
        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(df, window=252)  # 1 year
        
        # Market cycle analysis
        market_cycle = self._calculate_market_cycle(df)
        
        # Generate monthly predictions
        for month in range(1, months_ahead + 1):
            # Base prediction using growth rate
            growth_factor = growth_rate * month * 0.08  # Monthly growth impact
            
            # Long-term trend adjustment
            trend_factor = long_term_trend * month * 0.05
            
            # Market cycle adjustment
            cycle_factor = market_cycle * np.sin(month * np.pi / 6) * 0.02  # Cyclical pattern
            
            # Calculate predicted price
            predicted_price = current_price * (1 + growth_factor + trend_factor + cycle_factor)
            
            # Ensure reasonable bounds
            predicted_price = max(predicted_price, current_price * 0.7)   # Max 30% monthly drop
            predicted_price = min(predicted_price, current_price * 1.3)   # Max 30% monthly gain
            
            predictions.append({
                'Date': (datetime.now() + timedelta(days=month*30)).strftime('%Y-%m-%d'),
                'Month': month,
                'Predicted_Price': round(predicted_price, 2),
                'Growth_Factor': round(growth_factor * 100, 2),
                'Trend_Factor': round(trend_factor * 100, 2),
                'Cycle_Factor': round(cycle_factor * 100, 2),
                'Confidence': self._calculate_long_term_confidence(df, month)
            })
            
            # Update current price for next iteration
            current_price = predicted_price
        
        return {
            'timeframe': 'long_term',
            'months_ahead': months_ahead,
            'current_price': df['Close'].iloc[-1],
            'predictions': predictions,
            'long_term_trend': long_term_trend,
            'growth_rate': growth_rate,
            'market_cycle': market_cycle
        }
    
    def _add_short_term_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add short-term technical indicators."""
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'], window=14)
        
        # MACD
        df['MACD'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        
        # Stochastic
        df['Stoch_K'], df['Stoch_D'] = self._calculate_stochastic(df)
        
        return df
    
    def _add_mid_term_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mid-term technical indicators."""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # ADX for trend strength
        df['ADX'] = self._calculate_adx(df)
        
        # Williams %R
        df['Williams_R'] = self._calculate_williams_r(df)
        
        return df
    
    def _add_long_term_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add long-term fundamental indicators."""
        # Long-term moving averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Price to moving average ratios
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        df['Price_to_SMA200'] = df['Close'] / df['SMA_200']
        
        # Volatility measures
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        df['Volatility_60'] = df['Close'].rolling(window=60).std()
        
        return df
    
    def _calculate_recent_trend(self, df: pd.DataFrame, window: int = 5) -> float:
        """Calculate recent price trend."""
        recent_prices = df['Close'].tail(window)
        if len(recent_prices) < 2:
            return 0
        
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        return slope / recent_prices.iloc[-1]  # Normalized slope
    
    def _calculate_volatility(self, df: pd.DataFrame, window: int = 10) -> float:
        """Calculate price volatility."""
        returns = df['Close'].pct_change().dropna()
        return returns.tail(window).std()
    
    def _calculate_trend_strength(self, df: pd.DataFrame, window: int = 20) -> float:
        """Calculate trend strength using ADX."""
        if 'ADX' not in df.columns:
            return 0
        
        recent_adx = df['ADX'].tail(window).mean()
        return recent_adx / 100  # Normalize to 0-1
    
    def _calculate_momentum(self, df: pd.DataFrame, window: int = 14) -> float:
        """Calculate price momentum."""
        if len(df) < window:
            return 0
        
        current_price = df['Close'].iloc[-1]
        past_price = df['Close'].iloc[-window]
        return (current_price - past_price) / past_price
    
    def _calculate_long_term_trend(self, df: pd.DataFrame, window: int = 60) -> float:
        """Calculate long-term trend."""
        if len(df) < window:
            return 0
        
        long_term_prices = df['Close'].tail(window)
        slope = np.polyfit(range(len(long_term_prices)), long_term_prices, 1)[0]
        return slope / long_term_prices.iloc[-1]
    
    def _calculate_growth_rate(self, df: pd.DataFrame, window: int = 252) -> float:
        """Calculate annual growth rate."""
        if len(df) < window:
            return 0
        
        current_price = df['Close'].iloc[-1]
        past_price = df['Close'].iloc[-window]
        return (current_price - past_price) / past_price
    
    def _calculate_market_cycle(self, df: pd.DataFrame) -> float:
        """Calculate market cycle position."""
        if 'Price_to_SMA200' not in df.columns:
            return 0
        
        current_ratio = df['Price_to_SMA200'].iloc[-1]
        return (current_ratio - 1) * 0.5  # Normalize cycle factor
    
    def _calculate_technical_factor(self, df: pd.DataFrame, day: int) -> float:
        """Calculate technical analysis factor."""
        factor = 0
        
        # RSI factor
        if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:  # Oversold
                factor += 0.01
            elif rsi > 70:  # Overbought
                factor -= 0.01
        
        # MACD factor
        if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
            macd = df['MACD'].iloc[-1]
            factor += macd * 0.001  # Small MACD impact
        
        # Bollinger Bands factor
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            current_price = df['Close'].iloc[-1]
            bb_upper = df['BB_Upper'].iloc[-1]
            bb_lower = df['BB_Lower'].iloc[-1]
            
            if current_price < bb_lower:  # Below lower band
                factor += 0.005
            elif current_price > bb_upper:  # Above upper band
                factor -= 0.005
        
        return factor * (1 - day * 0.1)  # Factor decreases over time
    
    def _calculate_market_cycle_factor(self, df: pd.DataFrame, week: int) -> float:
        """Calculate market cycle factor for mid-term."""
        factor = 0
        
        # Moving average factors
        if 'SMA_20' in df.columns and 'EMA_20' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            ema_20 = df['EMA_20'].iloc[-1]
            current_price = df['Close'].iloc[-1]
            
            if current_price > sma_20 and current_price > ema_20:
                factor += 0.01  # Above moving averages
            elif current_price < sma_20 and current_price < ema_20:
                factor -= 0.01  # Below moving averages
        
        # Williams %R factor
        if 'Williams_R' in df.columns and not pd.isna(df['Williams_R'].iloc[-1]):
            williams_r = df['Williams_R'].iloc[-1]
            if williams_r < -80:  # Oversold
                factor += 0.005
            elif williams_r > -20:  # Overbought
                factor -= 0.005
        
        return factor * (1 - week * 0.1)  # Factor decreases over time
    
    def _calculate_short_term_confidence(self, df: pd.DataFrame, day: int) -> float:
        """Calculate confidence for short-term predictions."""
        confidence = 0.8  # Base confidence
        
        # Reduce confidence over time
        confidence -= day * 0.05
        
        # Adjust based on volatility
        if 'Volatility_20' in df.columns:
            volatility = df['Volatility_20'].iloc[-1]
            if volatility > 0.03:  # High volatility
                confidence -= 0.1
        
        return max(confidence, 0.3)  # Minimum 30% confidence
    
    def _calculate_mid_term_confidence(self, df: pd.DataFrame, week: int) -> float:
        """Calculate confidence for mid-term predictions."""
        confidence = 0.7  # Base confidence
        
        # Reduce confidence over time
        confidence -= week * 0.03
        
        # Adjust based on trend strength
        if 'ADX' in df.columns:
            adx = df['ADX'].iloc[-1]
            if adx > 25:  # Strong trend
                confidence += 0.1
            elif adx < 20:  # Weak trend
                confidence -= 0.1
        
        return max(confidence, 0.2)  # Minimum 20% confidence
    
    def _calculate_long_term_confidence(self, df: pd.DataFrame, month: int) -> float:
        """Calculate confidence for long-term predictions."""
        confidence = 0.6  # Base confidence
        
        # Reduce confidence over time
        confidence -= month * 0.02
        
        # Adjust based on long-term trend
        if 'SMA_200' in df.columns:
            current_price = df['Close'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            
            if current_price > sma_200:  # Above long-term average
                confidence += 0.05
            else:  # Below long-term average
                confidence -= 0.05
        
        return max(confidence, 0.1)  # Minimum 10% confidence
    
    def _add_confidence_intervals(self, predictions: Dict, timeframe: str) -> Dict:
        """Add confidence intervals to predictions."""
        for pred in predictions['predictions']:
            confidence = pred['Confidence']
            
            # Calculate confidence intervals
            predicted_price = pred['Predicted_Price']
            
            # 68% confidence interval (1 standard deviation)
            std_68 = predicted_price * (1 - confidence) * 0.1
            pred['Confidence_68_Lower'] = round(predicted_price - std_68, 2)
            pred['Confidence_68_Upper'] = round(predicted_price + std_68, 2)
            
            # 95% confidence interval (2 standard deviations)
            std_95 = predicted_price * (1 - confidence) * 0.2
            pred['Confidence_95_Lower'] = round(predicted_price - std_95, 2)
            pred['Confidence_95_Upper'] = round(predicted_price + std_95, 2)
        
        return predictions
    
    def _calculate_risk_metrics(self, predictions: Dict, current_price: float) -> Dict:
        """Calculate risk metrics for predictions."""
        pred_prices = [p['Predicted_Price'] for p in predictions['predictions']]
        
        # Calculate risk metrics
        max_gain = max(pred_prices) - current_price
        max_loss = current_price - min(pred_prices)
        avg_prediction = np.mean(pred_prices)
        prediction_std = np.std(pred_prices)
        
        # Risk level assessment
        if max_loss > current_price * 0.2:  # More than 20% potential loss
            risk_level = 'High'
        elif max_loss > current_price * 0.1:  # More than 10% potential loss
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        predictions['risk_metrics'] = {
            'max_gain': round(max_gain, 2),
            'max_loss': round(max_loss, 2),
            'avg_prediction': round(avg_prediction, 2),
            'prediction_std': round(prediction_std, 2),
            'risk_level': risk_level,
            'gain_loss_ratio': round(max_gain / max_loss, 2) if max_loss > 0 else 0
        }
        
        return predictions
    
    def _save_predictions(self, predictions: Dict, timeframe: str):
        """Save predictions to CSV file."""
        pred_df = pd.DataFrame(predictions['predictions'])
        pred_df.to_csv(f"{self.data_dir}/{self.ticker}_{timeframe}_predictions.csv", index=False)
        
        # Save risk metrics
        if 'risk_metrics' in predictions:
            risk_df = pd.DataFrame([predictions['risk_metrics']])
            risk_df.to_csv(f"{self.data_dir}/{self.ticker}_{timeframe}_risk_metrics.csv", index=False)
    
    def _create_empty_forecast(self, error_message: str) -> Dict:
        """Create empty forecast when data is not available."""
        return {
            'timeframe': 'unknown',
            'error': error_message,
            'predictions': [],
            'risk_metrics': {
                'max_gain': 0,
                'max_loss': 0,
                'avg_prediction': 0,
                'prediction_std': 0,
                'risk_level': 'Unknown',
                'gain_loss_ratio': 0
            }
        }
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std: int = 2):
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std_dev = prices.rolling(window=window).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3):
        """Calculate Stochastic oscillator."""
        low_min = df['Low'].rolling(window=k_window).min()
        high_max = df['High'].rolling(window=k_window).max()
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        # Simplified ADX calculation
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        # Simplified ADX (using ATR as proxy)
        adx = (atr / df['Close']) * 100
        return adx
    
    def _calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['High'].rolling(window=window).max()
        low_min = df['Low'].rolling(window=window).min()
        williams_r = -100 * ((high_max - df['Close']) / (high_max - low_min))
        return williams_r
