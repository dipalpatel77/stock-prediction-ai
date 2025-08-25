#!/usr/bin/env python3
"""
Optimized Trading Strategy Module
Consolidates all trading strategy functionality from duplicate files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .optimized_sentiment_analyzer import OptimizedSentimentAnalyzer
from .optimized_technical_indicators import OptimizedTechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

class OptimizedTradingStrategy:
    """
    Optimized trading strategy that consolidates functionality from:
    - strategies.py
    - enhanced_strategy.py
    - strategy_library.py
    - strategy_tuner.py
    - ensemble_predictor.py
    """
    
    def __init__(self, sentiment_analyzer: Optional[OptimizedSentimentAnalyzer] = None):
        """
        Initialize the optimized trading strategy.
        
        Args:
            sentiment_analyzer: Optional OptimizedSentimentAnalyzer instance
        """
        self.sentiment_analyzer = sentiment_analyzer
        self.technical_analyzer = OptimizedTechnicalIndicators()
        
        # Strategy weights
        self.strategy_weights = {
            'lstm_prediction': 0.35,      # 35% weight to LSTM predictions
            'technical_indicators': 0.25,  # 25% weight to technical analysis
            'sentiment_analysis': 0.40     # 40% weight to comprehensive sentiment
        }
        
        # Signal thresholds
        self.signal_thresholds = {
            'strong_buy': 0.7,
            'buy': 0.3,
            'hold': -0.3,
            'sell': -0.7
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_position_size': 0.1,  # 10% of portfolio
            'stop_loss_pct': 0.05,     # 5% stop loss
            'take_profit_pct': 0.15,   # 15% take profit
            'max_drawdown': 0.20       # 20% max drawdown
        }
    
    def generate_enhanced_signals(self, ticker: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate enhanced trading signals combining multiple analysis methods.
        
        Args:
            ticker: Stock ticker symbol
            df: DataFrame with price data and indicators
            
        Returns:
            DataFrame with enhanced trading signals
        """
        print(f"📈 Generating enhanced trading signals for {ticker}...")
        
        try:
            if df is None:
                print("❌ No data provided for signal generation")
                return pd.DataFrame()
            
            # Add technical indicators if not present
            if not any(col.startswith('RSI') for col in df.columns):
                df = self.technical_analyzer.add_all_indicators(df)
            
            # Generate signals
            signals_df = self._generate_comprehensive_signals(df, ticker)
            
            # Add risk management
            signals_df = self._add_risk_management(signals_df)
            
            # Add position sizing
            signals_df = self._add_position_sizing(signals_df)
            
            print(f"✅ Generated enhanced signals for {ticker}")
            return signals_df
            
        except Exception as e:
            print(f"❌ Error generating enhanced signals: {e}")
            return pd.DataFrame()
    
    def _generate_comprehensive_signals(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate comprehensive trading signals.
        
        Args:
            df: DataFrame with price data and indicators
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with comprehensive signals
        """
        df = df.copy()
        df['Signal'] = 'HOLD'
        df['Signal_Strength'] = 0.0
        df['Signal_Reason'] = ''
        df['Technical_Score'] = 0.0
        df['Sentiment_Score'] = 0.0
        df['Prediction_Score'] = 0.0
        df['Overall_Score'] = 0.0
        
        for i in range(20, len(df)):  # Start from 20 to ensure indicators are available
            # Technical analysis score
            technical_score = self._calculate_technical_score(df, i)
            df.loc[df.index[i], 'Technical_Score'] = technical_score
            
            # Sentiment analysis score
            sentiment_score = self._calculate_sentiment_score(df, i, ticker)
            df.loc[df.index[i], 'Sentiment_Score'] = sentiment_score
            
            # Prediction score (if available)
            prediction_score = self._calculate_prediction_score(df, i)
            df.loc[df.index[i], 'Prediction_Score'] = prediction_score
            
            # Overall score (weighted combination)
            overall_score = (
                technical_score * self.strategy_weights['technical_indicators'] +
                sentiment_score * self.strategy_weights['sentiment_analysis'] +
                prediction_score * self.strategy_weights['lstm_prediction']
            )
            df.loc[df.index[i], 'Overall_Score'] = overall_score
            
            # Generate signal based on overall score
            signal, strength, reason = self._determine_signal(overall_score, df, i)
            df.loc[df.index[i], 'Signal'] = signal
            df.loc[df.index[i], 'Signal_Strength'] = strength
            df.loc[df.index[i], 'Signal_Reason'] = reason
        
        return df
    
    def _calculate_technical_score(self, df: pd.DataFrame, i: int) -> float:
        """
        Calculate technical analysis score.
        
        Args:
            df: DataFrame with technical indicators
            i: Current index
            
        Returns:
            Technical score between -1 and 1
        """
        score = 0.0
        count = 0
        
        # Price vs Moving Averages
        if all(col in df.columns for col in ['SMA_10', 'SMA_20', 'SMA_50']):
            close_price = df['Close'].iloc[i]
            sma_10 = df['SMA_10'].iloc[i]
            sma_20 = df['SMA_20'].iloc[i]
            sma_50 = df['SMA_50'].iloc[i]
            
            # Positive if price above all MAs
            if close_price > sma_10 and close_price > sma_20 and close_price > sma_50:
                score += 0.3
            # Negative if price below all MAs
            elif close_price < sma_10 and close_price < sma_20 and close_price < sma_50:
                score -= 0.3
            count += 1
        
        # RSI Analysis
        if 'RSI_14' in df.columns and not pd.isna(df['RSI_14'].iloc[i]):
            rsi = df['RSI_14'].iloc[i]
            if rsi < 30:  # Oversold
                score += 0.2
            elif rsi > 70:  # Overbought
                score -= 0.2
            count += 1
        
        # MACD Analysis
        if all(col in df.columns for col in ['MACD', 'Signal_Line']):
            macd = df['MACD'].iloc[i]
            signal = df['Signal_Line'].iloc[i]
            macd_prev = df['MACD'].iloc[i-1]
            signal_prev = df['Signal_Line'].iloc[i-1]
            
            if macd > signal and macd_prev <= signal_prev:
                score += 0.3  # Bullish crossover
            elif macd < signal and macd_prev >= signal_prev:
                score -= 0.3  # Bearish crossover
            count += 1
        
        # Bollinger Bands Analysis
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            close_price = df['Close'].iloc[i]
            bb_upper = df['BB_Upper'].iloc[i]
            bb_lower = df['BB_Lower'].iloc[i]
            
            if close_price < bb_lower:  # Oversold
                score += 0.2
            elif close_price > bb_upper:  # Overbought
                score -= 0.2
            count += 1
        
        # Stochastic Analysis
        if 'Stoch_K' in df.columns and not pd.isna(df['Stoch_K'].iloc[i]):
            stoch_k = df['Stoch_K'].iloc[i]
            if stoch_k < 20:  # Oversold
                score += 0.15
            elif stoch_k > 80:  # Overbought
                score -= 0.15
            count += 1
        
        # Volume Analysis
        if 'Volume_Price_Confirmation' in df.columns:
            volume_conf = df['Volume_Price_Confirmation'].iloc[i]
            score += volume_conf * 0.1
            count += 1
        
        # Normalize score
        if count > 0:
            score = score / count
        
        return max(-1, min(1, score))
    
    def _calculate_sentiment_score(self, df: pd.DataFrame, i: int, ticker: str) -> float:
        """
        Calculate sentiment analysis score.
        
        Args:
            df: DataFrame with data
            i: Current index
            ticker: Stock ticker symbol
            
        Returns:
            Sentiment score between -1 and 1
        """
        if self.sentiment_analyzer is None:
            return 0.0
        
        try:
            # Get sentiment data for the current date
            current_date = df.index[i]
            
            # Analyze sentiment for the ticker
            sentiment_df = self.sentiment_analyzer.analyze_stock_sentiment(ticker, days_back=30)
            
            if not sentiment_df.empty:
                # Find sentiment for the current date or use the most recent
                date_sentiment = sentiment_df[sentiment_df['date'].dt.date == current_date.date()]
                if not date_sentiment.empty:
                    return date_sentiment['overall_sentiment'].iloc[0]
                else:
                    # Use the most recent sentiment
                    return sentiment_df['overall_sentiment'].iloc[-1]
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating sentiment score: {e}")
            return 0.0
    
    def _calculate_prediction_score(self, df: pd.DataFrame, i: int) -> float:
        """
        Calculate prediction score from LSTM or other ML models.
        
        Args:
            df: DataFrame with predictions
            i: Current index
            
        Returns:
            Prediction score between -1 and 1
        """
        try:
            # Look for prediction columns
            prediction_cols = [col for col in df.columns if 'prediction' in col.lower() or 'forecast' in col.lower()]
            
            if prediction_cols:
                # Use the first prediction column found
                pred_col = prediction_cols[0]
                if not pd.isna(df[pred_col].iloc[i]):
                    current_price = df['Close'].iloc[i]
                    predicted_price = df[pred_col].iloc[i]
                    
                    # Calculate percentage change
                    pct_change = (predicted_price - current_price) / current_price
                    
                    # Normalize to -1 to 1 range
                    return max(-1, min(1, pct_change * 10))  # Scale factor of 10
            
            return 0.0
            
        except Exception as e:
            print(f"Error calculating prediction score: {e}")
            return 0.0
    
    def _determine_signal(self, overall_score: float, df: pd.DataFrame, i: int) -> Tuple[str, float, str]:
        """
        Determine trading signal based on overall score.
        
        Args:
            overall_score: Overall analysis score
            df: DataFrame with data
            i: Current index
            
        Returns:
            Tuple of (signal, strength, reason)
        """
        reasons = []
        
        # Determine signal based on score
        if overall_score > self.signal_thresholds['strong_buy']:
            signal = 'STRONG_BUY'
            strength = overall_score
        elif overall_score > self.signal_thresholds['buy']:
            signal = 'BUY'
            strength = overall_score
        elif overall_score < self.signal_thresholds['sell']:
            signal = 'SELL'
            strength = abs(overall_score)
        elif overall_score < self.signal_thresholds['strong_buy']:
            signal = 'STRONG_SELL'
            strength = abs(overall_score)
        else:
            signal = 'HOLD'
            strength = 0.0
        
        # Add reasons based on individual scores
        technical_score = df['Technical_Score'].iloc[i]
        sentiment_score = df['Sentiment_Score'].iloc[i]
        prediction_score = df['Prediction_Score'].iloc[i]
        
        if technical_score > 0.3:
            reasons.append('Strong technical indicators')
        elif technical_score < -0.3:
            reasons.append('Weak technical indicators')
        
        if sentiment_score > 0.3:
            reasons.append('Positive sentiment')
        elif sentiment_score < -0.3:
            reasons.append('Negative sentiment')
        
        if prediction_score > 0.3:
            reasons.append('Bullish prediction')
        elif prediction_score < -0.3:
            reasons.append('Bearish prediction')
        
        if not reasons:
            reasons.append('Mixed signals')
        
        return signal, strength, '; '.join(reasons)
    
    def _add_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add risk management features to signals.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with risk management features
        """
        df = df.copy()
        
        # Add stop loss levels
        df['Stop_Loss'] = np.nan
        df['Take_Profit'] = np.nan
        
        for i in range(len(df)):
            if df['Signal'].iloc[i] in ['BUY', 'STRONG_BUY']:
                current_price = df['Close'].iloc[i]
                df.loc[df.index[i], 'Stop_Loss'] = current_price * (1 - self.risk_params['stop_loss_pct'])
                df.loc[df.index[i], 'Take_Profit'] = current_price * (1 + self.risk_params['take_profit_pct'])
            elif df['Signal'].iloc[i] in ['SELL', 'STRONG_SELL']:
                current_price = df['Close'].iloc[i]
                df.loc[df.index[i], 'Stop_Loss'] = current_price * (1 + self.risk_params['stop_loss_pct'])
                df.loc[df.index[i], 'Take_Profit'] = current_price * (1 - self.risk_params['take_profit_pct'])
        
        # Add risk level
        df['Risk_Level'] = 'LOW'
        for i in range(len(df)):
            if df['Signal_Strength'].iloc[i] > 0.8:
                df.loc[df.index[i], 'Risk_Level'] = 'HIGH'
            elif df['Signal_Strength'].iloc[i] > 0.5:
                df.loc[df.index[i], 'Risk_Level'] = 'MEDIUM'
        
        return df
    
    def _add_position_sizing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add position sizing recommendations.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            DataFrame with position sizing
        """
        df = df.copy()
        df['Position_Size'] = 0.0
        df['Position_Recommendation'] = ''
        
        for i in range(len(df)):
            signal = df['Signal'].iloc[i]
            strength = df['Signal_Strength'].iloc[i]
            risk_level = df['Risk_Level'].iloc[i]
            
            if signal in ['BUY', 'STRONG_BUY']:
                # Base position size on signal strength and risk level
                base_size = self.risk_params['max_position_size']
                
                if risk_level == 'HIGH':
                    position_size = base_size * 0.5  # Reduce size for high risk
                elif risk_level == 'MEDIUM':
                    position_size = base_size * 0.75
                else:
                    position_size = base_size
                
                # Adjust for signal strength
                position_size *= strength
                
                df.loc[df.index[i], 'Position_Size'] = position_size
                df.loc[df.index[i], 'Position_Recommendation'] = f'Buy {position_size:.1%} of portfolio'
                
            elif signal in ['SELL', 'STRONG_SELL']:
                # For sell signals, recommend reducing position
                position_size = strength * self.risk_params['max_position_size']
                df.loc[df.index[i], 'Position_Size'] = -position_size
                df.loc[df.index[i], 'Position_Recommendation'] = f'Sell {position_size:.1%} of position'
            
            else:  # HOLD
                df.loc[df.index[i], 'Position_Recommendation'] = 'Maintain current position'
        
        return df
    
    def backtest_strategy(self, df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
        """
        Backtest the trading strategy.
        
        Args:
            df: DataFrame with signals and price data
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        print("🔄 Backtesting trading strategy...")
        
        try:
            # Initialize backtest variables
            capital = initial_capital
            position = 0
            trades = []
            equity_curve = []
            
            for i in range(len(df)):
                current_price = df['Close'].iloc[i]
                signal = df['Signal'].iloc[i]
                position_size = df['Position_Size'].iloc[i]
                
                # Execute trades based on signals
                if signal in ['BUY', 'STRONG_BUY'] and position <= 0:
                    # Buy signal
                    shares_to_buy = int((capital * position_size) / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        capital -= cost
                        position += shares_to_buy
                        
                        trades.append({
                            'date': df.index[i],
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
                
                elif signal in ['SELL', 'STRONG_SELL'] and position > 0:
                    # Sell signal
                    shares_to_sell = int(position * abs(position_size))
                    if shares_to_sell > 0:
                        proceeds = shares_to_sell * current_price
                        capital += proceeds
                        position -= shares_to_sell
                        
                        trades.append({
                            'date': df.index[i],
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_price,
                            'proceeds': proceeds
                        })
                
                # Calculate current equity
                current_equity = capital + (position * current_price)
                equity_curve.append({
                    'date': df.index[i],
                    'equity': current_equity,
                    'capital': capital,
                    'position': position
                })
            
            # Calculate performance metrics
            equity_df = pd.DataFrame(equity_curve)
            equity_df['returns'] = equity_df['equity'].pct_change()
            
            total_return = (equity_df['equity'].iloc[-1] - initial_capital) / initial_capital
            annualized_return = total_return * (252 / len(equity_df))
            volatility = equity_df['returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            equity_df['peak'] = equity_df['equity'].expanding().max()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()
            
            results = {
                'initial_capital': initial_capital,
                'final_equity': equity_df['equity'].iloc[-1],
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'trades': trades,
                'equity_curve': equity_curve
            }
            
            print(f"✅ Backtest completed. Total return: {total_return:.2%}")
            return results
            
        except Exception as e:
            print(f"❌ Error in backtesting: {e}")
            return {}

# Convenience functions for backward compatibility
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatibility function."""
    strategy = OptimizedTradingStrategy()
    return strategy.generate_enhanced_signals("", df)

def backtest(df: pd.DataFrame, initial_capital: float = 10000) -> Tuple[pd.DataFrame, Dict, List]:
    """Backward compatibility function."""
    strategy = OptimizedTradingStrategy()
    results = strategy.backtest_strategy(df, initial_capital)
    
    # Return format expected by existing code
    backtest_df = pd.DataFrame(results.get('equity_curve', []))
    metrics = {k: v for k, v in results.items() if k not in ['trades', 'equity_curve']}
    trades = results.get('trades', [])
    
    return backtest_df, metrics, trades
