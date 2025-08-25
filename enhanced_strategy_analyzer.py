#!/usr/bin/env python3
"""
Enhanced Strategy Analyzer
Combines technical analysis with market sentiment and economic factors
"""

import pandas as pd
import numpy as np
import os
import requests
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedStrategyAnalyzer:
    """
    Enhanced strategy analyzer with sentiment and economic factors
    """
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        self.sentiment_data = {}
        self.economic_data = {}
        self.market_data = {}
        
    def analyze_comprehensive_strategy(self, df):
        """Analyze comprehensive strategy combining all factors."""
        try:
            print("🔍 Analyzing comprehensive strategy...")
            
            # Technical analysis
            technical_signals = self._analyze_technical_signals(df)
            
            # Market sentiment analysis
            sentiment_signals = self._analyze_market_sentiment()
            
            # Economic factors analysis
            economic_signals = self._analyze_economic_factors()
            
            # Market regime analysis
            regime_signals = self._analyze_market_regime(df)
            
            # Combine all signals
            combined_strategy = self._combine_signals(
                technical_signals, sentiment_signals, 
                economic_signals, regime_signals
            )
            
            return combined_strategy
            
        except Exception as e:
            print(f"❌ Error in comprehensive strategy analysis: {e}")
            return None
    
    def _analyze_technical_signals(self, df):
        """Analyze technical indicators and generate signals."""
        try:
            signals = {
                'trend': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'volatility': 'NEUTRAL',
                'volume': 'NEUTRAL',
                'strength': 0.0
            }
            
            # Get latest values
            latest = df.iloc[-1]
            
            # Trend analysis
            if 'EMA_5' in latest and 'EMA_20' in latest:
                if latest['EMA_5'] > latest['EMA_20']:
                    signals['trend'] = 'BULLISH'
                    signals['strength'] += 0.2
                else:
                    signals['trend'] = 'BEARISH'
                    signals['strength'] -= 0.2
            
            # Momentum analysis
            if 'RSI_14' in latest:
                rsi = latest['RSI_14']
                if rsi > 70:
                    signals['momentum'] = 'OVERBOUGHT'
                    signals['strength'] -= 0.1
                elif rsi < 30:
                    signals['momentum'] = 'OVERSOLD'
                    signals['strength'] += 0.1
                elif rsi > 50:
                    signals['momentum'] = 'BULLISH'
                    signals['strength'] += 0.1
                else:
                    signals['momentum'] = 'BEARISH'
                    signals['strength'] -= 0.1
            
            # MACD analysis
            if 'MACD' in latest and 'MACD_Signal' in latest:
                if latest['MACD'] > latest['MACD_Signal']:
                    signals['strength'] += 0.15
                else:
                    signals['strength'] -= 0.15
            
            # Volume analysis
            if 'Volume_Ratio' in latest:
                vol_ratio = latest['Volume_Ratio']
                if vol_ratio > 1.5:
                    signals['volume'] = 'HIGH'
                    signals['strength'] += 0.1
                elif vol_ratio < 0.5:
                    signals['volume'] = 'LOW'
                    signals['strength'] -= 0.1
            
            # Bollinger Bands analysis
            if 'BB_Position' in latest:
                bb_pos = latest['BB_Position']
                if bb_pos > 0.8:
                    signals['strength'] -= 0.1  # Near upper band
                elif bb_pos < 0.2:
                    signals['strength'] += 0.1  # Near lower band
            
            return signals
            
        except Exception as e:
            print(f"Warning: Error in technical analysis: {e}")
            return {'trend': 'NEUTRAL', 'momentum': 'NEUTRAL', 'volatility': 'NEUTRAL', 'volume': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_market_sentiment(self):
        """Analyze market sentiment factors."""
        try:
            signals = {
                'sentiment': 'NEUTRAL',
                'strength': 0.0
            }
            
            # Get stock info
            stock = yf.Ticker(self.ticker)
            
            # Analyst recommendations
            try:
                recommendations = stock.recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.iloc[-1]
                    if 'To Grade' in latest_rec:
                        grade = latest_rec['To Grade']
                        if grade in ['Buy', 'Strong Buy']:
                            signals['sentiment'] = 'BULLISH'
                            signals['strength'] += 0.2
                        elif grade in ['Sell', 'Strong Sell']:
                            signals['sentiment'] = 'BEARISH'
                            signals['strength'] -= 0.2
            except:
                pass
            
            # Institutional ownership
            try:
                institutional = stock.institutional_holders
                if institutional is not None and not institutional.empty:
                    total_ownership = institutional['Shares'].sum()
                    if total_ownership > 0:
                        signals['strength'] += 0.1  # Positive institutional presence
            except:
                pass
            
            # News sentiment (simplified)
            signals['strength'] += 0.05  # Neutral news sentiment
            
            return signals
            
        except Exception as e:
            print(f"Warning: Error in sentiment analysis: {e}")
            return {'sentiment': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_economic_factors(self):
        """Analyze economic factors affecting the stock."""
        try:
            signals = {
                'economic': 'NEUTRAL',
                'strength': 0.0
            }
            
            # Get market data
            market_tickers = ['^GSPC', '^IXIC', '^DJI']  # S&P 500, NASDAQ, DOW
            market_data = {}
            
            for ticker in market_tickers:
                try:
                    market = yf.Ticker(ticker)
                    hist = market.history(period='5d')
                    if not hist.empty:
                        market_data[ticker] = {
                            'current': hist['Close'].iloc[-1],
                            'change_5d': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                        }
                except:
                    continue
            
            # Analyze market correlation
            if market_data:
                avg_market_change = np.mean([data['change_5d'] for data in market_data.values()])
                
                if avg_market_change > 1:
                    signals['economic'] = 'BULLISH'
                    signals['strength'] += 0.15
                elif avg_market_change < -1:
                    signals['economic'] = 'BEARISH'
                    signals['strength'] -= 0.15
            
            # Sector performance (simplified)
            signals['strength'] += 0.05  # Neutral sector performance
            
            return signals
            
        except Exception as e:
            print(f"Warning: Error in economic analysis: {e}")
            return {'economic': 'NEUTRAL', 'strength': 0.0}
    
    def _analyze_market_regime(self, df):
        """Analyze market regime and volatility."""
        try:
            signals = {
                'regime': 'NEUTRAL',
                'volatility': 'NORMAL',
                'strength': 0.0
            }
            
            # Calculate volatility
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
                
                if volatility > 0.03:  # High volatility
                    signals['volatility'] = 'HIGH'
                    signals['strength'] -= 0.1
                elif volatility < 0.01:  # Low volatility
                    signals['volatility'] = 'LOW'
                    signals['strength'] += 0.05
            
            # Trend strength
            if 'Trend_Strength' in df.columns:
                trend_strength = df['Trend_Strength'].iloc[-1]
                if trend_strength > 0.05:
                    signals['regime'] = 'TRENDING'
                    signals['strength'] += 0.1
                elif trend_strength < 0.01:
                    signals['regime'] = 'SIDEWAYS'
                    signals['strength'] -= 0.05
            
            return signals
            
        except Exception as e:
            print(f"Warning: Error in market regime analysis: {e}")
            return {'regime': 'NEUTRAL', 'volatility': 'NORMAL', 'strength': 0.0}
    
    def _combine_signals(self, technical, sentiment, economic, regime):
        """Combine all signals into a comprehensive strategy."""
        try:
            # Calculate total strength
            total_strength = (
                technical['strength'] * 0.4 +  # Technical analysis weight
                sentiment['strength'] * 0.25 +  # Sentiment weight
                economic['strength'] * 0.2 +   # Economic weight
                regime['strength'] * 0.15      # Market regime weight
            )
            
            # Determine overall signal
            if total_strength > 0.3:
                signal = "STRONG_BUY"
                confidence = "HIGH"
            elif total_strength > 0.1:
                signal = "BUY"
                confidence = "MEDIUM"
            elif total_strength < -0.3:
                signal = "STRONG_SELL"
                confidence = "HIGH"
            elif total_strength < -0.1:
                signal = "SELL"
                confidence = "MEDIUM"
            else:
                signal = "HOLD"
                confidence = "LOW"
            
            # Create comprehensive strategy
            strategy = {
                'signal': signal,
                'confidence': confidence,
                'strength': total_strength,
                'technical': technical,
                'sentiment': sentiment,
                'economic': economic,
                'regime': regime,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return strategy
            
        except Exception as e:
            print(f"Warning: Error combining signals: {e}")
            return None
    
    def generate_risk_assessment(self, df, prediction_price, current_price):
        """Generate risk assessment for the prediction."""
        try:
            risk_assessment = {
                'risk_level': 'MEDIUM',
                'stop_loss': 0.0,
                'take_profit': 0.0,
                'position_size': 'MEDIUM',
                'timeframe': 'MEDIUM'
            }
            
            # Calculate volatility
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                volatility = returns.rolling(20).std().iloc[-1]
                
                # Risk level based on volatility
                if volatility > 0.04:
                    risk_assessment['risk_level'] = 'HIGH'
                    risk_assessment['position_size'] = 'SMALL'
                elif volatility < 0.015:
                    risk_assessment['risk_level'] = 'LOW'
                    risk_assessment['position_size'] = 'LARGE'
            
            # Calculate stop loss and take profit
            expected_return = (prediction_price - current_price) / current_price
            
            if expected_return > 0:
                # Bullish prediction
                risk_assessment['stop_loss'] = current_price * 0.95  # 5% stop loss
                risk_assessment['take_profit'] = prediction_price * 1.05  # 5% above prediction
            else:
                # Bearish prediction
                risk_assessment['stop_loss'] = current_price * 1.05  # 5% stop loss
                risk_assessment['take_profit'] = prediction_price * 0.95  # 5% below prediction
            
            # Timeframe recommendation
            if abs(expected_return) > 0.05:  # 5% expected return
                risk_assessment['timeframe'] = 'SHORT'
            elif abs(expected_return) > 0.02:  # 2% expected return
                risk_assessment['timeframe'] = 'MEDIUM'
            else:
                risk_assessment['timeframe'] = 'LONG'
            
            return risk_assessment
            
        except Exception as e:
            print(f"Warning: Error in risk assessment: {e}")
            return None
    
    def save_strategy_report(self, strategy, risk_assessment, ticker):
        """Save strategy report to file."""
        try:
            report = {
                'ticker': ticker,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'strategy': strategy,
                'risk_assessment': risk_assessment
            }
            
            # Save to CSV
            report_df = pd.DataFrame([{
                'Ticker': ticker,
                'Timestamp': report['timestamp'],
                'Signal': strategy['signal'],
                'Confidence': strategy['confidence'],
                'Strength': strategy['strength'],
                'Risk_Level': risk_assessment['risk_level'],
                'Stop_Loss': risk_assessment['stop_loss'],
                'Take_Profit': risk_assessment['take_profit'],
                'Position_Size': risk_assessment['position_size'],
                'Timeframe': risk_assessment['timeframe']
            }])
            
            report_df.to_csv(f"data/{ticker}_enhanced_strategy_report.csv", index=False)
            print(f"💾 Strategy report saved: data/{ticker}_enhanced_strategy_report.csv")
            
            return True
            
        except Exception as e:
            print(f"Warning: Error saving strategy report: {e}")
            return False

def display_enhanced_strategy_results(ticker, strategy, risk_assessment, current_price, prediction_price):
    """Display enhanced strategy results."""
    print("\n🎯 ENHANCED STRATEGY ANALYSIS")
    print("=" * 70)
    print(f"📊 Stock: {ticker}")
    print(f"💰 Current Price: ₹{current_price:.2f}")
    print(f"🎯 Predicted Price: ₹{prediction_price:.2f}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)
    print()
    
    # Display strategy signal
    print("💡 STRATEGY SIGNAL:")
    print(f"   Signal: {strategy['signal']}")
    print(f"   Confidence: {strategy['confidence']}")
    print(f"   Strength: {strategy['strength']:.3f}")
    print()
    
    # Display component analysis
    print("🔍 COMPONENT ANALYSIS:")
    print(f"   Technical: {strategy['technical']['trend']} (Strength: {strategy['technical']['strength']:.3f})")
    print(f"   Sentiment: {strategy['sentiment']['sentiment']} (Strength: {strategy['sentiment']['strength']:.3f})")
    print(f"   Economic: {strategy['economic']['economic']} (Strength: {strategy['economic']['strength']:.3f})")
    print(f"   Market Regime: {strategy['regime']['regime']} (Strength: {strategy['regime']['strength']:.3f})")
    print()
    
    # Display risk assessment
    print("⚠️ RISK ASSESSMENT:")
    print(f"   Risk Level: {risk_assessment['risk_level']}")
    print(f"   Position Size: {risk_assessment['position_size']}")
    print(f"   Timeframe: {risk_assessment['timeframe']}")
    print(f"   Stop Loss: ₹{risk_assessment['stop_loss']:.2f}")
    print(f"   Take Profit: ₹{risk_assessment['take_profit']:.2f}")
    print()
    
    # Display trading recommendation
    print("📋 TRADING RECOMMENDATION:")
    if strategy['signal'] == "STRONG_BUY":
        print("   🟢 STRONG BUY - High confidence bullish signal")
        print("   📈 Expected significant upward movement")
    elif strategy['signal'] == "BUY":
        print("   🟡 BUY - Moderate confidence bullish signal")
        print("   📈 Expected moderate upward movement")
    elif strategy['signal'] == "STRONG_SELL":
        print("   🔴 STRONG SELL - High confidence bearish signal")
        print("   📉 Expected significant downward movement")
    elif strategy['signal'] == "SELL":
        print("   🟠 SELL - Moderate confidence bearish signal")
        print("   📉 Expected moderate downward movement")
    else:
        print("   ⚪ HOLD - Neutral signal, wait for better opportunities")
        print("   ➡️ No clear directional bias")
    
    print()
    print("=" * 70)

def run_enhanced_strategy_analysis(ticker, df, prediction_price):
    """Run enhanced strategy analysis."""
    try:
        print(f"🔍 Running enhanced strategy analysis for {ticker}...")
        
        # Initialize analyzer
        analyzer = EnhancedStrategyAnalyzer(ticker)
        
        # Get current price
        current_price = float(df['Close'].iloc[-1])
        
        # Analyze comprehensive strategy
        strategy = analyzer.analyze_comprehensive_strategy(df)
        if strategy is None:
            return False
        
        # Generate risk assessment
        risk_assessment = analyzer.generate_risk_assessment(df, prediction_price, current_price)
        if risk_assessment is None:
            return False
        
        # Save strategy report
        analyzer.save_strategy_report(strategy, risk_assessment, ticker)
        
        # Display results
        display_enhanced_strategy_results(ticker, strategy, risk_assessment, current_price, prediction_price)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in enhanced strategy analysis: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Strategy Analyzer")
    print("=" * 50)
    
    # This module is designed to be used with the improved prediction engine
    print("This module provides enhanced strategy analysis capabilities.")
    print("Use it in conjunction with the improved prediction engine.")
