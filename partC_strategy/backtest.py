import pandas as pd
import numpy as np
from typing import Dict, List

def backtest(df, initial_cash=10000):
    """
    Simple backtesting engine.
    
    Args:
        df (pd.DataFrame): DataFrame with signals and price data
        initial_cash (float): Initial capital for backtesting
        
    Returns:
        tuple: (df, metrics, trades)
    """
    df = df.copy()
    
    # Find the Close column dynamically
    close_col = None
    for col in df.columns:
        if col.startswith('Close_'):
            close_col = col
            break
    
    if close_col is None:
        # Fallback: look for any column containing 'Close'
        for col in df.columns:
            if 'Close' in col:
                close_col = col
                break
    
    if close_col is None:
        print("⚠️ Could not find Close column, using first column as price")
        close_col = df.columns[0]
    
    cash = initial_cash
    position = 0
    trades = []
    
    # Add portfolio value tracking
    df['Portfolio_Value'] = initial_cash
    df['Cash'] = initial_cash
    df['Position'] = 0
    df['Shares'] = 0
    
    for i in range(len(df)):
        current_price = df[close_col].iloc[i]
        signal = df['Signal'].iloc[i]
        
        # Execute trades
        if signal == "BUY" and cash >= current_price and position == 0:
            # Buy signal - use all available cash
            shares = cash // current_price
            position = shares
            cash -= shares * current_price
            
            trades.append({
                'date': df.index[i] if hasattr(df.index[i], 'strftime') else i,
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'cash_after': cash,
                'portfolio_value': cash + (position * current_price)
            })
            
        elif signal == "SELL" and position > 0:
            # Sell signal - sell all shares
            cash += position * current_price
            
            trades.append({
                'date': df.index[i] if hasattr(df.index[i], 'strftime') else i,
                'action': 'SELL',
                'price': current_price,
                'shares': position,
                'cash_after': cash,
                'portfolio_value': cash
            })
            
            position = 0
        
        # Update portfolio tracking
        df.at[i, 'Cash'] = cash
        df.at[i, 'Shares'] = position
        df.at[i, 'Position'] = position * current_price
        df.at[i, 'Portfolio_Value'] = cash + (position * current_price)
    
    # Calculate final metrics
    final_value = cash + (position * df[close_col].iloc[-1])
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    
    # Calculate additional metrics
    buy_hold_return = ((df[close_col].iloc[-1] - df[close_col].iloc[0]) / df[close_col].iloc[0]) * 100
    
    # Calculate max drawdown
    portfolio_values = df['Portfolio_Value'].values
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Calculate Sharpe ratio (simplified)
    returns = df['Portfolio_Value'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    
    metrics = {
        'Initial Value': initial_cash,
        'Final Value': final_value,
        'Total Return (%)': total_return,
        'Buy & Hold Return (%)': buy_hold_return,
        'Number of Trades': len(trades),
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate (%)': 0  # Would need to calculate based on trade profitability
    }
    
    # Calculate win rate if we have trades
    if len(trades) > 0:
        profitable_trades = 0
        for i in range(0, len(trades), 2):  # Check buy-sell pairs
            if i + 1 < len(trades):
                buy_price = trades[i]['price']
                sell_price = trades[i + 1]['price']
                if sell_price > buy_price:
                    profitable_trades += 1
        
        metrics['Win Rate (%)'] = (profitable_trades / (len(trades) // 2)) * 100 if len(trades) > 1 else 0
    
    print(f"💰 Backtest Results:")
    print(f"   Initial Capital: ${initial_cash:,.2f}")
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:.2f}%")
    print(f"   Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"   Number of Trades: {len(trades)}")
    print(f"   Max Drawdown: {max_drawdown:.2f}%")
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")
    
    return df, metrics, trades

class BacktestStrategy:
    """Backtesting strategy class for enhanced analysis."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run_backtest(self, signals_df: pd.DataFrame) -> Dict:
        """
        Run backtest on signals DataFrame.
        
        Args:
            signals_df: DataFrame with signals and price data
            
        Returns:
            Dictionary with backtest results
        """
        try:
            if signals_df.empty:
                return None
            
            # Find price column
            price_col = None
            for col in signals_df.columns:
                if 'Close' in col or 'Price' in col:
                    price_col = col
                    break
            
            if price_col is None:
                price_col = signals_df.columns[0]  # Use first column as fallback
            
            # Initialize backtest variables
            capital = self.initial_capital
            position = 0
            trades = []
            equity_curve = []
            
            for i in range(len(signals_df)):
                current_price = signals_df[price_col].iloc[i]
                signal = signals_df['Signal'].iloc[i] if 'Signal' in signals_df.columns else 'HOLD'
                
                # Execute trades based on signals
                if signal in ['BUY', 'STRONG_BUY'] and position <= 0:
                    # Buy signal
                    shares_to_buy = int(capital / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        capital -= cost
                        position += shares_to_buy
                        
                        trades.append({
                            'date': signals_df.index[i],
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost
                        })
                
                elif signal in ['SELL', 'STRONG_SELL'] and position > 0:
                    # Sell signal
                    proceeds = position * current_price
                    capital += proceeds
                    
                    trades.append({
                        'date': signals_df.index[i],
                        'action': 'SELL',
                        'shares': position,
                        'price': current_price,
                        'proceeds': proceeds
                    })
                    
                    position = 0
                
                # Calculate current equity
                current_equity = capital + (position * current_price)
                equity_curve.append({
                    'date': signals_df.index[i],
                    'equity': current_equity,
                    'capital': capital,
                    'position': position
                })
            
            # Calculate performance metrics
            if equity_curve:
                equity_df = pd.DataFrame(equity_curve)
                equity_df['returns'] = equity_df['equity'].pct_change()
                
                final_equity = equity_df['equity'].iloc[-1]
                total_return = (final_equity - self.initial_capital) / self.initial_capital
                
                # Calculate annualized return
                if len(equity_df) > 1:
                    days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
                    annualized_return = total_return * (365 / days) if days > 0 else 0
                else:
                    annualized_return = 0
                
                # Calculate volatility
                volatility = equity_df['returns'].std() * np.sqrt(252) if len(equity_df) > 1 else 0
                
                # Calculate Sharpe ratio
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                
                # Calculate drawdown
                equity_df['peak'] = equity_df['equity'].expanding().max()
                equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
                max_drawdown = equity_df['drawdown'].min()
                
                # Calculate buy and hold return
                if len(signals_df) > 1:
                    buy_hold_return = (signals_df[price_col].iloc[-1] / signals_df[price_col].iloc[0] - 1)
                else:
                    buy_hold_return = 0
                
                results = {
                    'initial_capital': self.initial_capital,
                    'final_equity': final_equity,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'buy_hold_return': buy_hold_return,
                    'total_trades': len(trades),
                    'win_rate': self._calculate_win_rate(trades),
                    'profit_factor': self._calculate_profit_factor(trades),
                    'analysis_date': pd.Timestamp.now().isoformat()
                }
                
                return results
            else:
                return None
                
        except Exception as e:
            print(f"Error in backtesting: {e}")
            return None
    
    def _calculate_win_rate(self, trades: List) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        
        profitable_trades = 0
        total_trades = 0
        
        for i in range(0, len(trades), 2):  # Process buy-sell pairs
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    profit = sell_trade['proceeds'] - buy_trade['cost']
                    if profit > 0:
                        profitable_trades += 1
                    total_trades += 1
        
        return profitable_trades / total_trades if total_trades > 0 else 0.0
    
    def _calculate_profit_factor(self, trades: List) -> float:
        """Calculate profit factor from trades."""
        if not trades:
            return 0.0
        
        total_profit = 0.0
        total_loss = 0.0
        
        for i in range(0, len(trades), 2):  # Process buy-sell pairs
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    profit = sell_trade['proceeds'] - buy_trade['cost']
                    if profit > 0:
                        total_profit += profit
                    else:
                        total_loss += abs(profit)
        
        return total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0
