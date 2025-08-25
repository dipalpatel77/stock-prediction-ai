import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# Optional dependency to fetch data if user wants (yfinance used elsewhere in repo)
try:
    import yfinance as yf
except Exception:
    yf = None

class SectorTrendAnalyzer:
    """
    Analyze sector / group trend from price series and produce:
      - momentum returns (3/6/12 months)
      - moving-average trend (SMA50 vs SMA200)
      - realized volatility
      - aggregated score and recommendation: OVERWEIGHT / NEUTRAL / UNDERWEIGHT

    Use either:
      - analyze_sector_from_prices(price_df): where price_df columns are tickers or members
      - analyze_sector_by_tickers(tickers, start, end): will fetch via yfinance if available
    """

    def __init__(self, benchmark: str = "SPY"):
        self.benchmark = benchmark

    def fetch_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance not installed; pass price DataFrame instead")
        data = yf.download(tickers, start=start, end=end, progress=False, threads=False)["Adj Close"]
        # if single ticker, ensure DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.sort_index()

    def _period_return(self, series: pd.Series, days: int) -> float:
        if len(series) < days + 1:
            return float("nan")
        return (series.iloc[-1] / series.iloc[-1 - days] - 1.0) * 100.0

    def analyze_sector_from_prices(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        price_df: DataFrame indexed by date with columns = tickers (Adj Close)
        Returns summary DataFrame indexed by ticker with columns:
          - ret_63, ret_126, ret_252 (approx 3/6/12 months)
          - sma50, sma200, sma_trend (1 if sma50>sma200 else 0)
          - vol_21 (21-day annualized vol %)
          - momentum_z (z-scored mean of returns)
          - vol_z (z-scored vol)
          - score (aggregate), recommendation
        """
        df = price_df.copy().dropna(how="all")
        # ensure enough history
        summary = []
        for col in df.columns:
            s = df[col].dropna()
            if s.empty:
                continue
            ret_63 = self._period_return(s, 63)
            ret_126 = self._period_return(s, 126)
            ret_252 = self._period_return(s, 252)
            sma50 = s.rolling(50).mean().iat[-1] if len(s) >= 50 else np.nan
            sma200 = s.rolling(200).mean().iat[-1] if len(s) >= 200 else np.nan
            sma_trend = 1 if (pd.notna(sma50) and pd.notna(sma200) and sma50 > sma200) else 0
            # realized vol (21-day), annualized %
            returns = s.pct_change().dropna()
            vol_21 = returns.rolling(21).std().iat[-1] * np.sqrt(252) * 100 if len(returns) >= 21 else np.nan
            summary.append({
                "ticker": col,
                "ret_63": ret_63,
                "ret_126": ret_126,
                "ret_252": ret_252,
                "sma50": sma50,
                "sma200": sma200,
                "sma_trend": sma_trend,
                "vol_21": vol_21
            })
        summary_df = pd.DataFrame(summary).set_index("ticker")

        # momentum: z-score each return column across tickers, then average
        for c in ["ret_63", "ret_126", "ret_252"]:
            if c in summary_df.columns:
                summary_df[f"{c}_z"] = (summary_df[c] - summary_df[c].mean()) / (summary_df[c].std(ddof=0) if summary_df[c].std(ddof=0) != 0 else 1.0)

        # combine momentum z-scores (prefer recent more)
        z_cols = [c for c in summary_df.columns if c.endswith("_z")]
        if z_cols:
            weights = {"ret_63_z": 0.5, "ret_126_z": 0.3, "ret_252_z": 0.2}
            momentum_vals = []
            for idx, row in summary_df.iterrows():
                mv = 0.0
                wsum = 0.0
                for z in z_cols:
                    w = weights.get(z, 0.0)
                    v = row.get(z, np.nan)
                    if pd.notna(v):
                        mv += w * v
                        wsum += w
                momentum_vals.append(mv / wsum if wsum > 0 else np.nan)
            summary_df["momentum_z"] = momentum_vals
        else:
            summary_df["momentum_z"] = np.nan

        # volatility z (lower vol preferred)
        if "vol_21" in summary_df.columns:
            summary_df["vol_z"] = (summary_df["vol_21"] - summary_df["vol_21"].mean()) / (summary_df["vol_21"].std(ddof=0) if summary_df["vol_21"].std(ddof=0) != 0 else 1.0)
        else:
            summary_df["vol_z"] = 0.0

        # aggregate score: momentum positive, trend adds, volatility subtracts
        # score = 0.7 * momentum_z + 0.2 * (sma_trend*1) - 0.1 * vol_z
        summary_df["score"] = 0.7 * summary_df["momentum_z"].fillna(0) + 0.2 * summary_df["sma_trend"].fillna(0) - 0.1 * summary_df["vol_z"].fillna(0)

        # recommendation thresholds
        def _rec(s):
            if pd.isna(s):
                return "NEUTRAL"
            if s >= 0.5:
                return "OVERWEIGHT"
            if s <= -0.5:
                return "UNDERWEIGHT"
            return "NEUTRAL"

        summary_df["recommendation"] = summary_df["score"].apply(_rec)

        return summary_df.sort_values("score", ascending=False)

    def analyze_sector_by_tickers(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance not available; pass price DataFrame to analyze_sector_from_prices")
        price_df = self.fetch_prices(tickers, start, end)
        return self.analyze_sector_from_prices(price_df)

    def recommend_allocation(self, summary_df: pd.DataFrame, base_alloc: float = 0.1) -> Dict[str, float]:
        """
        Return suggested allocation per ticker based on recommendation and score.
        base_alloc is default allocation for NEUTRAL.
        OVERWEIGHT gets base_alloc * (1 + normalized_score)
        UNDERWEIGHT gets base_alloc * max(0, 1 - abs(normalized_score))
        Normalization: min-max of score to 0..1
        """
        out = {}
        scores = summary_df["score"].copy()
        # normalize between 0..1
        if scores.max() == scores.min():
            norm = (scores - scores.min()) * 0.0
        else:
            norm = (scores - scores.min()) / (scores.max() - scores.min())
        for idx, row in summary_df.iterrows():
            rec = row["recommendation"]
            s_norm = norm.loc[idx]
            if rec == "OVERWEIGHT":
                out[idx] = base_alloc * (1.0 + s_norm)
            elif rec == "UNDERWEIGHT":
                out[idx] = max(0.0, base_alloc * (1.0 - s_norm))
            else:
                out[idx] = base_alloc
        return out

    def analyze_sector_trends(self, ticker: str) -> Dict:
        """
        Analyze sector trends for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with sector analysis results
        """
        try:
            # Get stock info to determine sector
            if yf is None:
                return {
                    'ticker': ticker,
                    'sector': 'Unknown',
                    'sector_performance': 0.0,
                    'sector_momentum': 0.0,
                    'sector_volatility': 0.0,
                    'recommendation': 'NEUTRAL',
                    'analysis_date': pd.Timestamp.now().isoformat()
                }
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Define sector ETFs for comparison
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Industrials': 'XLI',
                'Energy': 'XLE',
                'Consumer Defensive': 'XLP',
                'Basic Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
                'Utilities': 'XLU'
            }
            
            # Get sector ETF for comparison
            sector_etf = sector_etfs.get(sector, 'SPY')  # Default to SPY if sector not found
            
            # Analyze sector performance
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=252)  # 1 year
            
            try:
                sector_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)
                if not sector_data.empty:
                    # Calculate sector metrics
                    sector_returns = sector_data['Adj Close'].pct_change().dropna()
                    sector_performance = (sector_data['Adj Close'].iloc[-1] / sector_data['Adj Close'].iloc[0] - 1) * 100
                    sector_momentum = sector_returns.rolling(20).mean().iloc[-1] * 252 * 100
                    sector_volatility = sector_returns.std() * np.sqrt(252) * 100
                    
                    # Determine recommendation based on performance
                    if sector_performance > 10:
                        recommendation = 'OVERWEIGHT'
                    elif sector_performance < -5:
                        recommendation = 'UNDERWEIGHT'
                    else:
                        recommendation = 'NEUTRAL'
                else:
                    sector_performance = 0.0
                    sector_momentum = 0.0
                    sector_volatility = 0.0
                    recommendation = 'NEUTRAL'
                    
            except Exception as e:
                print(f"Error analyzing sector ETF {sector_etf}: {e}")
                sector_performance = 0.0
                sector_momentum = 0.0
                sector_volatility = 0.0
                recommendation = 'NEUTRAL'
            
            return {
                'ticker': ticker,
                'sector': sector,
                'industry': industry,
                'sector_etf': sector_etf,
                'sector_performance': sector_performance,
                'sector_momentum': sector_momentum,
                'sector_volatility': sector_volatility,
                'recommendation': recommendation,
                'analysis_date': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in sector trend analysis: {e}")
            return {
                'ticker': ticker,
                'sector': 'Unknown',
                'sector_performance': 0.0,
                'sector_momentum': 0.0,
                'sector_volatility': 0.0,
                'recommendation': 'NEUTRAL',
                'analysis_date': pd.Timestamp.now().isoformat(),
                'error': str(e)
            }