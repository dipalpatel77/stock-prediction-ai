import os
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# optional yfinance
try:
    import yfinance as yf
except Exception:
    yf = None

class CompanyFinancialAnalyzer:
    """
    Load balance-sheet / income-statement / cash-flow data and compute simple ratio scores.
    - load_from_csv(path): expects standard financial statement CSV (columns/indices flexible)
    - fetch_via_yfinance(ticker): best-effort fetch (yfinance)
    - analyze(fs: dict) -> summary dict with ratios and normalized bias (-1..1)
    """

    def __init__(self):
        # weightings for aggregate financial score
        self.weights = {
            "liquidity": 0.25,
            "leverage": 0.25,
            "profitability": 0.30,
            "growth": 0.20
        }

    def load_from_csv(self, path: str) -> Dict[str, pd.DataFrame]:
        """
        Try to load CSVs named balance_sheet_*.csv / income_statement_*.csv / cashflow_*.csv or a single file.
        Returns dict with keys: balance_sheet,income_statement,cashflow (DataFrames or empty)
        """
        out = {"balance_sheet": None, "income_statement": None, "cashflow": None}
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        # single combined file attempt
        df = pd.read_csv(path, index_col=0)
        out["balance_sheet"] = df
        return out

    def fetch_via_yfinance(self, ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Best-effort fetch from yfinance. Returns dict with DataFrames (columns = periods).
        """
        if yf is None:
            return {"balance_sheet": None, "income_statement": None, "cashflow": None}
        tk = yf.Ticker(ticker)
        try:
            bs = tk.balance_sheet.transpose() if hasattr(tk, "balance_sheet") and tk.balance_sheet is not None else None
        except Exception:
            bs = None
        try:
            inc = tk.financials.transpose() if hasattr(tk, "financials") and tk.financials is not None else None
        except Exception:
            inc = None
        try:
            cf = tk.cashflow.transpose() if hasattr(tk, "cashflow") and tk.cashflow is not None else None
        except Exception:
            cf = None
        return {"balance_sheet": bs, "income_statement": inc, "cashflow": cf}

    def _safe_get(self, df: pd.DataFrame, candidates):
        if df is None:
            return np.nan
        for c in candidates:
            if c in df.columns:
                # take latest row
                try:
                    return float(df[c].dropna().iat[-1])
                except Exception:
                    return float(df[c].iat[-1])
        # try index lookup
        for c in candidates:
            if c in df.index:
                try:
                    return float(df.loc[c].dropna().iat[-1])
                except Exception:
                    try:
                        return float(df.loc[c])
                    except Exception:
                        continue
        return np.nan

    def analyze(self, data: Dict[str, Optional[pd.DataFrame]]) -> Dict[str, Any]:
        """
        Compute core ratios and return normalized financial_bias in [-1,1].
        Ratios:
          - current_ratio = totalCurrentAssets / totalCurrentLiabilities
          - debt_to_equity = totalLiab / totalStockholderEquity
          - roa = netIncome / totalAssets
          - revenue_growth = latest_rev / prev_rev - 1
        Score: higher is better (positive bias)
        """
        bs = data.get("balance_sheet")
        inc = data.get("income_statement")
        cf = data.get("cashflow")

        # candidates for common fields (yfinance uses e.g., 'Total Assets', 'Total Liab', 'Total Current Assets')
        total_current_assets = self._safe_get(bs, ["Total Current Assets", "totalCurrentAssets", "TotalCurrentAssets", "Current Assets"])
        total_current_liab = self._safe_get(bs, ["Total Current Liabilities", "totalCurrentLiabilities", "TotalCurrentLiabilities", "Current Liabilities"])
        total_assets = self._safe_get(bs, ["Total Assets", "totalAssets"])
        total_liab = self._safe_get(bs, ["Total Liab", "TotalLiab", "totalLiab"])
        shareholder_equity = self._safe_get(bs, ["Total Stockholder Equity", "Total Shareholder Equity", "Stockholders Equity", "Total Equity", "totalStockholderEquity"])

        net_income = self._safe_get(inc, ["Net Income", "NetIncome", "Net Income Applicable To Common Shares", "Net Income Loss"])
        revenue = self._safe_get(inc, ["Total Revenue", "Revenue", "Sales, net", "Net Sales"])
        # attempt prior period revenue for growth: try second last column if DataFrame with columns by period
        revenue_prev = np.nan
        try:
            if isinstance(inc, pd.DataFrame) and inc.shape[0] >= 2:
                # assume rows are periods; try last two rows in 'Total Revenue' or first column logic
                for col in ["Total Revenue", "Revenue", "Sales, net", "Net Sales"]:
                    if col in inc.columns:
                        vals = inc[col].dropna().values
                        if vals.size >= 2:
                            revenue = float(vals[-1])
                            revenue_prev = float(vals[-2])
                            break
        except Exception:
            revenue_prev = np.nan

        # compute ratios
        current_ratio = float(total_current_assets / total_current_liab) if total_current_liab and not np.isnan(total_current_assets) and not np.isnan(total_current_liab) and total_current_liab != 0 else np.nan
        debt_to_equity = float(total_liab / shareholder_equity) if shareholder_equity and not np.isnan(total_liab) and not np.isnan(shareholder_equity) and shareholder_equity != 0 else np.nan
        roa = float(net_income / total_assets) if total_assets and not np.isnan(net_income) and not np.isnan(total_assets) and total_assets != 0 else np.nan
        revenue_growth = float((revenue / revenue_prev - 1.0)) if revenue_prev and revenue_prev != 0 and not np.isnan(revenue) else np.nan

        # score each component into range [-1,1]
        def _score_current(cr):
            if np.isnan(cr): return 0.0
            if cr >= 2.0: return 1.0
            if cr >= 1.0: return (cr - 1.0) / 1.0
            return (cr - 1.0) / 1.0  # negative

        def _score_leverage(dte):
            if np.isnan(dte): return 0.0
            if dte <= 0.5: return 1.0
            if dte <= 1.5: return 1.0 - (dte - 0.5) / 1.0
            # heavy leverage -> negative
            return max(-1.0, 1.0 - (dte - 0.5) / 2.0)

        def _score_roa(r):
            if np.isnan(r): return 0.0
            # typical ROA: 0.05 = 5% good
            if r >= 0.05: return 1.0
            return (r / 0.05)  # can be negative

        def _score_growth(g):
            if np.isnan(g): return 0.0
            # > 20% exceptional
            if g >= 0.2: return 1.0
            return (g / 0.2)

        s_liq = _score_current(current_ratio)
        s_lev = _score_leverage(debt_to_equity)
        s_roa = _score_roa(roa)
        s_gro = _score_growth(revenue_growth)

        # combine weighted
        weights = self.weights
        # missing -> treat as zero-contribution
        comp_sum = (weights["liquidity"] * s_liq +
                    weights["leverage"] * s_lev +
                    weights["profitability"] * s_roa +
                    weights["growth"] * s_gro)
        # normalize to -1..1 (weights sum to 1)
        financial_score = float(comp_sum)

        # clamp
        financial_score = max(-1.0, min(1.0, financial_score))

        rec = "NEUTRAL"
        if financial_score >= 0.4:
            rec = "STRONG"
        elif financial_score <= -0.4:
            rec = "WEAK"

        return {
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "roa": roa,
            "revenue_growth": revenue_growth,
            "liquidity_score": s_liq,
            "leverage_score": s_lev,
            "profitability_score": s_roa,
            "growth_score": s_gro,
            "financial_score": financial_score,
            "recommendation": rec
        }

    def get_financial_bias(self, ticker: str = None, csv_path: str = None) -> Dict[str, Any]:
        """
        Convenience: load data for ticker (via yfinance) or CSV and return analyze() dict.
        """
        data = {"balance_sheet": None, "income_statement": None, "cashflow": None}
        if csv_path:
            try:
                data = self.load_from_csv(csv_path)
            except Exception:
                pass
        elif ticker:
            try:
                data = self.fetch_via_yfinance(ticker)
            except Exception:
                pass
        return self.analyze(data)