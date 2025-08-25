import pandas as pd
import yfinance as yf

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load historical price data. Try ticker as given, and if no data found and the ticker
    looks like an Indian ticker (no dot), try common exchange suffixes (.NS, .BO).
    
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL').
        start (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
    
    Returns:
        pd.DataFrame: Stock price data.
    """
    candidates = [ticker]
    # if user passed a plain symbol, try common Indian suffixes
    if "." not in ticker:
        candidates += [ticker + suf for suf in (".NS", ".BO", ".NSE")]
    tried = []
    for tk in candidates:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                return df
            tried.append(f"{tk}: empty")
        except Exception as e:
            tried.append(f"{tk}: {e}")
            continue
    # no data found after trying candidates
    raise ValueError(f"No data found for {ticker} between {start} and {end}. Tried: {tried}")
