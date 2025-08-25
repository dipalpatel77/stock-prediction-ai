import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by filling missing values.
    """
    df = df.ffill()  # forward fill
    df = df.bfill()  # backward fill
    print(f"✅ Data cleaned. Final shape: {df.shape}")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical indicators to stock data.
    """
    # Moving Averages
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df = df.dropna()
    print(f"📊 Added technical indicators. Final shape: {df.shape}")
    return df
