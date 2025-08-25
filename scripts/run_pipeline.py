import os
import json
import argparse
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# fetcher
try:
    import yfinance as yf
except Exception:
    yf = None

# optional model trainer import (best-effort)
try:
    from partB_model.model_builder import train_model
except Exception:
    train_model = None

# ensemble predictor (prefer in-process call)
try:
    from partC_strategy.ensemble_predictor import ensemble_predict
except Exception:
    ensemble_predict = None

def fetch_and_preprocess(ticker: str, period: str = "2y", interval: str = "1d", out_dir: str = "data"):
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")
    os.makedirs(out_dir, exist_ok=True)
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No price data for {ticker}")
    # basic preprocessing: keep Date index, compute Adj Close if available, simple returns and vols
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    df = df.rename_axis("Date")
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["ATR_14"] = (df["High"] - df["Low"]).rolling(14).mean()  # simple ATR proxy
    # save preprocessed CSV used by ensemble
    out_path = os.path.join(out_dir, f"preprocessed_{ticker.replace('/','_')}.csv")
    df.to_csv(out_path)
    return df, out_path

def try_train_model(ticker: str, df: pd.DataFrame, epochs: int = 5):
    if train_model is None:
        print("Model trainer not available in repo, skipping training.")
        return None
    try:
        print("Starting model training (this may take time)...")
        model_path = train_model(ticker=ticker, data=df, epochs=epochs)
        print("Model training finished, model saved at:", model_path)
        return model_path
    except Exception as e:
        print("Model training failed:", e)
        return None

def run_ensemble_inprocess(ticker: str, headlines_json: str = None, sector_members: str = None, window_size: int = 60):
    headlines = None
    if headlines_json and os.path.exists(headlines_json):
        try:
            headlines = json.load(open(headlines_json, "r"))
        except Exception:
            headlines = None
    sector_members_list = [s.strip().upper() for s in sector_members.split(",")] if sector_members else None

    if ensemble_predict is None:
        # fallback to subprocess call
        cmd = [sys.executable, "-m", "partC_strategy.ensemble_predictor", "--stock", ticker]
        if headlines_json:
            cmd += ["--headlines-json", headlines_json]
        if sector_members:
            cmd += ["--sector-members", sector_members]
        print("Calling ensemble via subprocess:", " ".join(cmd))
        import subprocess
        subprocess.run(cmd, check=False)
        return None

    # call directly
    res = ensemble_predict(stock=ticker, headlines=headlines, sector_members=sector_members_list, window_size=window_size)
    return res

def main():
    p = argparse.ArgumentParser(description="Run preprocess -> train (optional) -> ensemble non-interactively for a ticker")
    p.add_argument("--ticker", "-t", required=True, help="Ticker (use exchange suffix e.g. RELIANCE.NS)")
    p.add_argument("--period", default="2y", help="yfinance period (default 2y)")
    p.add_argument("--no-train", action="store_true", help="Skip model training step")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs if trainer available")
    p.add_argument("--headlines-json", default=None, help="Path to JSON array of headlines for company events")
    p.add_argument("--sector-members", default=None, help="Comma-separated sector tickers")
    p.add_argument("--window-size", type=int, default=60, help="window size for model prediction / ensemble")
    args = p.parse_args()

    ticker = args.ticker.upper().strip()
    print(f"[{datetime.now().isoformat()}] Running pipeline for {ticker}")

    try:
        df, preproc_path = fetch_and_preprocess(ticker, period=args.period)
        print("Preprocessing done, saved:", preproc_path)
    except Exception as e:
        print("Preprocessing failed:", e)
        return

    if not args.no_train:
        try:
            try_train_model(ticker, df, epochs=args.epochs)
        except Exception as e:
            print("Training stage error:", e)

    # run ensemble/prediction
    try:
        res = run_ensemble_inprocess(ticker, headlines_json=args.headlines_json, sector_members=args.sector_members, window_size=args.window_size)
        # if result returned as dict, save it
        out_path = os.path.join("data", f"{ticker}_pipeline_result.json")
        if isinstance(res, dict):
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print("Ensemble result saved to", out_path)
            print(json.dumps(res, indent=2))
        else:
            print("Ensemble ran (no in-process result to save).")
    except Exception as e:
        print("Ensemble step failed:", e)

if __name__ == "__main__":
    main()