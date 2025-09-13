#!/usr/bin/env python3
"""
Indian Stock Mapper - Angel API Only
Downloads and processes Indian stock data using only Angel Broking's API.
Provides comprehensive mapping and filtering for Indian stocks.
"""

import io, re, json, os
from datetime import datetime, timedelta
import requests, pandas as pd

# ============ CONFIG ============
SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
UA = {"User-Agent": "Mozilla/5.0"}

# Create data directory structure
DATA_DIR = "angel_data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# Cache file paths
CACHE_FILE = os.path.join(CACHE_DIR, "angel_master_cache.json")
CACHE_METADATA_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")

# What counts as each bucket (Angel sometimes varies spellings; we normalize)
EQUITY_TAGS      = {"", "EQ"}                      # equities (empty string and EQ)
DERIVATIVE_TAGS  = {"FUTSTK","OPTSTK","FUTIDX","OPTIDX","FNO", "OPTFUT", "FUTCOM", "OPTCUR", "FUTCUR", "OPTIRC", "FUTIRC", "OPTTIR", "FUTIRT", "OPTFUT", "FUTENR", "OPTTIR", "FUTBAS"}    # derivatives
INDEX_TAGS       = {"INDEX","INDICES", "INDEX"}                            # indices

# Output file paths (now in organized folders)
OUT_EQ_NSE = os.path.join(EXPORT_DIR, "angel_nse_equities.csv")
OUT_EQ_BSE = os.path.join(EXPORT_DIR, "angel_bse_equities.csv")
OUT_DRV_NSE = os.path.join(EXPORT_DIR, "angel_nse_derivatives.csv")
OUT_DRV_BSE = os.path.join(EXPORT_DIR, "angel_bse_derivatives.csv")
OUT_IDX_NSE = os.path.join(EXPORT_DIR, "angel_nse_indices.csv")
OUT_IDX_BSE = os.path.join(EXPORT_DIR, "angel_bse_indices.csv")
OUT_ALL_EQUITIES = os.path.join(EXPORT_DIR, "angel_all_equities.csv")

# ============ CACHE MANAGEMENT ============
def is_cache_valid():
    """Check if the cached data is still valid (less than 24 hours old)."""
    if not os.path.exists(CACHE_METADATA_FILE):
        return False
    
    try:
        with open(CACHE_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        cache_time = datetime.fromisoformat(metadata['cache_time'])
        current_time = datetime.now()
        
        # Check if cache is from today
        return cache_time.date() == current_time.date()
    except Exception:
        return False

def save_cache(data):
    """Save data to cache with metadata."""
    try:
        # Save the data
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f)
        
        # Save metadata
        metadata = {
            'cache_time': datetime.now().isoformat(),
            'record_count': len(data),
            'source_url': SCRIP_MASTER_URL
        }
        with open(CACHE_METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        
        print(f"💾 Cache saved: {len(data)} records")
    except Exception as e:
        print(f"⚠️ Warning: Could not save cache: {e}")

def load_cache():
    """Load data from cache if valid."""
    if not is_cache_valid():
        return None
    
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        
        print(f"📂 Cache loaded: {len(data)} records")
        return data
    except Exception as e:
        print(f"⚠️ Warning: Could not load cache: {e}")
        return None

# ============ DOWNLOADERS ============
def fetch_json(url, headers=UA, timeout=60):
    """Fetch JSON data from URL."""
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ============ ANGEL MASTER + SPLITS ============
def load_angel_master(force_download=False):
    """
    Load Angel master data with smart caching.
    
    Args:
        force_download: If True, ignore cache and download fresh data
        
    Returns:
        DataFrame with Angel master data
    """
    # Check cache first (unless forced download)
    if not force_download:
        cached_data = load_cache()
        if cached_data is not None:
            df = pd.DataFrame(cached_data)
            return normalize_angel_data(df)
    
    # Download fresh data
    print("🔄 Downloading fresh Angel master data...")
    try:
        data = fetch_json(SCRIP_MASTER_URL)
        
        # Save to cache
        save_cache(data)
        
        # Convert to DataFrame and normalize
        df = pd.DataFrame(data)
        return normalize_angel_data(df)
        
    except Exception as e:
        print(f"❌ Failed to download Angel master data: {e}")
        
        # Try to load from cache as fallback
        print("🔄 Attempting to load from cache as fallback...")
        cached_data = load_cache()
        if cached_data is not None:
            df = pd.DataFrame(cached_data)
            return normalize_angel_data(df)
        else:
            raise Exception("No data available - download failed and no cache found")

def normalize_angel_data(df):
    """Normalize Angel master DataFrame."""
    # Ensure expected columns exist
    for c in ["token","symbol","name","exch_seg","instrumenttype","lotsize","tick_size"]:
        if c not in df.columns: 
            df[c] = None
    
    # Normalize
    df["exchange"] = df["exch_seg"].astype(str).str.upper().str.strip()
    df["instrumenttype"] = df["instrumenttype"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["name"]   = df["name"].astype(str).str.strip()
    
    # Dedupe
    df = df.drop_duplicates(subset=["exchange","symbol","instrumenttype","token"])
    
    print(f"✅ Angel master processed: {len(df)} records")
    return df

def split_and_export(df: pd.DataFrame, force_export=False):
    """
    Split Angel data by exchange and instrument type, then export to CSV.
    
    Args:
        df: DataFrame to split and export
        force_export: If True, overwrite existing files
    """
    def sel(exch, tags):
        return df[(df["exchange"]==exch) & (df["instrumenttype"].isin(tags))]\
                 [["exchange","symbol","name","token","lotsize","tick_size","instrumenttype"]].copy()

    print("🔄 Splitting and exporting data...")
    
    # Check if files already exist and are from today
    if not force_export and os.path.exists(OUT_ALL_EQUITIES):
        file_time = datetime.fromtimestamp(os.path.getmtime(OUT_ALL_EQUITIES))
        if file_time.date() == datetime.now().date():
            print("📂 Export files already exist from today, skipping export...")
            return load_existing_exports()
    
    # Equities
    nse_eq = sel("NSE", EQUITY_TAGS).sort_values("symbol")
    bse_eq = sel("BSE", EQUITY_TAGS).sort_values("symbol")
    nse_eq.to_csv(OUT_EQ_NSE, index=False)
    bse_eq.to_csv(OUT_EQ_BSE, index=False)

    # Derivatives
    nse_drv = sel("NSE", DERIVATIVE_TAGS).sort_values(["instrumenttype","symbol"])
    bse_drv = sel("BSE", DERIVATIVE_TAGS).sort_values(["instrumenttype","symbol"])
    nse_drv.to_csv(OUT_DRV_NSE, index=False)
    bse_drv.to_csv(OUT_DRV_BSE, index=False)

    # Indices
    nse_idx = sel("NSE", INDEX_TAGS).sort_values("symbol")
    bse_idx = sel("BSE", INDEX_TAGS).sort_values("symbol")
    nse_idx.to_csv(OUT_IDX_NSE, index=False)
    bse_idx.to_csv(OUT_IDX_BSE, index=False)

    # All equities combined
    all_equities = pd.concat([nse_eq, bse_eq], ignore_index=True).sort_values("symbol")
    all_equities.to_csv(OUT_ALL_EQUITIES, index=False)

    print(f"✅ Data exported to {EXPORT_DIR}:")
    print(f"   - NSE Equities: {len(nse_eq)} records")
    print(f"   - BSE Equities: {len(bse_eq)} records")
    print(f"   - NSE Derivatives: {len(nse_drv)} records")
    print(f"   - BSE Derivatives: {len(bse_drv)} records")
    print(f"   - NSE Indices: {len(nse_idx)} records")
    print(f"   - BSE Indices: {len(bse_idx)} records")
    print(f"   - All Equities: {len(all_equities)} records")

    return {
        "nse_eq": nse_eq, 
        "bse_eq": bse_eq, 
        "nse_drv": nse_drv, 
        "bse_drv": bse_drv, 
        "nse_idx": nse_idx, 
        "bse_idx": bse_idx,
        "all_equities": all_equities
    }

def load_existing_exports():
    """Load existing exported files if they exist."""
    try:
        nse_eq = pd.read_csv(OUT_EQ_NSE) if os.path.exists(OUT_EQ_NSE) else pd.DataFrame()
        bse_eq = pd.read_csv(OUT_EQ_BSE) if os.path.exists(OUT_EQ_BSE) else pd.DataFrame()
        nse_drv = pd.read_csv(OUT_DRV_NSE) if os.path.exists(OUT_DRV_NSE) else pd.DataFrame()
        bse_drv = pd.read_csv(OUT_DRV_BSE) if os.path.exists(OUT_DRV_BSE) else pd.DataFrame()
        nse_idx = pd.read_csv(OUT_IDX_NSE) if os.path.exists(OUT_IDX_NSE) else pd.DataFrame()
        bse_idx = pd.read_csv(OUT_IDX_BSE) if os.path.exists(OUT_IDX_BSE) else pd.DataFrame()
        all_equities = pd.read_csv(OUT_ALL_EQUITIES) if os.path.exists(OUT_ALL_EQUITIES) else pd.DataFrame()
        
        return {
            "nse_eq": nse_eq, 
            "bse_eq": bse_eq, 
            "nse_drv": nse_drv, 
            "bse_drv": bse_drv, 
            "nse_idx": nse_idx, 
            "bse_idx": bse_idx,
            "all_equities": all_equities
        }
    except Exception as e:
        print(f"⚠️ Warning: Could not load existing exports: {e}")
        return {}

def get_cache_info():
    """Get information about the current cache status."""
    if not os.path.exists(CACHE_METADATA_FILE):
        return {"status": "no_cache", "message": "No cache found"}
    
    try:
        with open(CACHE_METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        cache_time = datetime.fromisoformat(metadata['cache_time'])
        current_time = datetime.now()
        age_hours = (current_time - cache_time).total_seconds() / 3600
        
        return {
            "status": "valid" if is_cache_valid() else "expired",
            "cache_time": cache_time.isoformat(),
            "age_hours": round(age_hours, 2),
            "record_count": metadata['record_count'],
            "is_today": cache_time.date() == current_time.date()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============ UTILITY FUNCTIONS ============
def get_symbol_info(symbol: str, df: pd.DataFrame = None) -> dict | None:
    """
    Get comprehensive symbol information for a given symbol.
    
    Args:
        symbol: Stock symbol to search for
        df: DataFrame to search in (if None, loads Angel master)
        
    Returns:
        Dictionary with symbol information or None if not found
    """
    if df is None:
        df = load_angel_master()
    
    symbol_upper = symbol.upper().strip()
    
    # Search for exact match
    match = df[df['symbol'] == symbol_upper]
    
    if not match.empty:
        row = match.iloc[0]
        return {
            'symbol': row['symbol'],
            'name': row['name'],
            'exchange': row['exchange'],
            'token': row['token'],
            'instrumenttype': row['instrumenttype'],
            'lotsize': row['lotsize'],
            'tick_size': row['tick_size']
        }
    
    return None

def search_symbols(query: str, df: pd.DataFrame = None, limit: int = 10) -> pd.DataFrame:
    """
    Search for symbols by name or symbol.
    
    Args:
        query: Search query
        df: DataFrame to search in (if None, loads Angel master)
        limit: Maximum number of results
        
    Returns:
        DataFrame with matching symbols
    """
    if df is None:
        df = load_angel_master()
    
    query_upper = query.upper()
    
    # Search in names and symbols
    mask = (
        df['name'].str.contains(query_upper, case=False, na=False) |
        df['symbol'].str.contains(query_upper, case=False, na=False)
    )
    
    results = df[mask].head(limit)
    return results

def get_equities_by_exchange(exchange: str = "NSE", df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Get all equities for a specific exchange.
    
    Args:
        exchange: Exchange name ("NSE" or "BSE")
        df: DataFrame to filter (if None, loads Angel master)
        
    Returns:
        DataFrame with equities for the specified exchange
    """
    if df is None:
        df = load_angel_master()
    
    exchange_upper = exchange.upper()
    return df[(df['exchange'] == exchange_upper) & (df['instrumenttype'].isin(EQUITY_TAGS))]

def get_statistics(df: pd.DataFrame = None) -> dict:
    """
    Get statistics about the Angel master data.
    
    Args:
        df: DataFrame to analyze (if None, loads Angel master)
        
    Returns:
        Dictionary with statistics
    """
    if df is None:
        df = load_angel_master()
    
    stats = {
        'total_records': len(df),
        'exchanges': df['exchange'].nunique(),
        'unique_symbols': df['symbol'].nunique(),
        'unique_names': df['name'].nunique(),
        'instrument_types': df['instrumenttype'].nunique(),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cache_info': get_cache_info()
    }
    
    # Exchange breakdown
    for exchange in df['exchange'].unique():
        if pd.notna(exchange):
            exchange_data = df[df['exchange'] == exchange]
            stats[f'{exchange.lower()}_records'] = len(exchange_data)
            stats[f'{exchange.lower()}_symbols'] = exchange_data['symbol'].nunique()
    
    # Instrument type breakdown
    for inst_type in df['instrumenttype'].unique():
        if pd.notna(inst_type):
            inst_data = df[df['instrumenttype'] == inst_type]
            stats[f'{inst_type.lower()}_records'] = len(inst_data)
    
    return stats

def clear_cache():
    """Clear the cache and force fresh download on next run."""
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        if os.path.exists(CACHE_METADATA_FILE):
            os.remove(CACHE_METADATA_FILE)
        print("🗑️ Cache cleared successfully")
    except Exception as e:
        print(f"❌ Failed to clear cache: {e}")

# ============ MAIN ============
if __name__ == "__main__":
    print("🚀 Indian Stock Mapper - Angel API Only (Smart Caching)")
    print("=" * 60)
    
    # Show cache status
    cache_info = get_cache_info()
    print(f"📂 Cache Status: {cache_info['status']}")
    if cache_info['status'] != 'no_cache':
        print(f"   Last Updated: {cache_info['cache_time']}")
        print(f"   Age: {cache_info['age_hours']} hours")
        print(f"   Records: {cache_info['record_count']}")
    
    # Load Angel master (with smart caching)
    angel = load_angel_master()
    
    # Split and export (with smart caching)
    splits = split_and_export(angel)
    
    # Show statistics
    print("\n📊 Statistics:")
    stats = get_statistics(angel)
    for key, value in stats.items():
        if key != 'cache_info':
            print(f"   {key}: {value}")
    
    # Test symbol lookup
    print("\n🔍 Testing Symbol Lookup:")
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN']
    for symbol in test_symbols:
        info = get_symbol_info(symbol, angel)
        if info:
            print(f"   ✅ {symbol}: {info['name']} ({info['exchange']})")
        else:
            print(f"   ❌ {symbol}: Not found")
    
    print("\n✅ Done!")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📂 Cache Directory: {CACHE_DIR}")
    print(f"📤 Export Directory: {EXPORT_DIR}")
    print(f"- Equities: {OUT_EQ_NSE}, {OUT_EQ_BSE}")
    print(f"- Derivatives: {OUT_DRV_NSE}, {OUT_DRV_BSE}")
    print(f"- Indices: {OUT_IDX_NSE}, {OUT_IDX_BSE}")
    print(f"- All Equities: {OUT_ALL_EQUITIES}")
