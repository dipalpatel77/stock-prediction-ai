# Angel One API Integration Setup Guide

## 🚀 Overview

This guide will help you set up the Angel One API integration to download historical data for Indian stocks. The system now supports automatic data downloading for 100+ Indian stocks.

## 📋 Prerequisites

1. **Angel One Account**: You need an active Angel One trading account
2. **API Access**: Enable API access in your Angel One account
3. **API Credentials**: Generate API key and auth token

## 🔧 Step-by-Step Setup

### 1. Install Required Packages

```bash
pip install requests python-dotenv
```

### 2. Set Up API Credentials

Run the configuration helper:

```bash
python angel_one_config.py
```

This will guide you through:
- Setting up your API credentials
- Testing the connection
- Viewing supported stocks

### 3. Manual Setup (Alternative)

If you prefer manual setup, create a `.env` file in your project root:

```env
# Angel One API Credentials
ANGEL_ONE_API_KEY=your_api_key_here
ANGEL_ONE_AUTH_TOKEN=your_auth_token_here
CLIENT_IP=127.0.0.1
MAC_ADDRESS=00:00:00:00:00:00
```

## 📊 Supported Stocks

The system supports 100+ Indian stocks including:

**Major Stocks:**
- RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
- HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK
- AXISBANK, ASIANPAINT, MARUTI, HCLTECH, SUNPHARMA

**Banking:**
- SBIN, HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK
- INDUSINDBK, BANKBARODA, CANBK, UNIONBANK, PNB

**Technology:**
- TCS, INFY, HCLTECH, WIPRO, TECHM

**Automotive:**
- TATAMOTORS, MARUTI, EICHERMOT, M&M, BAJAJ-AUTO

**Pharmaceuticals:**
- SUNPHARMA, CIPLA, DRREDDY, LUPIN, CADILAHC

## 🔍 How to Get API Credentials

### 1. Create Angel One Account
- Visit [Angel One](https://www.angelone.in/)
- Complete KYC and account verification

### 2. Enable API Access
- Log in to your Angel One account
- Go to Settings → API Access
- Enable API trading
- Accept terms and conditions

### 3. Generate API Credentials
- In API Access section, click "Generate API Key"
- Note down your API Key and Auth Token
- Keep these credentials secure

## 📈 Data Intervals Available

The API supports multiple time intervals:

| Interval | Description | Max Days per Request |
|----------|-------------|---------------------|
| ONE_MINUTE | 1 Minute | 30 |
| THREE_MINUTE | 3 Minutes | 60 |
| FIVE_MINUTE | 5 Minutes | 100 |
| TEN_MINUTE | 10 Minutes | 100 |
| FIFTEEN_MINUTE | 15 Minutes | 200 |
| THIRTY_MINUTE | 30 Minutes | 200 |
| ONE_HOUR | 1 Hour | 400 |
| ONE_DAY | 1 Day | 2000 |

## 🚀 Usage Examples

### Basic Data Download

```python
from angel_one_data_downloader import AngelOneDataDownloader

# Initialize downloader
downloader = AngelOneDataDownloader(
    api_key="your_api_key",
    auth_token="your_auth_token"
)

# Download 1 year of daily data for RELIANCE
df = downloader.get_latest_data("RELIANCE", days=365, interval="ONE_DAY")
```

### Download Specific Date Range

```python
# Download data for specific dates
df = downloader.download_historical_data(
    symbol="TCS",
    from_date="2024-01-01",
    to_date="2024-08-25",
    interval="ONE_DAY"
)
```

### Intraday Data

```python
# Download 5-minute intraday data
df = downloader.get_latest_data("HDFCBANK", days=30, interval="FIVE_MINUTE")
```

## 🔄 Integration with Prediction System

The Angel One data downloader is automatically integrated with the prediction system:

1. **Automatic Data Loading**: When you run predictions, the system first tries to load existing data
2. **Fallback to API**: If no data exists, it automatically downloads from Angel One
3. **Data Enhancement**: Downloaded data is automatically enhanced with technical indicators
4. **Caching**: Data is saved locally for future use

### Running Predictions with New Data

```bash
python run_stock_prediction.py
```

The system will:
1. Check for existing data
2. Download fresh data if needed
3. Apply advanced ML algorithms
4. Generate predictions with confidence intervals

## 🛠️ Troubleshooting

### Common Issues

1. **"API credentials not found"**
   - Run `python angel_one_config.py` to set up credentials
   - Ensure `.env` file exists in project root

2. **"Connection failed"**
   - Verify your API key and auth token
   - Check if your Angel One account has API access enabled
   - Ensure your IP is whitelisted

3. **"Token not found for symbol"**
   - The stock symbol might not be in our database
   - Add the token manually to `stock_tokens` dictionary
   - Contact support for new stock additions

4. **Rate Limiting**
   - The system includes automatic rate limiting
   - Wait 1-2 seconds between requests
   - Reduce the number of concurrent requests

### Error Messages

- **HTTP 401**: Invalid API credentials
- **HTTP 403**: API access not enabled
- **HTTP 429**: Rate limit exceeded
- **HTTP 500**: Server error, try again later

## 📁 File Structure

```
project/
├── angel_one_data_downloader.py    # Main downloader class
├── angel_one_config.py             # Configuration helper
├── run_stock_prediction.py         # Main prediction system
├── .env                            # API credentials (create this)
├── data/                           # Downloaded data storage
│   ├── RELIANCE_one_day_data.csv
│   ├── TCS_one_day_data.csv
│   └── ...
└── ANGEL_ONE_SETUP.md             # This guide
```

## 🔒 Security Notes

1. **Never commit `.env` file** to version control
2. **Keep API credentials secure**
3. **Use environment variables** for production
4. **Regularly rotate API keys**
5. **Monitor API usage** to avoid rate limits

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify your Angel One account status
3. Test API connection using `angel_one_config.py`
4. Contact Angel One support for API issues
5. Check the project documentation

## 🎯 Next Steps

After setting up Angel One API:

1. **Test the connection**: Run `python angel_one_config.py`
2. **Download sample data**: Try downloading data for RELIANCE
3. **Run predictions**: Use `python run_stock_prediction.py`
4. **Explore features**: Try different intervals and timeframes
5. **Add more stocks**: Request additional stock symbols if needed

---

**Happy Trading! 📈**
