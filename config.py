# API Keys Configuration
# Add your API keys here for enhanced sentiment analysis

# News API - Get from https://newsapi.org/
NEWSAPI_KEY = "your_newsapi_key_here"

# Alpha Vantage - Get from https://www.alphavantage.co/
ALPHAVANTAGE_KEY = "your_alphavantage_key_here"

# Twitter API - Get from https://developer.twitter.com/
TWITTER_BEARER_TOKEN = "your_twitter_bearer_token_here"

# StockTwits API - Get from https://api.stocktwits.com/
STOCKTWITS_TOKEN = "your_stocktwits_token_here"

# Function to get all API keys
def get_api_keys():
    """
    Get all API keys for sentiment analysis.
    
    Returns:
        dict: Dictionary containing all API keys
    """
    return {
        'newsapi': NEWSAPI_KEY,
        'alphavantage': ALPHAVANTAGE_KEY,
        'twitter_bearer_token': TWITTER_BEARER_TOKEN,
        'stocktwits_token': STOCKTWITS_TOKEN
    }

# Function to check if API keys are configured
def check_api_keys():
    """
    Check if API keys are properly configured.
    
    Returns:
        dict: Status of each API key
    """
    keys = get_api_keys()
    status = {}
    
    for key_name, key_value in keys.items():
        if key_value and key_value != f"your_{key_name}_key_here":
            status[key_name] = "✅ Configured"
        else:
            status[key_name] = "❌ Not configured"
    
    return status

if __name__ == "__main__":
    print("🔑 API Keys Status:")
    status = check_api_keys()
    for key, status_text in status.items():
        print(f"   {key}: {status_text}")
