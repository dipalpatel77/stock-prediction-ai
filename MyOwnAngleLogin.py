import requests
import pyotp
import json
import pandas as pd

# ====== CONFIG ======
API_KEY       = "3PMAARNa "
CLIENT_ID     = "D54448"
PASSWORD      = "2251"
TOTP_SECRET   = "NP4SAXOKMTJQZ4KZP2TBTYXRCE"   # Base32 secret from AngelOne
API_BASE_URL  = "https://apiconnect.angelbroking.com/rest/secure/angelbroking"

# Symbol you want historical data for:
SYMBOL_NAME   = "INFY"      # Example: Infosys
EXCHANGE      = "NSE"       # NSE or BSE

# ====== STEP 1: Get Symbol Token ======
print("📥 Downloading instruments file...")
instruments_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
instruments = requests.get(instruments_url).json()

df = pd.DataFrame(instruments)
df = df[df["exch_seg"] == EXCHANGE]   # Filter exchange
df = df[df["symbol"].str.upper().str.contains(SYMBOL_NAME.upper())]


if df.empty:
    raise Exception(f"❌ Symbol {SYMBOL_NAME} not found in {EXCHANGE} instruments list!")

symbol_token = df.iloc[0]["token"]
print(f"✅ Found Symbol: {SYMBOL_NAME} | Token: {symbol_token}")

# ====== STEP 2: Generate TOTP ======
totp = pyotp.TOTP(TOTP_SECRET).now()

# ====== STEP 3: Login & Get Access Token ======
login_payload = {
    "clientcode": CLIENT_ID,
    "password": PASSWORD,
    "totp": totp
}

headers = {
    "Content-Type": "application/json",
    "X-UserType": "USER",
    "X-SourceID": "WEB",
    "X-ClientLocalIP": "127.0.0.1",
    "X-ClientPublicIP": "127.0.0.1",
    "X-MACAddress": "00:00:00:00:00:00",
    "Accept": "application/json",
    "X-APIKey": API_KEY
}

print("🔐 Logging in...")
res = requests.post(
    "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword",
    json={
        "clientcode": CLIENT_ID,
        "password": PASSWORD,
        "totp": totp
    },
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-UserType": "USER",
        "X-SourceID": "WEB",
        "X-ClientLocalIP": "127.0.0.1",
        "X-ClientPublicIP": "127.0.0.1",
        "X-MACAddress": "XX:XX:XX:XX:XX:XX",
        "X-PrivateKey": API_KEY
    }
)

print("🔍 Status Code:", res.status_code)
print("🔍 Response Text:", res.text)   # <-- see raw output

try:
    res_json = res.json()
except Exception as e:
    raise Exception(f"❌ Failed to parse JSON. Raw response:\n{res.text}") from e

res_json = res.json()
if res_json.get("status") != True:
    raise Exception("❌ Login failed! Check credentials/TOTP.")

jwt_token = res_json["data"]["jwtToken"]
print("✅ Login success!")

# ====== STEP 4: Call Historical API (Candle Data) ======
candle_payload = {
    "exchange": EXCHANGE,
    "symboltoken": str(symbol_token),
    "interval": "FIVE_MINUTE",   # ONE_MINUTE, FIVE_MINUTE, TEN_MINUTE, DAY, etc.
    "fromdate": "2025-08-20 09:15",  # format: YYYY-MM-DD HH:MM
    "todate": "2025-08-20 15:30"
}

candle_headers = headers.copy()
candle_headers["Authorization"] = f"Bearer {jwt_token}"

print("📊 Fetching Historical Data...")
candle_res = requests.post(
    f"{API_BASE_URL}/historical/v1/getCandleData",
    data=json.dumps(candle_payload),
    headers=candle_headers
)

candle_json = candle_res.json()
candles = candle_json.get("data", [])

if not candles:
    raise Exception("❌ No data received. Check symbol/date range.")

print("📈 Sample Data:", candles[:2])  # Print first 2 candles

# ====== STEP 5: Save Data ======
# Save as JSON
with open(f"{SYMBOL_NAME}_historical.json", "w") as f:
    json.dump(candle_json, f, indent=4)

# Save as CSV (OHLCV format)
df_candles = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
df_candles.to_csv(f"{SYMBOL_NAME}_historical.csv", index=False)

print(f"✅ Historical data saved to {SYMBOL_NAME}_historical.json and {SYMBOL_NAME}_historical.csv")
