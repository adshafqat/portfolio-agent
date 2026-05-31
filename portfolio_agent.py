import json
import time
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
JSON_FILE = "portfolio.json"

def get_price_on_date(ticker_symbol, target_date):
    """
    Fetches price using yfinance with window padding to ensure 
    we hit a valid trading day.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        # Define a 7-day window to catch the nearest trade
        start_date = (date_obj - timedelta(days=2)).strftime("%Y-%m-%d")
        end_date = (date_obj + timedelta(days=5)).strftime("%Y-%m-%d")
        
        hist = ticker.history(start=start_date, end=end_date)
        if not hist.empty:
            return float(hist['Close'].iloc[0])
    except Exception as e:
        print(f"   ⚠️ Error fetching {ticker_symbol}: {e}")
    return None

def run_historical_analysis():
    with open(JSON_FILE, 'r') as f:
        config = json.load(f)
        
    dates = config.get("comparison_dates", {})
    leaderboard = []
    
    print(f"⏳ Fetching verified market data: {dates['date_a']} ➡️ {dates['date_b']}...")
    
    for item in config["holdings"]:
        ticker = item.get("ticker")
        if not ticker: continue
        
        p1 = get_price_on_date(ticker, dates['date_a'])
        p2 = get_price_on_date(ticker, dates['date_b'])
        
        if p1 and p2:
            # Normalization
            pa = p1 / 100.0 if item["is_pence"] else p1
            pb = p2 / 100.0 if item["is_pence"] else p2
            
            growth = ((pb - pa) / pa) * 100
            
            leaderboard.append({
                "name": item["name"],
                "growth": round(growth, 2),
                "val_b": round(item["shares_owned"] * pb, 2)
            })
            print(f"   📊 {item['name']}: {round(growth, 2)}% growth")
        
        time.sleep(0.5)

    leaderboard.sort(key=lambda x: x["growth"], reverse=True)
    
    # Send to AI
    prompt = f"Portfolio growth data: {json.dumps(leaderboard)}. Recommend allocation for £{config['cash_balance_gbp']} into top 3."
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    print("\n=== STRATEGY ===\n", response.text)

if __name__ == "__main__":
    run_historical_analysis()
