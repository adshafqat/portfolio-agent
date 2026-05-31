import os
import json
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"
JSON_FILE = "portfolio.json"

def scrape_live_price(identifier: str) -> float:
    """Scrapes the live market price from the FT tearsheet page."""
    for extension in [":GBX", ":GBP"]:
        url = f"https://markets.ft.com/data/funds/tearsheet/summary?s={identifier}{extension}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # FT summary page target for live price
                price_el = soup.find("span", class_="mod-ui-data-list__value")
                if price_el:
                    return float(price_el.text.strip().replace(",", ""))
        except:
            continue
    return None

def run_live_analysis():
    with open(JSON_FILE, 'r') as f:
        config = json.load(f)
    
    holdings_snapshot = []
    print(f"⏳ Refreshing live market prices for {len(config['holdings'])} holdings...")
    
    for item in config["holdings"]:
        raw_price = scrape_live_price(item["isin"])
        if raw_price:
            price = raw_price / 100.0 if item["is_pence"] else raw_price
            current_val = round(item["shares_owned"] * price, 2)
            holdings_snapshot.append({
                "name": item["name"],
                "price": price,
                "current_value": current_val
            })
            print(f"   📊 {item['name']}: £{price:,.4f}")
        time.sleep(0.5)

    # Send to AI for allocation strategy
    prompt = f"Portfolio snapshot: {json.dumps(holdings_snapshot)}. Recommend how to allocate cash balance of £{config['cash_balance_gbp']} based on current values."
    
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    print("\n=== LIVE PORTFOLIO STRATEGY ===\n", response.text)

if __name__ == "__main__":
    run_live_analysis()
