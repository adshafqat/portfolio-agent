import os
import ssl
import json
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"
JSON_FILE = "portfolio.json"

def load_portfolio_config():
    with open(JSON_FILE, 'r') as f:
        return json.load(f)

def get_ft_historical_price(session, isin, target_date):
    """
    Performs a token-handshake to bypass FT security and extract historical data.
    """
    formatted_date = target_date.replace("-", "/")
    url = f"https://markets.ft.com/data/funds/tearsheet/historical?s={isin}:GBP"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    # 1. GET initial page to harvest CSRF tokens and cookies
    try:
        resp = session.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # 2. Collect all hidden inputs (this includes security tokens)
        payload = {hidden['name']: hidden.get('value', '') 
                   for hidden in soup.find_all("input", type="hidden") if hidden.get('name')}
        
        # 3. Inject our specific dates
        payload.update({
            'historicalForm-dateFrom': formatted_date,
            'historicalForm-dateTo': formatted_date,
            'historicalForm-submit': 'Update'
        })

        # 4. POST the request
        post_resp = session.post(url, headers=headers, data=payload, timeout=10)
        soup_post = BeautifulSoup(post_resp.text, 'html.parser')
        
        table = soup_post.find("table", class_="mod-ui-table")
        if table:
            rows = table.find_all("tr")
            if len(rows) > 1:
                cells = rows[1].find_all("td")
                return float(cells[4].text.replace(",", ""))
    except Exception as e:
        print(f"   ⚠️ Parsing error for {isin}: {e}")
    return None

def run_historical_analysis():
    config = load_portfolio_config()
    dates = config.get("comparison_dates", {})
    session = requests.Session()
    
    leaderboard = []
    print(f"⏳ Querying FT historical engine: {dates['date_a']} ➡️ {dates['date_b']}...")
    
    for item in config["holdings"]:
        pa_raw = get_ft_historical_price(session, item["isin"], dates['date_a'])
        pb_raw = get_ft_historical_price(session, item["isin"], dates['date_b'])
        
        if pa_raw and pb_raw:
            pa = pa_raw / 100.0 if item["is_pence"] else pa_raw
            pb = pb_raw / 100.0 if item["is_pence"] else pb_raw
            
            growth = ((pb - pa) / pa) * 100
            val_b = item["shares_owned"] * pb
            
            leaderboard.append({
                "name": item["name"],
                "growth_percentage": round(growth, 2),
                "value_date_b": round(val_b, 2)
            })
            print(f"   📊 {item['name']}: {round(growth, 2)}% growth")
        
        time.sleep(1.2) # Strictly enforced pause to prevent FT blocking

    # Sort by performance
    leaderboard.sort(key=lambda x: x["growth_percentage"], reverse=True)
    
    # Send to AI
    user_prompt = f"Analyze this portfolio momentum data: {json.dumps(leaderboard)}. Allocate £{config['cash_balance_gbp']} into the top 3."
    
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_prompt
    )
    print("\n=== INVESTMENT STRATEGY ===\n")
    print(response.text)

if __name__ == "__main__":
    run_historical_analysis()
