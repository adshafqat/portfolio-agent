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

def scrape_historical_price_with_session(session, identifier: str, target_date: str) -> float:
    """
    First executes a GET request to harvest dynamic form tracking state, 
    then POSTs the date filter parameters to extract the true historical price.
    """
    formatted_date = target_date.replace("-", "/")
    
    for extension in [":GBX", ":GBP"]:
        url = f"https://markets.ft.com/data/funds/tearsheet/historical?s={identifier}{extension}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': url
        }
        try:
            # Step 1: Establish context and pull down the hidden CSRF/Form state tokens
            get_response = session.get(url, headers=headers, timeout=10)
            if get_response.status_code != 200:
                continue
                
            soup_get = BeautifulSoup(get_response.text, 'html.parser')
            
            # Locate the historical form tracking tokens
            form = soup_get.find("form", id="historicalForm")
            if not form:
                # Fallback to general input search if form ID names slightly deviate
                form = soup_get.find("form")
                
            # Build payload starting with all default hidden fields found on the page
            payload = {}
            if form:
                for hidden_input in form.find_all("input", type="hidden"):
                    if hidden_input.get("name"):
                        payload[hidden_input["name"]] = hidden_input.get("value", "")
            
            # Step 2: Inject your target dates into the payload fields
            payload.update({
                'historicalForm-dateFrom': formatted_date,
                'historicalForm-dateTo': formatted_date,
                'historicalForm-submit': 'Update'
            })
            
            # Step 3: Post the validated payload back to the server
            post_response = session.post(url, headers=headers, data=payload, timeout=10)
            if post_response.status_code != 200:
                continue
                
            soup_post = BeautifulSoup(post_response.text, 'html.parser')
            data_table = soup_post.find("table", class_="mod-ui-table")
            if data_table:
                rows = data_table.find_all("tr")
                if len(rows) > 1:
                    cells = rows[1].find_all("td")
                    if cells:
                        close_price_text = cells[4].text.strip().replace(",", "")
                        return round(float(close_price_text), 4)
        except Exception:
            pass
    return None

def run_historical_analysis():
    config = load_portfolio_config()
    cash_balance = float(config.get("cash_balance_gbp", 0))
    dates = config.get("comparison_dates", {})
    
    date_a = dates.get("date_a")
    date_b = dates.get("date_b")
    
    if not date_a or not date_b:
        print("❌ Runtime Aborted: Missing 'date_a' or 'date_b' configuration values inside JSON.")
        return

    leaderboard = []
    total_pl_portfolio = 0.0
    
    # Initialize persistent HTTP session to manage cookies and tokens automatically
    session = requests.Session()
    
    print(f"⏳ Querying authenticated historical data window: {date_a} ➡️ {date_b}...")
    print(f"📋 Total items to process: {len(config['holdings'])} holdings.\n")
    
    for idx, item in enumerate(config["holdings"], start=1):
        isin = item["isin"]
        name = item["name"]
        shares = float(item["shares_owned"])
        is_pence = item.get("is_pence", False)
        
        # Pull historical values using the validated session workflow
        raw_price_a = scrape_historical_price_with_session(session, isin, date_a)
        if not raw_price_a and "ticker" in item:
            raw_price_a = scrape_historical_price_with_session(session, item["ticker"].split('.')[0], date_a)
            
        raw_price_b = scrape_historical_price_with_session(session, isin, date_b)
        if not raw_price_b and "ticker" in item:
            raw_price_b = scrape_historical_price_with_session(session, item["ticker"].split('.')[0], date_b)
            
        if raw_price_a is None or raw_price_b is None:
            print(f"   ⚠️ [{idx}/{len(config['holdings'])}] Skipping {name}: Historical data row not found.")
            continue
            
        # Standardize units
        price_a = raw_price_a / 100.0 if is_pence else raw_price_a
        price_b = raw_price_b / 100.0 if is_pence else raw_price_b
        
        # Position calculations
        val_a = shares * price_a
        val_b = shares * price_b
        abs_pl = val_b - val_a
        
        # Prevent division by zero if asset had a technical pricing anomaly
        pct_change = ((price_b - price_a) / price_a) * 100 if price_a > 0 else 0.0
        
        total_pl_portfolio += abs_pl
        
        leaderboard.append({
            "name": name,
            "isin": isin,
            "price_date_a": round(price_a, 4),
            "price_date_b": round(price_b, 4),
            "value_date_a": round(val_a, 2),
            "value_date_b": round(val_b, 2),
            "profit_loss_gbp": round(abs_pl, 2),
            "growth_percentage": round(pct_change, 2)
        })
        print(f"   📊 [{idx}/{len(config['holdings'])}] {name}: {round(pct_change, 2)}% growth")
        
        # Moderate pause between items to prevent anti-bot rate-limiting
        time.sleep(0.6)

    if not leaderboard:
        print("❌ Error: No fund metrics could be extracted. Verify that your target dates do not fall on bank holidays or weekends.")
        return

    leaderboard.sort(key=lambda x: x["growth_percentage"], reverse=True)
    
    system_instruction = (
        "You are a portfolio performance and cash deployment optimization agent. "
        "You are given pre-sorted historical calculations and financial data. You must preserve all numbers exactly "
        "as provided by the runtime. Do not recalculate totals or apply alternative rounding formulas."
    )
    
    user_prompt = (
        f"Generate a Performance Momentum & Cash Deployment Report based on the window from {date_a} to {date_b}.\n\n"
        f"--- METRICS MATRIX ---\n"
        f"{json.dumps(leaderboard, indent=2)}\n\n"
        f"--- CONTEXT LOGS ---\n"
        f"Total Portfolio Profit/Loss over this window: £{total_pl_portfolio:,.2f}\n"
        f"Cash Available for Investment: £{cash_balance:,.2f}\n\n"
        f"REQUIRED MARKDOWN OUTPUT:\n"
        f"1. A clean comparison table: (Fund Name, Value on {date_a}, Value on {date_b}, Net Profit/Loss (£), Growth Rate (%))\n"
        f"2. Executive Insight: Highlight the total profit/loss statement across this specific period, and call out the top 3 highest performing funds.\n"
        f"3. Momentum Deployment Strategy: Allocate the cash balance of £{cash_balance:,.2f} strictly into those top 3 funds. Calculate whole shares to buy based on the price_date_b value, showing the cash leftover down to the penny."
    )
    
    print("\n🚀 Passing verified tracking data into Gemini inference engine...")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=0.1)
    )
    
    print("\n" + "="*60)
    print("      HISTORICAL PERFORMANCE & ALLOCATION ENGINE REPORT")
    print("="*60 + "\n")
    print(response.text)

if __name__ == "__main__":
    run_historical_analysis()
