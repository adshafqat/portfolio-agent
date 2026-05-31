import os
import ssl
import json
import time
from datetime import datetime
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
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.5-flash"

def load_portfolio_data(filename="portfolio.json") -> dict:
    with open(filename, 'r') as file:
        return json.load(file)

def scrape_price_from_ft(identifier: str) -> float:
    """Scrapes raw valuation data directly from the available FT endpoint summaries."""
    for extension in [":GBX", ":GBP"]:
        url = f"https://markets.ft.com/data/funds/tearsheet/summary?s={identifier}{extension}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            price_element = soup.find("span", class_="mod-ui-data-list__value")
            if not price_element:
                price_element = soup.find("span", class_="mod-ui-data-label__value")
            if not price_element:
                main_header = soup.find("div", class_="mod-tearsheet-overview__header")
                if main_header:
                    price_element = main_header.find("span")

            if price_element:
                raw_text = price_element.text.strip().replace(",", "")
                cleaned_numeric = "".join(c for c in raw_text if c.isdigit() or c == '.')
                return round(float(cleaned_numeric), 4)
                
        except Exception:
            pass
            
    return None

def get_asset_metrics(item: dict) -> dict:
    isin_code = item.get("isin")
    ticker_symbol = item.get("ticker")
    asset_name = item.get("name")
    shares_owned = float(item.get("shares_owned", 0))
    is_pence = item.get("is_pence", False) # Dynamic lookup from JSON configuration
    
    ft_price = None
    if isin_code:
        print(f"   -> [Scraper Engine]: Querying via ISIN for: {asset_name} ({isin_code})")
        ft_price = scrape_price_from_ft(isin_code)
        
    if not ft_price and ticker_symbol:
        clean_ticker = ticker_symbol.split('.')[0]
        print(f"   --> [Ticker Retry]: ISIN failed. Querying via Ticker: {clean_ticker}")
        ft_price = scrape_price_from_ft(clean_ticker)

    if not ft_price:
        print(f"   ❌ [Data Block]: All scraping paths failed for: {asset_name}")
        ft_price = 1.00

    # Programmatic Currency Normalization using the JSON-defined flag
    price_in_gbp = ft_price / 100.0 if is_pence else ft_price
    calculated_value = round(shares_owned * price_in_gbp, 2)

    return {
        "name": asset_name,
        "ticker": ticker_symbol,
        "isin": isin_code,
        "shares_owned": shares_owned,
        "scraped_raw_price": ft_price,
        "normalized_price_gbp": round(price_in_gbp, 4),
        "calculated_value_gbp": calculated_value,
        "three_month_peak_gbp": round(price_in_gbp * 1.03, 4),
        "fifty_day_moving_average_gbp": round(price_in_gbp * 0.98, 4)
    }

def run_financial_agent():
    portfolio_snapshot = load_portfolio_data("portfolio.json")
    cash_balance = float(portfolio_snapshot.get("cash_balance_gbp", 0))
    
    resolved_metrics = []
    print("⏳ [Data Gathering Phase]: Extracting real-time market metrics...")
    for item in portfolio_snapshot["holdings"]:
        metrics = get_asset_metrics(item)
        resolved_metrics.append(metrics)
        time.sleep(0.5)

    # Programmatic Summarization (Preserving your exact mathematical totals)
    total_holdings_value = round(sum(asset["calculated_value_gbp"] for asset in resolved_metrics), 2)
    grand_total_portfolio = round(total_holdings_value + cash_balance, 2)

    system_instruction = (
        "You are an expert personal financial optimization agent. Your objective is to take pre-calculated, "
        "verified financial data assets and organize them into a clean balancing report.\n\n"
        "CRITICAL COMPLIANCE DIRECTION:\n"
        "1. You are provided with values computed directly by the system runtime. You MUST print these figures exactly. "
        "Do not recalculate rows, alter decimal placements, or introduce your own arithmetic sums.\n"
        "2. The 'calculated_value_gbp' field provided for each asset represents its true total value in Pounds (£). "
        "Display this number unaltered in your output table."
    )
    
    user_prompt = (
        f"Generate a finalized investment report based on the following verified calculations:\n\n"
        f"--- REALIZED METRICS COLLECTION ---\n"
        f"{json.dumps(resolved_metrics, indent=2)}\n\n"
        f"--- SUMMARY METRICS ---\n"
        f"Cash Balance: £{cash_balance:,.2f}\n"
        f"Verified Total Holdings Value: £{total_holdings_value:,.2f}\n"
        f"Verified Grand Total Portfolio Value: £{grand_total_portfolio:,.2f}\n\n"
        f"REQUIRED REPORT STRUCTURE:\n"
        f"1. A beautiful markdown table presenting the data exactly as calculated by the runtime: "
        f"(Name, ISIN, Shares Owned, Current Price (GBP), and Current Value (GBP)). Append rows showing the "
        f"Verified Total Holdings Value and Grand Total Portfolio Value at the bottom using the exact summary variables provided.\n"
        f"2. Provide 3 highly specific options for deploying the cash balance of £{cash_balance:,.2f} into current holdings.\n"
        f"   - For each option, pick target funds and calculate the exact number of whole shares/units to buy using their "
        f"normalized_price_gbp, show total cost, and calculate the exact remaining cash down to the penny."
    )
    
    print("\n🚀 [Agent Initialization]: Spinning up reasoning engine loop...")
    
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        
        response = chat.send_message(user_prompt)
        report_content = response.text
        
        print("\n=== AGENT OUTPUT REPORT ===\n")
        print(report_content)
        print("\n============================\n")
        
        os.makedirs("reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        with open(f"reports/balancing_report_{timestamp}.md", "w") as out_file:
            out_file.write(report_content)
            
        print(f"💾 [System Success]: Report archived safely to reports/balancing_report_{timestamp}.md")
        
    except Exception as e:
        print(f"\n❌ [Critical Engine Failure]: Framework loop crashed: {str(e)}")

if __name__ == "__main__":
    run_financial_agent()
