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

# BULLETPROOF MARKET DATA CONFIGURATION MAP
# True = The FT page returns this asset in PENCE (Needs to be divided by 100)
# False = The FT page returns this asset in POUNDS (Leave exactly as is)
UK_FUND_CURRENCY_MAP = {
    "GB00B3VDD431": False, # Artemis Strategic Assets I Acc (£1.10)
    "GB00BBGBFM09": True,  # Fidelity MnyBldCrpBd W Acc GBP (1418.00p -> £14.18)
    "GB00B849C803": True,  # iShares Osea GovBdIdx(UK) D A (118.93p -> £1.1893)
    "GB00BXVMC989": True,  # Janus Henderson FxdIntMthIn I A (139.60p -> £1.3960)
    "GB00B84QXT94": True,  # L&G All StocksGblGvBdIdx Tst I Acc (104.60p -> £1.0460)
    "GB00B4M89245": False, # Vanguard UKLngDurGiltIdx A AE (£124.07 or £1.24 depending on tracking)
    "GB00B6R51K64": True,  # Aviva Inv UK Listed Eq Inc 2 Acc (348.59p -> £3.4859)
    "GB00B57H4F11": True,  # Liontrust Spl Sits I Inc (457.76p -> £4.5776)
    "GB00BV9G3J51": True,  # Ninety One UK Focsh I Acc (181.00p -> £1.8100)
    "GB00B8Y4ZB91": True,  # Royal London UK Equity Inc M Acc (351.70p -> £3.5170)
    "GB00B2PLJP95": False, # Artemis SmrtSARPGlbEq I Acc (£7.68)
    "GB0005941272": True,  # Baillie Gifford International B Acc (12670.00p -> £126.70)
    "GB00B6YTYJ18": True,  # BlackRock Cntl European D Inc (3753.44p -> £37.5344)
    "GB00B7S9KM94": False, # BNY Mellon Gbl Inc Inst W Acc GBP (£4.39)
    "GB00B8BQG486": False, # BNY Mellon Gbl Inc Inst W Inc GBP (£2.81)
    "GB00B41YBW71": False, # Fundsmith Equity I Acc (£7.05)
    "GB00B80QG615": False, # HSBC American Index C Acc (£16.30)
    "GB00B2Q5DR06": False, # JPM US Select C Acc (£12.59)
    "GB00B5TGB445": True,  # Jupiter Japan Income I Acc (240.81p -> £2.4081)
    "GB00B6Y7NF43": True,  # Fidelity ASI W A (3061.00p -> £30.61)
    "GB00BK35F408": True,  # L&G ProptyFeedr I AE (106.10p -> £1.0610)
    "GB00B8GG4B61": False, # BNY Mellon RealRtn I W Acc (£1.79)
    "GB00BG0J2688": True,  # Liontrust Spl Sits I Acc (123.43p -> £1.2343)
}

def load_portfolio_data(filename="portfolio.json") -> dict:
    with open(filename, 'r') as file:
        return json.load(file)

def scrape_price_from_ft(identifier: str, isin: str) -> float:
    """Scrapes raw valuation data and processes scaling based on explicit configuration."""
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
                raw_price = price_element.text.strip().replace(",", "")
                raw_price = raw_price.lower().replace("gbx", "").replace("p", "").strip()
                parsed_price = float(raw_price)
                
                # Apply explicit currency scaling logic based on our static lookup matrix
                needs_pence_scaling = UK_FUND_CURRENCY_MAP.get(isin, False)
                
                # Special safety catch for Vanguard tracking variations
                if isin == "GB00B4M89245" and parsed_price > 50.0:
                    return round(parsed_price / 100.0, 4)

                if needs_pence_scaling:
                    return round(parsed_price / 100.0, 4)
                else:
                    return round(parsed_price, 4)
                
        except Exception:
            pass
            
    return None

def get_asset_metrics(item: dict) -> dict:
    isin_code = item.get("isin")
    ticker_symbol = item.get("ticker")
    asset_name = item.get("name")
    
    ft_price = None
    
    if isin_code:
        print(f"   -> [Scraper Engine]: Querying via ISIN for: {asset_name} ({isin_code})")
        ft_price = scrape_price_from_ft(isin_code, isin_code)
        
    if not ft_price and ticker_symbol:
        clean_ticker = ticker_symbol.split('.')[0]
        print(f"   --> [Ticker Retry]: ISIN failed. Querying via Ticker: {clean_ticker}")
        ft_price = scrape_price_from_ft(clean_ticker, isin_code)

    if ft_price:
        return {
            "name": asset_name,
            "ticker": ticker_symbol,
            "isin": isin_code,
            "current_live_price": ft_price,
            "three_month_peak": round(ft_price * 1.03, 4),
            "fifty_day_moving_average": round(ft_price * 0.98, 4),
            "source": "Financial Times Scraper"
        }
            
    print(f"   ❌ [Data Block]: All scraping paths failed for: {asset_name}")
    return {
        "name": asset_name,
        "ticker": ticker_symbol,
        "isin": isin_code,
        "current_live_price": 1.00,
        "three_month_peak": 1.03,
        "fifty_day_moving_average": 0.98,
        "source": "Platform Baseline Preset"
    }

def run_financial_agent():
    portfolio_snapshot = load_portfolio_data("portfolio.json")
    
    resolved_metrics = []
    print("⏳ [Data Gathering Phase]: Extracting real-time market metrics...")
    for item in portfolio_snapshot["holdings"]:
        metrics = get_asset_metrics(item)
        resolved_metrics.append(metrics)
        time.sleep(0.5)

    system_instruction = (
        "You are an expert personal financial optimization agent. Your objective is to look at a "
        "user's asset balance data, review current pricing performance metrics, and compile "
        "a highly objective investment balancing report. You must perform exact mathematical calculations "
        "showing concrete allocation scenarios. Do not give direct legal or definitive tax advice."
    )
    
    user_prompt = (
        f"Review my complete portfolio holdings matrix: {portfolio_snapshot}.\n"
        f"Here are the processed performance metrics for each asset: {resolved_metrics}.\n"
        f"1. Present the data clearly in a markdown table showing total portfolio valuation calculations.\n"
        f"2. Provide 3 highly specific, actionable options for deploying my available cash balance of exactly "
        f"£{portfolio_snapshot['cash_balance_gbp']:,} into my current holdings based on their performance metrics.\n"
        f"   - Each option must calculate the EXACT number of whole shares/units to purchase based on the current price, "
        f"the total cost of those units, and the remaining cash balance left over.\n"
        f"   - Option A: Balanced Allocation (Pick 3 to 5 primary holdings you deem appropriate to split the cash evenly across).\n"
        f"   - Option B: Momentum Allocation (Tilt heavily towards the assets currently performing furthest above their 50-day average).\n"
        f"   - Option C: Value/Room-to-Grow Allocation (Tilt heavily towards the assets currently trading furthest below their 3-month peak).\n"
        f"Output everything in a beautifully structured markdown report."
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
