import os
import ssl
import json
from datetime import datetime
import yfinance as yf
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

def get_stock_metrics(ticker_symbol: str, fallback_item: dict = None) -> dict:
    """Fetches real-time yfinance data. Safely falls back to platform metrics on failure."""
    print(f"   -> [Data Request]: Fetching metrics for: {ticker_symbol}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_3mo = ticker.history(period="3mo")
        
        if history_3mo.empty or len(history_3mo) < 2:
            raise ValueError("No historical tracking values available.")
            
        current_price = float(history_3mo['Close'].iloc[-1])
        three_month_peak = float(history_3mo['High'].max())
        fifty_day_moving_avg = float(history_3mo['Close'].tail(50).mean())
        
        return {
            "ticker": ticker_symbol,
            "current_live_price": round(current_price, 2),
            "three_month_peak": round(three_month_peak, 2),
            "fifty_day_moving_average": round(fifty_day_moving_avg, 2),
            "source": "Live Market API"
        }
    except Exception:
        if fallback_item:
            return {
                "ticker": ticker_symbol,
                "current_live_price": fallback_item.get("current_price"),
                "three_month_peak": fallback_item.get("three_month_peak"),
                "fifty_day_moving_average": fallback_item.get("fifty_day_moving_average"),
                "source": "Platform Baseline Fallback"
            }
        return {"error": f"Data context unavailable for {ticker_symbol}"}

def run_financial_agent():
    portfolio_snapshot = load_portfolio_data("portfolio.json")
    
    # Pre-fetch and assemble metrics locally to protect the loop structure
    resolved_metrics = []
    for item in portfolio_snapshot["holdings"]:
        metrics = get_stock_metrics(item["ticker"], fallback_item=item)
        resolved_metrics.append(metrics)

    system_instruction = (
        "You are an expert personal financial optimization agent. Your objective is to look at a "
        "user's asset balance data, review current pricing performance metrics, and compile "
        "a highly objective investment balancing report. You must perform exact mathematical calculations "
        "showing concrete allocation scenarios. Do not give direct legal or definitive tax advice."
    )
    
    user_prompt = (
        f"Review my current holdings and cash position: {portfolio_snapshot}.\n"
        f"Here are the processed performance metrics for each asset: {resolved_metrics}.\n"
        f"1. Present the current data cleanly in a markdown table showing total portfolio valuation calculations.\n"
        f"2. Provide 3 highly specific, actionable options for deploying the cash balance of exactly "
        f"£{portfolio_snapshot['cash_balance_gbp']:,} into my current holdings based on their performance metrics.\n"
        f"   - Each option must calculate the EXACT number of whole shares/units to purchase based on the current price, "
        f"the total cost of those units, and the remaining cash balance left over.\n"
        f"   - Option A: Balanced Allocation (split cash relatively evenly among holdings you deem appropriate).\n"
        f"   - Option B: Momentum Allocation (tilt heavily toward assets furthest above their 50-day average).\n"
        f"   - Option C: Value/Room-to-Grow Allocation (tilt heavily toward assets with the most room below their 3-month peak).\n"
        f"Output everything in a structured markdown report."
    )
    
    print("🚀 [Agent Initialization]: Spinning up reasoning engine loop...")
    
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
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