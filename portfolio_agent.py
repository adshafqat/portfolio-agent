import os
import ssl
import json
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Force fix SSL context blockages common on headless Linux cloud instances
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 2. Load Environment Credentials from local .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Error: GEMINI_API_KEY not found in .env file.")

# 3. Initialize the official Google GenAI Client
client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-2.5-flash"

# 4. Dynamic Portfolio Loader Function
def load_portfolio_data(filename="portfolio.json") -> dict:
    """Reads and parses the asset allocation data from a local JSON file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"⚠️ {filename} not found. Falling back to default baseline profile.")
        return {
            "cash_balance_usd": 0.0,
            "holdings": []
        }
    except json.JSONDecodeError:
        raise ValueError(f"❌ Error: {filename} contains invalid JSON formatting.")

# 5. Define the Analytical Tool for the Agent
def get_stock_metrics(ticker_symbol: str) -> dict:
    """
    Fetches real-time market data, the 50-day moving average, and 3-month high 
    metrics for a specified stock ticker symbol from Yahoo Finance.
    """
    print(f"   -> [Tool Executing]: Extracting live market data for ticker: {ticker_symbol}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_3mo = ticker.history(period="3mo")
        
        if history_3mo.empty or len(history_3mo) < 2:
            return {"error": f"No market data could be retrieved for ticker {ticker_symbol}"}
            
        current_price = float(history_3mo['Close'].iloc[-1])
        three_month_peak = float(history_3mo['High'].max())
        
        closing_prices = history_3mo['Close']
        if len(closing_prices) >= 50:
            fifty_day_moving_avg = float(closing_prices.tail(50).mean())
        else:
            fifty_day_moving_avg = float(closing_prices.mean())
        
        return {
            "ticker": ticker_symbol,
            "current_live_price": round(current_price, 2),
            "three_month_peak": round(three_month_peak, 2),
            "fifty_day_moving_average": round(fifty_day_moving_avg, 2)
        }
    except Exception as e:
        return {"error": f"Failed to gather financial data for {ticker_symbol}: {str(e)}"}

# 6. Core Orchestration Engine
def run_financial_agent():
    # Load dynamic snapshot right when the script is run
    portfolio_snapshot = load_portfolio_data("portfolio.json")
    
    if not portfolio_snapshot["holdings"]:
        print("❌ Portfolio is empty or couldn't be read. Exiting.")
        return

    system_instruction = (
        "You are an expert personal financial optimization agent. Your objective is to look at a "
        "user's asset balance data, execute market tools to find historical performance, and compile "
        "a highly objective markdown investment balancing report. Do not give direct legal or definitive tax advice; "
        "provide structured options based on live data metrics."
    )
    
    user_prompt = (
        f"Review my current holdings and cash position: {portfolio_snapshot}. "
        f"Use the get_stock_metrics tool to look up live market data for each of my holdings. "
        f"Evaluate which ones are performing favorably compared to their averages, and write a "
        f"structured markdown report recommending alternative ways I could deploy my cash balance."
    )
    
    print("🚀 [Agent Initialization]: Spinning up reasoning engine loop...")
    
    try:
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[get_stock_metrics],
                temperature=0.2
            )
        )
        
        response = chat.send_message(user_prompt)
        
        print("\n=== AGENT OUTPUT REPORT ===\n")
        print(response.text)
        print("\n============================\n")
        
    except Exception as e:
        print(f"\n❌ [Critical Engine Failure]: Framework loop crashed: {str(e)}")

if __name__ == "__main__":
    run_financial_agent()