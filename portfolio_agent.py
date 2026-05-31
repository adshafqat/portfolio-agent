import os
import ssl
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
    raise ValueError(
        "Error: GEMINI_API_KEY not found. Please ensure your .env file is present "
        "in this directory and formatted as: GEMINI_API_KEY=\"your_key_here\""
    )

# 3. Initialize the official Google GenAI Client
client = genai.Client(api_key=api_key)

# We utilize gemini-2.5-flash as it features native support for automated tool orchestration schemas
MODEL_ID = "gemini-2.5-flash"

# 4. Mock Profile Data Input
MOCK_PORTFOLIO_SNAPSHOT = {
    "cash_balance_usd": 12500.00,
    "holdings": [
        {"ticker": "AAPL", "shares_owned": 15},
        {"ticker": "TSLA", "shares_owned": 8},
        {"ticker": "MSFT", "shares_owned": 5}
    ]
}

# 5. Define the Analytical Tool for the Agent
def get_stock_metrics(ticker_symbol: str) -> dict:
    """
    Fetches real-time market data, the 50-day moving average, and 3-month high 
    metrics for a specified stock ticker symbol from Yahoo Finance.
    
    Args:
        ticker_symbol: The string stock symbol (e.g., 'AAPL', 'MSFT').
        
    Returns:
        A dictionary containing live pricing variables and trend tracking metrics.
    """
    print(f"   -> [Tool Executing]: Extracting live market data for ticker: {ticker_symbol}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        history_3m = ticker.history(period="3mo")
        history_50d = ticker.history(period="3mo")
        
        if history_3m.empty:
            return {"error": f"No market data could be retrieved for ticker {ticker_symbol}"}
            
        current_price = float(history_3m['Close'].iloc[-1])
        three_month_peak = float(history_3m['High'].max())
        fifty_day_moving_avg = float(history_50d['Close'].mean())
        
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
    system_instruction = (
        "You are an expert personal financial optimization agent. Your objective is to look at a "
        "user's asset balance data, execute market tools to find historical performance, and compile "
        "a highly objective markdown investment balancing report. Do not give direct legal or definitive tax advice; "
        "provide structured options based on live data metrics."
    )
    
    user_prompt = (
        f"Review my current holdings and cash position: {MOCK_PORTFOLIO_SNAPSHOT}. "
        f"Use the get_stock_metrics tool to look up live market data for each of my holdings. "
        f"Evaluate which ones are performing favorably compared to their averages, and write a "
        f"structured markdown report recommending alternative ways I could deploy my $12,500 cash balance."
    )
    
    print("🚀 [Agent Initialization]: Spinning up reasoning engine loop...")
    
    try:
        # Create a managed chat container to handle automated execution loops
        chat = client.chats.create(
            model=MODEL_ID,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[get_stock_metrics],  # Map the execution capability tool
                temperature=0.2             # Low temperature optimizes analytical consistency
            )
        )
        
        # Fire the prompt inside the tracking session container
        response = chat.send_message(user_prompt)
        
        print("\n=== AGENT OUTPUT REPORT ===\n")
        print(response.text)
        print("\n============================\n")
        
    except Exception as e:
        print(f"\n❌ [Critical Engine Failure]: Framework loop crashed: {str(e)}")

if __name__ == "__main__":
    run_financial_agent()