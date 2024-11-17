# agents/market_data_agent.py
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime, timedelta
import asyncio
import pandas as pd

class MarketDataAgent:
    async def execute(self, tickers: List[str], days: int = 90) -> Dict[str, Any]:
        """Fetch stock data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            market_data = {}
            
            for ticker in tickers:
                try:
                    print(f"Fetching data for {ticker}...")
                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        # Use proper pandas methods to handle the data
                        close_prices = hist['Close']
                        latest_price = float(close_prices.iloc[-1])
                        first_price = float(close_prices.iloc[0])
                        price_change = latest_price - first_price
                        price_change_percent = (price_change / first_price * 100)

                        market_data[ticker] = {
                            "historical_prices": close_prices.ffill().tolist(),  # Using ffill() instead of fillna
                            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
                            "volumes": hist['Volume'].fillna(0).astype(int).tolist(),
                            "info": {
                                "current_price": latest_price,
                                "price_change": float(price_change),
                                "price_change_percent": float(price_change_percent),
                                "avg_volume": int(hist['Volume'].mean()),
                                "min_price": float(close_prices.min()),
                                "max_price": float(close_prices.max()),
                                "avg_price": float(close_prices.mean())
                            }
                        }
                        
                        print(f"Successfully fetched data for {ticker}")
                        print(f"Current price: ${latest_price:.2f}")
                        print(f"Price change: {price_change_percent:.2f}%")
                    else:
                        print(f"No data available for {ticker}")
                        
                except Exception as e:
                    print(f"Error fetching data for {ticker}: {str(e)}")
                    continue
                    
            return {"market_data": market_data}
            
        except Exception as e:
            print(f"Global error in execute: {str(e)}")
            return {"market_data": {}}

async def test_market_data():
    """Test the MarketDataAgent with various scenarios"""
    print("\n=== Testing Market Data Agent ===")
    agent = MarketDataAgent()
    
    test_scenarios = [
        {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "days": 30,
            "description": "Major tech stocks"
        },
        {
            "tickers": ["TSLA", "NVDA"],
            "days": 30,
            "description": "High-growth tech stocks"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['description']}")
        print(f"Tickers: {scenario['tickers']}")
        print(f"Days: {scenario['days']}")
        
        try:
            result = await agent.execute(scenario['tickers'], scenario['days'])
            print("\nResults:")
            for ticker, data in result['market_data'].items():
                print(f"\n{ticker}:")
                print(f"  Days of data: {len(data['historical_prices'])}")
                info = data['info']
                print(f"  Latest price: ${info['current_price']:.2f}")
                print(f"  Price range: ${info['min_price']:.2f} - ${info['max_price']:.2f}")
                print(f"  Average price: ${info['avg_price']:.2f}")
                print(f"  Price change: {info['price_change_percent']:.2f}%")
                print(f"  Latest date: {data['dates'][-1]}")
                print(f"  Average volume: {info['avg_volume']:,}")
                print(f"  Latest volume: {data['volumes'][-1]:,}")
        except Exception as e:
            print(f"Error in scenario: {str(e)}")

def main():
    """Run the test"""
    print("Starting Market Data Agent Test...")
    asyncio.run(test_market_data())

if __name__ == "__main__":
    main()