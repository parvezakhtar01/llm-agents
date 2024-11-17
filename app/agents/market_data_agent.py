# agents/market_data_agent.py
import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import logging
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from app.utils.metrics import MetricsTracker
except ImportError:
    # Fallback to local metrics for testing
    from metrics import MetricsTracker


class MarketDataAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_tracker = MetricsTracker()

    async def execute(self, tickers: List[str], days: int = 90) -> Dict[str, Any]:
        """Fetch stock data from Yahoo Finance with metrics tracking"""
        self.metrics_tracker.start_operation("market_data_execution")

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            market_data = {}

            for ticker in tickers:
                try:
                    self.metrics_tracker.start_operation(f"fetch_data_{ticker}")
                    self.logger.info(f"Fetching data for {ticker}...")

                    stock = yf.Ticker(ticker)
                    hist = stock.history(start=start_date, end=end_date)

                    if not hist.empty:
                        # Track data processing
                        self.metrics_tracker.start_operation(f"process_data_{ticker}")

                        close_prices = hist['Close']
                        latest_price = float(close_prices.iloc[-1])
                        first_price = float(close_prices.iloc[0])
                        price_change = latest_price - first_price
                        price_change_percent = (price_change / first_price * 100)

                        market_data[ticker] = {
                            "historical_prices": close_prices.ffill().tolist(),
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

                        self.logger.info(f"Successfully fetched data for {ticker}")
                        self.logger.info(f"Current price: ${latest_price:.2f}")
                        self.logger.info(f"Price change: {price_change_percent:.2f}%")

                        self.metrics_tracker.end_operation(f"process_data_{ticker}")
                    else:
                        self.logger.warning(f"No data available for {ticker}")

                    self.metrics_tracker.end_operation(f"fetch_data_{ticker}")

                except Exception as e:
                    self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                    self.metrics_tracker.end_operation(f"fetch_data_{ticker}")
                    continue

            self.metrics_tracker.end_operation("market_data_execution")
            return {
                "market_data": market_data,
                "metrics": self.metrics_tracker.get_final_metrics()
            }

        except Exception as e:
            self.logger.error(f"Global error in execute: {str(e)}")
            self.metrics_tracker.end_operation("market_data_execution")
            return {
                "market_data": {},
                "metrics": self.metrics_tracker.get_final_metrics()
            }


async def test_market_data():
    """Test the MarketDataAgent with various scenarios and metrics display"""
    print("\n=== Testing Market Data Agent ===")

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
        # Create new agent instance for each scenario to reset metrics
        agent = MarketDataAgent()

        print(f"\n{'=' * 50}")
        print(f"Testing scenario: {scenario['description']}")
        print(f"Tickers: {scenario['tickers']}")
        print(f"Days: {scenario['days']}")
        print(f"{'=' * 50}")

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

            # Print metrics
            print("\nExecution Metrics:")
            metrics = result.get('metrics', {}).get('execution_metrics', {})
            if metrics:
                print(f"  Total Duration: {metrics.get('total_duration', 0):.3f} seconds")

                # Operation timings
                print("\n  Operation Timings:")
                operation_timings = metrics.get('operation_timings', {})
                # Filter timings to only show operations for current scenario's tickers
                relevant_timings = {
                    op: timing for op, timing in operation_timings.items()
                    if op == "market_data_execution" or
                       any(ticker in op for ticker in scenario['tickers'])
                }
                for op, timing in relevant_timings.items():
                    if timing.get('duration'):
                        print(f"    {op}: {timing['duration']:.3f} seconds")

                # Memory usage
                memory = metrics.get('memory_usage', {})
                if memory:
                    print("\n  Memory Usage:")
                    print(f"    Current Memory: {memory.get('current_memory_mb', 0):.2f} MB")
                    print(f"    Peak Memory: {memory.get('peak_memory_mb', 0):.2f} MB")
                    print(f"    Memory Increase: {memory.get('memory_increase_mb', 0):.2f} MB")

        except Exception as e:
            print(f"Error in scenario: {str(e)}")


def main():
    """Run the test"""
    print("Starting Market Data Agent Test...")
    asyncio.run(test_market_data())


if __name__ == "__main__":
    main()
