import numpy as np
from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime
import sys
from pathlib import Path
import json

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from app.utils.metrics import MetricsTracker
except ImportError:
    # Fallback to local metrics for testing
    from metrics import MetricsTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AnalysisAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_tracker = MetricsTracker()
        self.supported_metrics = {
            "avg_return": self._calculate_average_return,
            "volatility": self._calculate_volatility,
            "total_return": self._calculate_total_return,
            "sharpe_ratio": self._calculate_sharpe_ratio
        }

    async def execute(self, market_data: Dict[str, Any], metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analyze stock data and identify best performing stock with metrics tracking
        """
        self.metrics_tracker.start_operation("analysis_execution")
        self.logger.info(f"Starting analysis with metrics: {metrics}")

        try:
            if not metrics:
                metrics = list(self.supported_metrics.keys())
                self.logger.info(f"No metrics specified, using all: {metrics}")

            analysis_results = []
            for ticker, data in market_data.get("market_data", {}).items():
                self.metrics_tracker.start_operation(f"analyze_{ticker}")
                self.logger.info(f"Analyzing {ticker}")

                try:
                    prices = data.get("historical_prices", [])
                    if not prices:
                        self.logger.warning(f"No price data for {ticker}, skipping")
                        continue

                    prices = np.array(prices)
                    metrics_data = {}

                    # Calculate requested metrics
                    for metric in metrics:
                        self.metrics_tracker.start_operation(f"calculate_{metric}_{ticker}")
                        if metric in self.supported_metrics:
                            metrics_data[metric] = self.supported_metrics[metric](prices)
                            self.logger.debug(f"{ticker} {metric}: {metrics_data[metric]:.2f}")
                        else:
                            self.logger.warning(f"Unsupported metric requested: {metric}")
                        self.metrics_tracker.end_operation(f"calculate_{metric}_{ticker}")

                    # Calculate score (using avg_return as default score)
                    score = metrics_data.get("avg_return", 0.0)

                    analysis_results.append({
                        "ticker": ticker,
                        "score": score,
                        "key_metrics": metrics_data,
                        "analysis_timestamp": str(datetime.now())
                    })

                    self.metrics_tracker.end_operation(f"analyze_{ticker}")

                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {str(e)}")
                    self.metrics_tracker.end_operation(f"analyze_{ticker}")
                    continue

            self.metrics_tracker.start_operation("determine_best_stock")
            if analysis_results:
                analysis_results.sort(key=lambda x: x["score"], reverse=True)
                best_stock = {
                    "ticker": analysis_results[0]["ticker"],
                    "reason": f"Highest average return of {analysis_results[0]['key_metrics']['avg_return']:.2f}%",
                    "metrics": analysis_results[0]["key_metrics"]
                }
            else:
                best_stock = {"ticker": None, "reason": "No valid stocks to analyze"}
            self.metrics_tracker.end_operation("determine_best_stock")

            self.metrics_tracker.end_operation("analysis_execution")

            return {
                "analysis_results": analysis_results,
                "best_stock": best_stock,
                "meta": {
                    "execution_time_seconds": self.metrics_tracker.metrics["timing"]["analysis_execution"]["duration"],
                    "analyzed_stocks": len(analysis_results),
                    "metrics_computed": metrics,
                    "timestamp": str(datetime.now())
                },
                "metrics": self.metrics_tracker.get_final_metrics()
            }

        except Exception as e:
            self.logger.error(f"Error in execute: {str(e)}")
            self.metrics_tracker.end_operation("analysis_execution")
            raise

    def _calculate_average_return(self, prices: np.ndarray) -> float:
        """Calculate average daily return"""
        daily_returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100
        return float(np.mean(daily_returns))

    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility"""
        daily_returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100
        return float(np.std(daily_returns))

    def _calculate_total_return(self, prices: np.ndarray) -> float:
        """Calculate total return over the period"""
        return float((prices[-1] - prices[0]) / prices[0] * 100)

    def _calculate_sharpe_ratio(self, prices: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        daily_returns = (prices[1:] - prices[:-1]) / prices[:-1]
        excess_returns = daily_returns - (risk_free_rate / 252)
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


async def test_analysis():
    """Test the AnalysisAgent with comprehensive scenarios and detailed metrics display"""
    logger = logging.getLogger("TestAnalysis")
    logger.info("Starting Analysis Agent Tests")

    test_scenarios = [
        {
            "name": "Basic Stock Analysis",
            "data": {
                "market_data": {
                    "AAPL": {
                        "historical_prices": [150.0, 152.5, 151.0, 153.0, 155.0],
                        "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                    },
                    "MSFT": {
                        "historical_prices": [280.0, 285.0, 282.0, 288.0, 290.0],
                        "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                    }
                }
            },
            "metrics": ["avg_return", "volatility"]
        },
        # ... other test scenarios remain the same ...
    ]

    for scenario in test_scenarios:
        print(f"\n{'=' * 50}")
        print(f"Executing test scenario: {scenario['name']}")
        print(f"{'=' * 50}")

        agent = AnalysisAgent()

        try:
            result = await agent.execute(scenario["data"], scenario["metrics"])

            # Print analysis results with improved formatting
            print("\nAnalysis Results:")
            if result["analysis_results"]:
                for analysis in result["analysis_results"]:
                    print(f"\n{analysis['ticker']}:")
                    # Format metrics in aligned columns
                    max_metric_length = max(len(metric) for metric in analysis["key_metrics"].keys())
                    for metric, value in analysis["key_metrics"].items():
                        print(f"  {metric:<{max_metric_length}}: {value:>8.2f}")
                    print(f"  {'Score':<{max_metric_length}}: {analysis['score']:>8.2f}")
            else:
                print("  No analysis results available")

            # Print best stock with improved formatting
            print(f"\nBest Performing Stock:")
            print(f"  Ticker: {result['best_stock']['ticker']}")
            print(f"  Reason: {result['best_stock']['reason']}")

            # Print metrics with improved formatting
            print("\nExecution Metrics:")
            metrics = result.get('metrics', {}).get('execution_metrics', {})
            if metrics:
                # Total duration in both seconds and microseconds
                total_duration = metrics.get('total_duration', 0)
                print(f"  Total Duration: {total_duration:.3f} seconds ({total_duration * 1000000:.0f} μs)")

                # Group and sort operation timings
                timings = metrics.get('operation_timings', {})
                if timings:
                    print("\n  Operation Timings:")
                    # Group timings by operation type
                    grouped_timings = {
                        "Main Operations": {k: v for k, v in timings.items()
                                            if k in ["analysis_execution", "determine_best_stock"]},
                        "Stock Analysis": {k: v for k, v in timings.items()
                                           if k.startswith("analyze_") and not k.startswith("analyze_metric")},
                        "Metric Calculations": {k: v for k, v in timings.items()
                                                if k.startswith("calculate_")}
                    }

                    for group_name, group_timings in grouped_timings.items():
                        if group_timings:
                            print(f"\n    {group_name}:")
                            max_op_length = max(len(k) for k in group_timings.keys())
                            for op, timing in sorted(group_timings.items(),
                                                     key=lambda x: x[1].get('duration', 0),
                                                     reverse=True):
                                duration = timing.get('duration', 0)
                                print(f"      {op:<{max_op_length}}: {duration:.6f} seconds "
                                      f"({duration * 1000000:.0f} μs)")

                # Memory usage with improved formatting
                memory = metrics.get('memory_usage', {})
                if memory:
                    print("\n  Memory Usage:")
                    print(f"    Current Memory: {memory.get('current_memory_mb', 0):>8.2f} MB")
                    print(f"    Peak Memory:    {memory.get('peak_memory_mb', 0):>8.2f} MB")
                    print(f"    Memory Delta:   {memory.get('memory_increase_mb', 0):>8.2f} MB")

        except Exception as e:
            logger.error(f"Error in test scenario {scenario['name']}: {str(e)}")
            print(f"\nError executing scenario: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_analysis())
