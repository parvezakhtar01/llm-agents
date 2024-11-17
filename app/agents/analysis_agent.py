import numpy as np
from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AnalysisAgent:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_metrics = {
            "avg_return": self._calculate_average_return,
            "volatility": self._calculate_volatility,
            "total_return": self._calculate_total_return,
            "sharpe_ratio": self._calculate_sharpe_ratio
        }

    async def execute(self, market_data: Dict[str, Any], metrics: List[str] = None) -> Dict[str, Any]:
        """
        Analyze stock data and identify best performing stock

        Args:
            market_data: Dictionary containing market data for multiple stocks
            metrics: List of metrics to calculate (defaults to all if None)

        Returns:
            Dictionary containing analysis results and best stock
        """
        self.logger.info(f"Starting analysis with metrics: {metrics}")
        analysis_results = []
        start_time = datetime.now()

        try:
            if not metrics:
                metrics = list(self.supported_metrics.keys())
                self.logger.info(f"No metrics specified, using all: {metrics}")

            for ticker, data in market_data.get("market_data", {}).items():
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
                        if metric in self.supported_metrics:
                            metrics_data[metric] = self.supported_metrics[metric](prices)
                            self.logger.debug(f"{ticker} {metric}: {metrics_data[metric]:.2f}")
                        else:
                            self.logger.warning(f"Unsupported metric requested: {metric}")

                    # Calculate score (using avg_return as default score)
                    score = metrics_data.get("avg_return", 0.0)

                    analysis_results.append({
                        "ticker": ticker,
                        "score": score,
                        "key_metrics": metrics_data,
                        "analysis_timestamp": str(datetime.now())
                    })

                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {str(e)}")
                    continue

            # Determine best stock
            if analysis_results:
                analysis_results.sort(key=lambda x: x["score"], reverse=True)
                best_stock = {
                    "ticker": analysis_results[0]["ticker"],
                    "reason": f"Highest average return of {analysis_results[0]['key_metrics']['avg_return']:.2f}%",
                    "metrics": analysis_results[0]["key_metrics"]
                }
            else:
                best_stock = {"ticker": None, "reason": "No valid stocks to analyze"}

            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Analysis completed in {execution_time:.2f} seconds")

            return {
                "analysis_results": analysis_results,
                "best_stock": best_stock,
                "meta": {
                    "execution_time_seconds": execution_time,
                    "analyzed_stocks": len(analysis_results),
                    "metrics_computed": metrics,
                    "timestamp": str(datetime.now())
                }
            }

        except Exception as e:
            self.logger.error(f"Error in execute: {str(e)}")
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
        excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))


# Enhanced test scenarios
async def test_analysis():
    """Test the AnalysisAgent with comprehensive scenarios"""
    logger = logging.getLogger("TestAnalysis")
    logger.info("Starting Analysis Agent Tests")

    agent = AnalysisAgent()

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
        {
            "name": "Empty Data Test",
            "data": {"market_data": {}},
            "metrics": ["avg_return"]
        },
        {
            "name": "Invalid Metrics Test",
            "data": {
                "market_data": {
                    "AAPL": {
                        "historical_prices": [150.0, 152.5, 151.0, 153.0, 155.0],
                        "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                    }
                }
            },
            "metrics": ["invalid_metric", "avg_return"]
        },
        {
            "name": "Comprehensive Analysis",
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
            "metrics": ["avg_return", "volatility", "total_return", "sharpe_ratio"]
        }
    ]

    for scenario in test_scenarios:
        logger.info(f"\nExecuting test scenario: {scenario['name']}")
        try:
            result = await agent.execute(scenario["data"], scenario["metrics"])
            logger.info(f"Test scenario completed: {scenario['name']}")
            logger.info(f"Results: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Error in test scenario {scenario['name']}: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_analysis())
