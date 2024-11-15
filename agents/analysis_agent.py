# agents/analysis_agent.py
import numpy as np
from typing import Dict, Any, List
import asyncio

class AnalysisAgent:
    async def execute(self, market_data: Dict[str, Any], metrics: List[str]) -> Dict[str, Any]:
        """Analyze stock data and identify best performing stock"""
        analysis_results = []
        
        for ticker, data in market_data["market_data"].items():
            prices = data["historical_prices"]
            
            if not prices:
                continue
                
            prices = np.array(prices)
            
            metrics_data = {
                "avg_return": self._calculate_average_return(prices),
                "volatility": self._calculate_volatility(prices),
                "total_return": self._calculate_total_return(prices)
            }
            
            analysis_results.append({
                "ticker": ticker,
                "score": metrics_data["avg_return"],
                "key_metrics": metrics_data
            })
        
        if analysis_results:
            analysis_results.sort(key=lambda x: x["score"], reverse=True)
            best_stock = {
                "ticker": analysis_results[0]["ticker"],
                "reason": f"Highest average return of {analysis_results[0]['key_metrics']['avg_return']:.2f}%"
            }
        else:
            best_stock = {"ticker": None, "reason": "No valid stocks to analyze"}
        
        return {
            "analysis_results": analysis_results,
            "best_stock": best_stock
        }
    
    def _calculate_average_return(self, prices: np.ndarray) -> float:
        daily_returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100
        return float(np.mean(daily_returns))
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        daily_returns = (prices[1:] - prices[:-1]) / prices[:-1] * 100
        return float(np.std(daily_returns))
    
    def _calculate_total_return(self, prices: np.ndarray) -> float:
        return float((prices[-1] - prices[0]) / prices[0] * 100)

# Test execution code
async def test_analysis():
    """Test the AnalysisAgent with sample data"""
    print("\n=== Testing Analysis Agent ===")
    agent = AnalysisAgent()
    
    # Sample market data for testing
    test_data = {
        "market_data": {
            "AAPL": {
                "historical_prices": [150.0, 152.5, 151.0, 153.0, 155.0],
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "volumes": [1000000] * 5
            },
            "MSFT": {
                "historical_prices": [280.0, 285.0, 282.0, 288.0, 290.0],
                "dates": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
                "volumes": [2000000] * 5
            }
        }
    }
    
    test_scenarios = [
        {
            "metrics": ["average_return"],
            "description": "Basic performance metrics"
        },
        {
            "metrics": ["average_return", "volatility", "total_return"],
            "description": "Comprehensive analysis"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['description']}")
        try:
            result = await agent.execute(test_data, scenario['metrics'])
            print("\nResults:")
            print(f"Best performing stock: {result['best_stock']['ticker']}")
            print(f"Reason: {result['best_stock']['reason']}")
            
            print("\nDetailed Analysis:")
            for stock_result in result['analysis_results']:
                print(f"\n{stock_result['ticker']}:")
                for metric, value in stock_result['key_metrics'].items():
                    print(f"  {metric}: {value:.2f}")
        except Exception as e:
            print(f"Error in scenario: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_analysis())