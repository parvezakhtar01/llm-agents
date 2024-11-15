# test_agents_with_instructions.py
import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import json
from dotenv import load_dotenv

from agents.market_research_agent import MarketResearchAgent
from agents.market_data_agent import MarketDataAgent
from agents.analysis_agent import AnalysisAgent

load_dotenv()

async def test_market_research_agent():
    """
    Test Market Research Agent with various real-world instructions
    Evaluates the agent's ability to understand and process different types of queries
    """
    print("\n=== Testing Market Research Agent ===")
    agent = MarketResearchAgent(os.getenv("SERPAPI_API_KEY"))
    
    instructions = [
        {
            "instruction": "Find the top performing tech stocks focusing on AI and cloud computing companies",
            "expected_outcome": "Should return AI and cloud tech companies",
            "evaluation_criteria": ["Contains major tech companies", "Includes AI/cloud focused companies"]
        },
        {
            "instruction": "Identify the best renewable energy stocks that have shown growth in the last quarter",
            "expected_outcome": "Should return renewable energy companies",
            "evaluation_criteria": ["Contains renewable energy companies", "Recent market performers"]
        },
        {
            "instruction": "List the top healthcare stocks specializing in biotechnology and pharmaceuticals",
            "expected_outcome": "Should return biotech and pharma companies",
            "evaluation_criteria": ["Contains healthcare companies", "Includes biotech firms"]
        }
    ]
    
    for idx, test in enumerate(instructions, 1):
        print(f"\nTest {idx}: {test['instruction']}")
        print("Expected outcome:", test['expected_outcome'])
        
        try:
            result = await agent.execute(test['instruction'])
            print("\nResults:")
            print(f"Found {len(result['found_stocks'])} stocks")
            print("Sample stocks:", result['found_stocks'][:3])
            print("Confidence score:", result['confidence_score'])
            
            # Simple evaluation
            if result['found_stocks'] and result['confidence_score'] > 0.5:
                print("✓ Agent returned relevant results")
            else:
                print("⚠ Agent might need improvement for this type of query")
                
        except Exception as e:
            print(f"Error: {str(e)}")

async def test_market_data_agent():
    """
    Test Market Data Agent with realistic data requests
    Evaluates the agent's ability to fetch and process stock data
    """
    print("\n=== Testing Market Data Agent ===")
    agent = MarketDataAgent()
    
    test_scenarios = [
        {
            "tickers": ["AAPL", "MSFT", "GOOGL"],
            "description": "Major tech stocks - should have complete data",
            "days": 90
        },
        {
            "tickers": ["TSLA", "NVDA", "AMD"],
            "description": "High-volatility tech stocks",
            "days": 30
        },
        {
            "tickers": ["PFE", "JNJ", "MRNA"],
            "description": "Healthcare stocks",
            "days": 60
        }
    ]
    
    for idx, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {idx}: {scenario['description']}")
        print(f"Tickers: {scenario['tickers']}")
        
        try:
            result = await agent.execute(scenario['tickers'], scenario['days'])
            
            print("\nResults:")
            for ticker, data in result['market_data'].items():
                data_points = len(data['historical_prices'])
                print(f"{ticker}: {data_points} days of data")
                
                # Basic data quality check
                if data_points > 0:
                    latest_price = data['historical_prices'][-1]
                    print(f"Latest price: {latest_price}")
                else:
                    print("⚠ No data available")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

async def test_analysis_agent():
    """
    Test Analysis Agent with realistic analysis scenarios
    Evaluates the agent's analytical capabilities and insights
    """
    print("\n=== Testing Analysis Agent ===")
    agent = AnalysisAgent()
    
    # Create sample market data for testing
    sample_data = {
        "market_data": {
            "AAPL": {
                "historical_prices": [150.0, 152.5, 151.0, 153.0, 155.0],
                "dates": [
                    (datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d')
                    for x in range(5, 0, -1)
                ],
                "volumes": [1000000] * 5
            },
            "MSFT": {
                "historical_prices": [280.0, 285.0, 282.0, 288.0, 290.0],
                "dates": [
                    (datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d')
                    for x in range(5, 0, -1)
                ],
                "volumes": [2000000] * 5
            }
        }
    }
    
    analysis_scenarios = [
        {
            "description": "Basic performance analysis",
            "metrics": ["average_return"],
            "evaluation_criteria": ["Returns calculated", "Best stock identified"]
        },
        {
            "description": "Comprehensive analysis",
            "metrics": ["average_return", "volatility", "total_return"],
            "evaluation_criteria": ["Multiple metrics", "Comparative analysis"]
        }
    ]
    
    for idx, scenario in enumerate(analysis_scenarios, 1):
        print(f"\nScenario {idx}: {scenario['description']}")
        
        try:
            result = await agent.execute(sample_data, scenario['metrics'])
            
            print("\nResults:")
            print(f"Best performing stock: {result['best_stock']['ticker']}")
            print(f"Reason: {result['best_stock']['reason']}")
            
            print("\nDetailed Analysis:")
            for stock_result in result['analysis_results']:
                print(f"\n{stock_result['ticker']}:")
                for metric, value in stock_result['key_metrics'].items():
                    print(f"  {metric}: {value:.2f}")
                    
        except Exception as e:
            print(f"Error: {str(e)}")

async def main():
    """Run all agent tests"""
    print("Starting Agent Testing with Real Instructions")
    print("============================================")
    
    await test_market_research_agent()
    await test_market_data_agent()
    await test_analysis_agent()
    
    print("\nTesting Complete!")

if __name__ == "__main__":
    asyncio.run(main())