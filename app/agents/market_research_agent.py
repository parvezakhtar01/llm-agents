from dotenv import load_dotenv
from serpapi import GoogleSearch
import os
from typing import List, Dict, Any
import json
import asyncio
from openai import AsyncOpenAI
import logging
import sys

# Import metrics from utils
try:
    from app.utils.metrics import MetricsTracker
except ImportError:
    # Fallback for direct script execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from app.utils.metrics import MetricsTracker

SYSTEM_PROMPT = """
Key Objectives:
1. Identify relevant stocks from user queries
2. Validate stock symbols are correct
3. Consider market sectors and trends
4. Filter out irrelevant matches
5. Assign confidence scores

Guidelines:
- Prioritize widely-traded stocks on major exchanges
- Consider sector relationships (e.g., when searching for "AI stocks", include NVIDIA, Microsoft)
- Use multiple validation sources
- Provide reasoning for each stock selection
- Consider market capitalization and trading volume

Output Format Example:
{
    "analysis": {
        "understanding": "Query focuses on AI and cloud computing sectors",
        "search_strategy": "Looking for major tech companies with AI/cloud focus",
        "confidence_explanation": "High confidence in direct matches, medium for sector leaders"
    },
    "extracted_stocks": [
        {
            "ticker": "NVDA",
            "company_name": "NVIDIA Corporation",
            "sector": "Technology",
            "confidence": 0.9,
            "reasoning": "Leading AI chip manufacturer, directly relevant"
        }
    ]
}"""


class MarketResearchAgent:
    def __init__(self, serpapi_key: str, openai_key: str = None):
        """Initialize the agent with API keys and metrics tracker"""
        self.serpapi_key = serpapi_key
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.system_prompt = SYSTEM_PROMPT
        self.metrics_tracker = MetricsTracker()
        self.logger = logging.getLogger(__name__)

        self.known_tickers = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "GOOG": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "TSLA": "Tesla Inc.",
            "AMD": "Advanced Micro Devices Inc.",
            "NFLX": "Netflix Inc.",
            "IBM": "IBM Corporation",
            "CSCO": "Cisco Systems Inc.",
            "INTC": "Intel Corporation"
        }

    def _extract_direct_tickers(self, query: str) -> List[str]:
        """Extract known tickers directly mentioned in the query"""
        self.metrics_tracker.start_operation("extract_direct_tickers")
        found_tickers = []
        words = query.upper().split()
        for word in words:
            if word in self.known_tickers:
                found_tickers.append(word)
        self.metrics_tracker.end_operation("extract_direct_tickers")
        return found_tickers

    def _extract_stock_mentions(self, results: Dict) -> List[Dict]:
        """Extract stock mentions from search results"""
        self.metrics_tracker.start_operation("extract_stock_mentions")
        stocks = []
        seen_tickers = set()

        try:
            if "organic_results" in results:
                for result in results["organic_results"]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    text = f"{title} {snippet}"

                    # Check for known tickers
                    for ticker in self.known_tickers:
                        if ticker in text.upper() and ticker not in seen_tickers:
                            stocks.append({
                                "ticker": ticker,
                                "company_name": self.known_tickers[ticker],
                                "confidence": 0.8,
                                "source": "search_result"
                            })
                            seen_tickers.add(ticker)

        except Exception as e:
            self.logger.error(f"Error extracting stocks: {str(e)}")

        self.metrics_tracker.end_operation("extract_stock_mentions")
        return stocks

    async def _analyze_with_llm(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze search results if OpenAI client is available"""
        self.metrics_tracker.start_operation("llm_analysis")

        if not hasattr(self, 'openai_client'):
            self.metrics_tracker.end_operation("llm_analysis")
            return {
                "analysis": {
                    "understanding": f"Direct analysis of query: {query}",
                    "search_strategy": "Using direct ticker matching and search results",
                    "confidence_explanation": "Based on direct matches and known tickers"
                },
                "extracted_stocks": search_results
            }

        try:
            # Create the user message with the context
            user_message = f"""
            Analyze the following stock market query and search results:

            Query: {query}

            Search Results: {json.dumps(search_results, indent=2)}

            Known Tickers: {json.dumps(list(self.known_tickers.keys()))}
            """

            model_params = {
                "model": "gpt-4-1106-preview",
                "temperature": 0.7,
                "response_format": {"type": "json_object"}
            }

            response = await self.openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                **model_params
            )

            # Track model usage
            self.metrics_tracker.track_model_usage(
                model_name=model_params["model"],
                operation_type="market_research_analysis",
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                parameters=model_params
            )

            result = json.loads(response.choices[0].message.content)
            self.metrics_tracker.end_operation("llm_analysis")
            return result

        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {e}")
            self.metrics_tracker.end_operation("llm_analysis")
            return {
                "analysis": {
                    "understanding": f"Query analysis (fallback): {query}",
                    "confidence_explanation": "Using direct matching due to LLM error"
                },
                "extracted_stocks": search_results
            }

    async def execute(self, query: str) -> Dict[str, Any]:
        """Main execution method with metrics tracking"""
        self.metrics_tracker.start_operation("market_research_execution")

        try:
            self.logger.info("=== Starting Market Research Analysis ===")

            # Track direct ticker extraction
            direct_tickers = self._extract_direct_tickers(query)
            if direct_tickers:
                stocks = [
                    {
                        "ticker": ticker,
                        "company_name": self.known_tickers[ticker],
                        "confidence": 0.9,
                        "source": "direct_mention"
                    }
                    for ticker in direct_tickers
                ]

                self.metrics_tracker.end_operation("market_research_execution")
                return {
                    "found_stocks": stocks,
                    "confidence_score": 0.9,
                    "analysis": {
                        "understanding": "Direct ticker mentions found",
                        "strategy": "Using exact matches",
                        "confidence": "High confidence due to direct mentions"
                    },
                    "metrics": self.metrics_tracker.get_final_metrics()
                }

            # Track search operation
            self.metrics_tracker.start_operation("serp_search")
            search_query = f"{query} stock market ticker symbols NYSE NASDAQ"
            search = GoogleSearch({
                "q": search_query,
                "api_key": self.serpapi_key,
                "engine": "google",
                "google_domain": "google.com",
                "gl": "us",
                "hl": "en",
                "num": 10
            })

            results = search.get_dict()
            self.metrics_tracker.end_operation("serp_search")

            # Extract and analyze stocks
            stocks = self._extract_stock_mentions(results)

            # Get LLM analysis if available
            llm_analysis = await self._analyze_with_llm(query, stocks)

            self.metrics_tracker.end_operation("market_research_execution")
            return {
                "found_stocks": llm_analysis.get("extracted_stocks", stocks),
                "confidence_score": 0.8 if stocks else 0.3,
                "analysis": llm_analysis.get("analysis", {
                    "understanding": "Based on search results",
                    "strategy": "Using search and pattern matching",
                    "confidence": "Medium confidence from search results"
                }),
                "metrics": self.metrics_tracker.get_final_metrics()
            }

        except Exception as e:
            self.logger.error(f"Error in market research: {str(e)}")
            self.metrics_tracker.end_operation("market_research_execution")
            return {
                "found_stocks": [],
                "confidence_score": 0.0,
                "analysis": {"error": str(e)},
                "metrics": self.metrics_tracker.get_final_metrics()
            }


async def test_market_research():
    """Test the MarketResearchAgent with metrics display"""
    load_dotenv()

    serpapi_key = os.getenv("SERPAPI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")  # Optional

    if not serpapi_key:
        print("Error: SERPAPI_API_KEY not found in environment variables")
        return

    agent = MarketResearchAgent(serpapi_key, openai_key)

    test_queries = [
        "Top tech companies AAPL MSFT GOOGL",
        "NVIDIA and AMD semiconductor performance",
        "Tesla TSLA stock analysis",
        "FAANG stocks META NFLX AMZN",
        "Cloud computing leaders MSFT AMZN GOOGL"
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"Testing query: {query}")
        print(f"{'=' * 50}")
        try:
            result = await agent.execute(query)
            print("\nResults:")
            if result.get('found_stocks'):
                print(f"Found {len(result['found_stocks'])} stocks:")
                for stock in result['found_stocks']:
                    print(f"  {stock['ticker']}: {stock['company_name']}")
                    print(f"    Confidence: {stock['confidence']}")
                    print(f"    Source: {stock['source']}")
            print(f"\nOverall confidence score: {result.get('confidence_score')}")

            # Print metrics
            print("\nExecution Metrics:")
            metrics = result.get('metrics', {}).get('execution_metrics', {})
            if metrics:
                print(f"  Total Duration: {metrics.get('total_duration', 0):.3f} seconds")

                # Operation timings
                print("\n  Operation Timings:")
                for op, timing in metrics.get('operation_timings', {}).items():
                    if timing.get('duration'):
                        print(f"    {op}: {timing['duration']:.3f} seconds")

                # Memory usage
                memory = metrics.get('memory_usage', {})
                if memory:
                    print("\n  Memory Usage:")
                    print(f"    Current Memory: {memory.get('current_memory_mb', 0):.2f} MB")
                    print(f"    Peak Memory: {memory.get('peak_memory_mb', 0):.2f} MB")
                    print(f"    Memory Increase: {memory.get('memory_increase_mb', 0):.2f} MB")

                # Token usage
                tokens = metrics.get('token_usage', {})
                if tokens:
                    print("\n  Token Usage:")
                    print(f"    Prompt Tokens: {tokens.get('prompt_tokens', 0)}")
                    print(f"    Completion Tokens: {tokens.get('completion_tokens', 0)}")
                    print(f"    Total Tokens: {tokens.get('total_tokens', 0)}")

                # Model usage
                model_calls = metrics.get('model_usage', {}).get('calls', [])
                if model_calls:
                    print("\n  Model Usage:")
                    for call in model_calls:
                        print(f"    Operation: {call.get('operation')}")
                        print(f"    Model: {call.get('model')}")
                        print(f"    Total Tokens: {call.get('tokens', {}).get('total', 0)}")

        except Exception as e:
            print(f"Error testing query '{query}': {str(e)}")
            print(f"Error details: {traceback.format_exc()}")


if __name__ == "__main__":
    import traceback

    asyncio.run(test_market_research())
