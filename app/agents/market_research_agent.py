# agents/market_research_agent.py
from serpapi import GoogleSearch
import os
from typing import List, Dict, Any
import json
import asyncio
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
import re

SYSTEM_PROMPT = """You are an expert stock market researcher. Your role is to search and identify relevant stocks based on user queries and market trends.

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
        """Initialize the agent with API keys"""
        self.serpapi_key = serpapi_key
        if openai_key:
            self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.system_prompt = SYSTEM_PROMPT
        
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
        found_tickers = []
        words = query.upper().split()
        for word in words:
            if word in self.known_tickers:
                found_tickers.append(word)
        return found_tickers

    def _extract_stock_mentions(self, results: Dict) -> List[Dict]:
        """Extract stock mentions from search results"""
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
            print(f"Error extracting stocks: {str(e)}")
            
        return stocks

    async def _analyze_with_llm(self, query: str, search_results: List[Dict]) -> Dict[str, Any]:
        """Use LLM to analyze search results if OpenAI client is available"""
        if not hasattr(self, 'openai_client'):
            # Return simplified analysis without LLM
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

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            # Fallback to simplified analysis
            return {
                "analysis": {
                    "understanding": f"Query analysis (fallback): {query}",
                    "confidence_explanation": "Using direct matching due to LLM error"
                },
                "extracted_stocks": search_results
            }

    async def execute(self, query: str) -> Dict[str, Any]:
        """Main execution method"""
        try:
            print("\n=== Starting Market Research Analysis ===")
            
            # First, check for direct ticker mentions
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
                return {
                    "found_stocks": stocks,
                    "confidence_score": 0.9,
                    "analysis": {
                        "understanding": "Direct ticker mentions found",
                        "strategy": "Using exact matches",
                        "confidence": "High confidence due to direct mentions"
                    }
                }
            
            # If no direct tickers, perform search
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
            stocks = self._extract_stock_mentions(results)
            
            # Get LLM analysis if available
            llm_analysis = await self._analyze_with_llm(query, stocks)
            
            return {
                "found_stocks": llm_analysis.get("extracted_stocks", stocks),
                "confidence_score": 0.8 if stocks else 0.3,
                "analysis": llm_analysis.get("analysis", {
                    "understanding": "Based on search results",
                    "strategy": "Using search and pattern matching",
                    "confidence": "Medium confidence from search results"
                })
            }
            
        except Exception as e:
            print(f"Error in market research: {str(e)}")
            return {
                "found_stocks": [],
                "confidence_score": 0.0,
                "analysis": {"error": str(e)}
            }

async def test_market_research():
    """Test the MarketResearchAgent"""
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
        print(f"\nTesting query: {query}")
        try:
            result = await agent.execute(query)
            print("Results:")
            if result['found_stocks']:
                print(f"Found {len(result['found_stocks'])} stocks:")
                for stock in result['found_stocks']:
                    print(f"  {stock['ticker']}: {stock['company_name']}")
                    print(f"    Confidence: {stock['confidence']}")
                    print(f"    Source: {stock['source']}")
            print(f"Overall confidence score: {result['confidence_score']}")
        except Exception as e:
            print(f"Error testing query '{query}': {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_market_research())