# tests/test_stock_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from app.main import app

client = TestClient(app)

# Mock data
MOCK_MARKET_RESEARCH_RESPONSE = {
    "found_stocks": [
        {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "confidence": 0.9},
        {"ticker": "AMD", "company_name": "Advanced Micro Devices", "confidence": 0.85}
    ],
    "confidence_score": 0.9,
    "analysis": {
        "understanding": "Query focuses on AI chip manufacturers",
        "strategy": "Direct company matching",
        "confidence": "High confidence from exact matches"
    }
}

MOCK_MARKET_DATA_RESPONSE = {
    "market_data": {
        "NVDA": {
            "historical_prices": [100.0, 102.0, 105.0, 103.0, 106.0],
            "dates": ["2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-09"],
            "volumes": [1000000, 1200000, 980000, 1100000, 1300000],
            "info": {
                "current_price": 106.0,
                "price_change": 6.0,
                "price_change_percent": 6.0,
                "avg_volume": 1116000,
                "min_price": 100.0,
                "max_price": 106.0,
                "avg_price": 103.2
            }
        }
    }
}

MOCK_ANALYSIS_RESPONSE = {
    "analysis_results": [
        {
            "ticker": "NVDA",
            "score": 8.5,
            "key_metrics": {
                "avg_return": 1.5,
                "volatility": 0.8,
                "total_return": 6.0
            }
        }
    ],
    "best_stock": {
        "ticker": "NVDA",
        "reason": "Strong consistent growth and low volatility"
    }
}

@pytest.fixture
def mock_openai_response():
    class MockResponse:
        def __init__(self, content):
            self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': json.dumps(content)})()})]

    return MockResponse

@pytest.fixture
def mock_agents():
    with patch('agents.market_research_agent.MarketResearchAgent') as mock_research, \
         patch('agents.market_data_agent.MarketDataAgent') as mock_data, \
         patch('agents.analysis_agent.AnalysisAgent') as mock_analysis:
        
        mock_research.return_value.execute = AsyncMock(return_value=MOCK_MARKET_RESEARCH_RESPONSE)
        mock_data.return_value.execute = AsyncMock(return_value=MOCK_MARKET_DATA_RESPONSE)
        mock_analysis.return_value.execute = AsyncMock(return_value=MOCK_ANALYSIS_RESPONSE)
        
        yield {
            'research': mock_research,
            'data': mock_data,
            'analysis': mock_analysis
        }

@pytest.mark.asyncio
async def test_successful_stock_analysis(mock_openai_response, mock_agents):
    """Test successful stock analysis with valid query"""
    
    # Mock OpenAI responses
    mock_classification = {"is_stock_related": "yes", "confidence": 9.0}
    mock_task_breakdown = {
        "agents": [
            {"name": "market_research", "is_required": True, "task_step": 1},
            {"name": "market_data", "is_required": True, "task_step": 2},
            {"name": "analysis", "is_required": True, "task_step": 3}
        ]
    }
    mock_final_response = {
        "response": "NVIDIA shows strong performance in the AI chip sector",
        "confidence_score": 0.9
    }

    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=[
            mock_openai_response(mock_classification),
            mock_openai_response(mock_task_breakdown),
            mock_openai_response(mock_final_response)
        ])

        response = client.post(
            "/api/v0/stock/analyze",
            json={
                "instruction": "Find the best performing AI chip manufacturers",
                "parameters": {}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] == 0.9
        assert "NVIDIA" in data["response"]
        assert len(data["results"]) == 3
        assert "market_research" in data["results"]
        assert "market_data" in data["results"]
        assert "analysis" in data["results"]

@pytest.mark.asyncio
async def test_non_stock_related_query(mock_openai_response):
    """Test rejection of non-stock related query"""
    
    mock_classification = {"is_stock_related": "no", "confidence": 3.0}
    
    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(
            return_value=mock_openai_response(mock_classification)
        )

        response = client.post(
            "/api/v0/stock/analyze",
            json={
                "instruction": "What's the weather like today?",
                "parameters": {}
            }
        )

        assert response.status_code == 400
        assert "not sufficiently related to stock analysis" in response.json()["detail"]

@pytest.mark.asyncio
async def test_market_research_failure(mock_openai_response, mock_agents):
    """Test handling of market research agent failure"""
    
    # Mock success for classification and task breakdown
    mock_classification = {"is_stock_related": "yes", "confidence": 9.0}
    mock_task_breakdown = {
        "agents": [
            {"name": "market_research", "is_required": True, "task_step": 1}
        ]
    }

    # Make market research agent raise an exception
    mock_agents['research'].return_value.execute = AsyncMock(side_effect=Exception("API rate limit exceeded"))

    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=[
            mock_openai_response(mock_classification),
            mock_openai_response(mock_task_breakdown)
        ])

        response = client.post(
            "/api/v0/stock/analyze",
            json={
                "instruction": "Find tech stocks",
                "parameters": {}
            }
        )

        assert response.status_code == 500
        assert "Error in stock analysis" in response.json()["detail"]

@pytest.mark.asyncio
async def test_empty_instruction(mock_openai_response):
    """Test handling of empty instruction"""
    
    response = client.post(
        "/api/v0/stock/analyze",
        json={
            "instruction": "",
            "parameters": {}
        }
    )

    assert response.status_code == 422  # FastAPI validation error

@pytest.mark.asyncio
async def test_complex_query_workflow(mock_openai_response, mock_agents):
    """Test complete workflow with complex query and parameters"""
    
    mock_classification = {"is_stock_related": "yes", "confidence": 9.5}
    mock_task_breakdown = {
        "agents": [
            {"name": "market_research", "is_required": True, "task_step": 1},
            {"name": "market_data", "is_required": True, "task_step": 2},
            {"name": "analysis", "is_required": True, "task_step": 3}
        ]
    }
    mock_final_response = {
        "response": "Detailed analysis of AI chip manufacturers...",
        "confidence_score": 0.95
    }

    with patch('openai.AsyncOpenAI') as mock_openai:
        mock_openai.return_value.chat.completions.create = AsyncMock(side_effect=[
            mock_openai_response(mock_classification),
            mock_openai_response(mock_task_breakdown),
            mock_openai_response(mock_final_response)
        ])

        response = client.post(
            "/api/v0/stock/analyze",
            json={
                "instruction": "Analyze AI chip manufacturers with focus on revenue growth and market share",
                "parameters": {
                    "metrics": ["revenue_growth", "market_share"],
                    "timeframe": "1y",
                    "min_market_cap": 1000000000
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        
        # Verify workflow execution
        assert data["meta_information"]["query_classification"]["confidence"] == 9.5
        assert len(data["meta_information"]["execution_path"]) == 3
        assert "market_research" in data["results"]
        assert "market_data" in data["results"]
        assert "analysis" in data["results"]

if __name__ == "__main__":
    pytest.main([__file__])