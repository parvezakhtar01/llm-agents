# app/api/api_v0/endpoints/stock.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from datetime import datetime
from agents.market_research_agent import MarketResearchAgent
from agents.market_data_agent import MarketDataAgent
from agents.analysis_agent import AnalysisAgent
from openai import AsyncOpenAI

router = APIRouter(prefix='/stocks', tags=["Stocks"])
from openai import AsyncOpenAI

load_dotenv()

router = APIRouter(prefix='/stock', tags=["Stock"])

class StockAnalysisRequest(BaseModel):
    instruction: str
    parameters: Dict[str, Any] = {}

class StockAnalysisResponse(BaseModel):
    response: str
    confidence_score: float
    meta_information: Dict[str, Any]
    results: Dict[str, Any]

async def get_openai_client():
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client

class MasterAgent:
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        self.system_prompt = """You are a stock analysis assistant that helps classify and break down user queries about stocks.
        You should determine if queries are stock-related and help orchestrate the analysis process."""

    async def classify_query(self, instruction: str) -> Dict[str, Any]:
        """Classify if the query is stock-related using OpenAI"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Classify if this query is stock-related: {instruction}"}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in query classification: {str(e)}")

    async def parse_instruction(self, instruction: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the instruction to determine required agents and execution order"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Parse this instruction and determine required agents: {instruction}"}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in instruction parsing: {str(e)}")

    async def generate_final_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final response based on all agent results"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate final analysis from these results: {json.dumps(results)}"}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in response generation: {str(e)}")

@router.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(
    request: StockAnalysisRequest,
    openai_client: AsyncOpenAI = Depends(get_openai_client)
) -> Dict[str, Any]:
    """
    Analyze stocks based on user instruction following the exact workflow
    """
    try:
        # Initialize master agent and other agents
        master_agent = MasterAgent(openai_client)
        
        # Step 1: Query Classification (exact match with workflow)
        classification = await master_agent.classify_query(request.instruction)
        if classification.get("is_stock_related") != "yes" or classification.get("confidence", 0) < 7:
            raise HTTPException(
                status_code=400,
                detail="Query not sufficiently related to stock analysis"
            )

        # Step 2: Task Breakdown (exact match with workflow)
        parsed_tasks = await master_agent.parse_instruction(
            request.instruction,
            request.parameters
        )

        # Initialize agents based on task requirements
        agents = {
            "market_research": MarketResearchAgent(
                serpapi_key=os.getenv("SERPAPI_API_KEY"),
                openai_key=os.getenv("OPENAI_API_KEY")
            ),
            "market_data": MarketDataAgent(),
            "analysis": AnalysisAgent()
        }

        # Step 3: Agent Execution (exact match with workflow)
        results = {}
        for task in sorted(parsed_tasks["agents"], key=lambda x: x["task_step"]):
            if task["is_required"]:
                agent_name = task["name"]
                agent = agents.get(agent_name)
                if not agent:
                    continue
                    
                # Determine input for each agent based on previous results
                agent_input = None
                if agent_name == "market_research":
                    agent_input = request.instruction
                elif agent_name == "market_data":
                    # Get tickers from market research results
                    stocks = results.get("market_research", {}).get("found_stocks", [])
                    agent_input = [stock["ticker"] for stock in stocks if "ticker" in stock]
                elif agent_name == "analysis":
                    agent_input = results.get("market_data", {})
                
                # Execute agent with appropriate input
                results[agent_name] = await agent.execute(agent_input)

        # Step 4: Reanalyze and Prepare Response (exact match with workflow)
        final_response = await master_agent.generate_final_response(results)

        return StockAnalysisResponse(
            response=final_response.get("response", ""),
            confidence_score=final_response.get("confidence_score", 0.0),
            meta_information={
                "query_classification": classification,
                "execution_path": parsed_tasks["agents"],
                "timestamp": str(datetime.now())
            },
            results=results
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in stock analysis: {str(e)}"
        )