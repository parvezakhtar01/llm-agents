# app/api/api_v0/endpoints/stocks.py
import sys
import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
from datetime import datetime
from openai import AsyncOpenAI
import traceback

from app.prompts import query_cassification_prompt

# Import agents
from app.agents.market_research_agent import MarketResearchAgent
from app.agents.market_data_agent import MarketDataAgent
from app.agents.analysis_agent import AnalysisAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

router = APIRouter(prefix='/stocks', tags=["Stocks"])

# Define your function schema
classification_function = {
    "name": "classify_query",
    "description": "Classifies if a query is related to stock analysis and provides a confidence score.",
    "parameters": {
        "type": "object",
        "properties": {
            "is_stock_related": {
                "type": "boolean",
                "description": "Indicates if the query is related to stock analysis."
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score for the classification ranging from 0 to 10.",
                "minimum": 0,
                "maximum": 10
            }
        },
        "required": ["is_stock_related", "confidence"],
        "additionalProperties": False
    }
}


class StockAnalysisRequest(BaseModel):
    instruction: str = Field(..., description="User instruction for stock analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optional parameters for analysis")

class StockAnalysisResponse(BaseModel):
    response: str
    confidence_score: float
    meta_information: Dict[str, Any]
    results: Dict[str, Any]

async def get_openai_client() -> AsyncOpenAI:
    """Get OpenAI client with error handling"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        return AsyncOpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail="Error initializing AI services")

class MasterAgent:
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client
        self.system_prompt = query_cassification_prompt
        self.logger = logging.getLogger(__name__)

    async def _execute_llm_call(self, messages: List[Dict[str, str]], error_message: str) -> Dict[str, Any]:
        """Execute LLM call with error handling"""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            self.logger.error(f"{error_message}: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_message)

    async def classify_query(self, instruction: str) -> Dict[str, Any]:
        """Classify if the query is stock-related using OpenAI"""
        self.logger.info(f"Classifying query: {instruction}")
        return await self._execute_llm_call(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Classify if this query is stock-related: {instruction}"}
            ],
            error_message="Error in query classification"
        )

    async def parse_instruction(self, instruction: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the instruction to determine required agents and execution order"""
        self.logger.info(f"Parsing instruction: {instruction}")
        return await self._execute_llm_call(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Parse this instruction and determine required agents: {instruction}"}
            ],
            error_message="Error in instruction parsing"
        )

    async def generate_final_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final response based on all agent results"""
        self.logger.info("Generating final response")
        return await self._execute_llm_call(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Generate final analysis from these results: {json.dumps(results)}"}
            ],
            error_message="Error in response generation"
        )

async def execute_agent(
    agent_name: str,
    agent: Any,
    input_data: Any,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """Execute an agent with error handling"""
    try:
        logger.info(f"Executing {agent_name} with input: {input_data}")
        result = await agent.execute(input_data)
        logger.info(f"{agent_name} execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error executing {agent_name}: {str(e)}\n{traceback.format_exc()}")
        return None

@router.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(
    request: StockAnalysisRequest,
    openai_client: AsyncOpenAI = Depends(get_openai_client)
) -> Dict[str, Any]:
    """
    Analyze stocks based on user instruction with comprehensive error handling and logging
    """
    logger.info(f"Received analysis request: {request.instruction}")
    
    try:
        # Initialize master agent
        master_agent = MasterAgent(openai_client)
        
        # Query Classification
        classification = await master_agent.classify_query(request.instruction)
        logger.info(f"Query classification result: {classification}")
        
        if not classification.get("is_stock_related") == "yes" or classification.get("confidence", 0) < 7:
            logger.warning(f"Query rejected: not sufficiently stock-related. Classification: {classification}")
            raise HTTPException(
                status_code=400,
                detail="Query not sufficiently related to stock analysis"
            )

        # Task Breakdown
        parsed_tasks = await master_agent.parse_instruction(request.instruction, request.parameters)
        logger.info(f"Task breakdown: {parsed_tasks}")

        # Initialize agents
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            raise ValueError("SERPAPI API key not found in environment variables")

        agents = {
            "market_research": MarketResearchAgent(
                serpapi_key=serpapi_key,
                openai_key=os.getenv("OPENAI_API_KEY")
            ),
            "market_data": MarketDataAgent(),
            "analysis": AnalysisAgent()
        }

        # Execute agents in sequence
        results = {}
        for task in sorted(parsed_tasks["agents"], key=lambda x: x["task_step"]):
            if not task["is_required"]:
                continue

            agent_name = task["name"]
            agent = agents.get(agent_name)
            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue

            # Prepare input for each agent
            agent_input = None
            if agent_name == "market_research":
                agent_input = request.instruction
            elif agent_name == "market_data" and results.get("market_research"):
                stocks = results["market_research"].get("found_stocks", [])
                agent_input = [stock["ticker"] for stock in stocks if "ticker" in stock]
            elif agent_name == "analysis" and results.get("market_data"):
                agent_input = results["market_data"]
            
            if agent_input is None:
                logger.warning(f"No valid input for {agent_name}, skipping")
                continue

            # Execute agent
            agent_result = await execute_agent(agent_name, agent, agent_input, logger)
            if agent_result:
                results[agent_name] = agent_result
            else:
                logger.warning(f"No results from {agent_name}, continuing with available data")

        # Generate final response
        final_response = await master_agent.generate_final_response(results)
        logger.info("Analysis completed successfully")

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

    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in stock analysis: {str(e)}"
        )