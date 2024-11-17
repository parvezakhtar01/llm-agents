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

from app.prompts import query_classification_prompt, instruction_parsing_prompt, final_response_prompt

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
        self.logger = logging.getLogger(__name__)
        self.query_classification_prompt = query_classification_prompt.prompt
        self.parse_instruction_prompt = instruction_parsing_prompt.prompt
        self.final_response_prompt = final_response_prompt.prompt

        # Function schemas for different operations
        self.schemas = {
            "query_classification": {
                "name": "classify_query",
                "description": "Classifies if a query is related to stock analysis.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_stock_related": {
                            "type": "boolean",
                            "description": "Indicates if the query is related to stock analysis."
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence score from 0 to 10.",
                            "minimum": 0,
                            "maximum": 10
                        }
                    },
                    "required": ["is_stock_related", "confidence"]
                }
            },
            "instruction_parsing": {
                "name": "parse_instruction",
                "description": "Breaks down stock analysis instructions into tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agents": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "enum": ["market_research", "market_data", "analysis"]
                                    },
                                    "is_required": {"type": "boolean"},
                                    "task_step": {"type": "integer"},
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "search_criteria": {"type": "string"},
                                            "timeframe_days": {"type": "integer"},
                                            "metrics": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            }
                                        }
                                    }
                                },
                                "required": ["name", "is_required", "task_step"]
                            }
                        },
                        "execution_notes": {"type": "string"}
                    },
                    "required": ["agents"]
                }
            },
            "final_response": {
                "name": "generate_response",
                "description": "Generates final analysis response from results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "Detailed analysis summary"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "Overall confidence in the analysis",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "stock": {"type": "string"},
                                    "insight": {"type": "string"},
                                    "metrics": {"type": "object"}
                                }
                            }
                        }
                    },
                    "required": ["response", "confidence_score", "key_findings"]
                }
            }
        }

    async def _execute_llm_call(
            self,
            messages: List[Dict[str, str]],
            error_message: str,
            operation_type: str,
            temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Enhanced LLM call with operation-specific schema and error handling"""
        try:
            function_schema = self.schemas.get(operation_type)

            request_params = {
                "model": "gpt-4-1106-preview",
                "messages": messages,
                "temperature": temperature
            }

            if function_schema:
                request_params["functions"] = [function_schema]
                request_params["function_call"] = {"name": function_schema["name"]}
            else:
                request_params["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**request_params)

            if function_schema and response.choices[0].message.function_call:
                return json.loads(response.choices[0].message.function_call.arguments)
            return json.loads(response.choices[0].message.content)

        except json.JSONDecodeError as je:
            self.logger.error(f"JSON parsing error: {str(je)}")
            raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(je)}")
        except Exception as e:
            self.logger.error(f"{error_message}: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=error_message)

    async def classify_query(self, instruction: str) -> Dict[str, Any]:
        """Classify if the query is stock-related"""
        self.logger.info(f"Classifying query: {instruction}")
        messages = [
            {"role": "system", "content": self.query_classification_prompt},
            {"role": "user", "content": instruction}
        ]
        return await self._execute_llm_call(
            messages=messages,
            error_message="Error in query classification",
            operation_type="query_classification"
        )

    async def parse_instruction(self, instruction: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Parse instruction into structured tasks"""
        self.logger.info(f"Parsing instruction: {instruction}")
        messages = [
            {"role": "system", "content": self.parse_instruction_prompt},
            {"role": "user", "content": f"Instruction: {instruction}\nParameters: {json.dumps(parameters)}"}
        ]
        return await self._execute_llm_call(
            messages=messages,
            error_message="Error in instruction parsing",
            operation_type="instruction_parsing"
        )

    async def generate_final_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response based on all results"""
        self.logger.info("Generating final response")
        messages = [
            {"role": "system", "content": self.final_response_prompt},
            {"role": "user", "content": f"Please analyze these stock market results and provide a detailed summary: {json.dumps(results)}"}
        ]
        return await self._execute_llm_call(
            messages=messages,
            error_message="Error in response generation",
            operation_type="final_response"
        )


async def execute_agent(
        agent_name: str,
        agent: Any,
        task_config: Dict[str, Any],
        previous_results: Dict[str, Any],
        logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Execute an agent with enhanced error handling and parameter passing

    Args:
        agent_name: Name of the agent to execute
        agent: Agent instance
        task_config: Configuration and parameters for this task
        previous_results: Results from previously executed agents
        logger: Logger instance
    """
    try:
        # Prepare agent-specific input based on task configuration
        input_data = prepare_agent_input(agent_name, task_config, previous_results)
        if input_data is None:
            logger.warning(f"Could not prepare input for {agent_name}")
            return None

        logger.info(f"Executing {agent_name} with input: {input_data}")
        result = await agent.execute(**input_data)
        logger.info(f"{agent_name} execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error executing {agent_name}: {str(e)}", exc_info=True)
        return None


def prepare_agent_input(
        agent_name: str,
        task_config: Dict[str, Any],
        previous_results: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Prepare input for specific agents based on task configuration and previous results
    """
    parameters = task_config.get("parameters", {})

    if agent_name == "market_research":
        return {
            "query": parameters.get("search_criteria", ""),
        }

    elif agent_name == "market_data":
        if not previous_results.get("market_research"):
            return None

        stocks = previous_results["market_research"].get("found_stocks", [])
        return {
            "tickers": [stock["ticker"] for stock in stocks if "ticker" in stock],
            "days": parameters.get("timeframe_days", 90)
        }

    elif agent_name == "analysis":
        if not previous_results.get("market_data"):
            return None

        return {
            "market_data": previous_results["market_data"],
            "metrics": parameters.get("metrics", ["avg_return", "volatility", "total_return"])
        }

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
        logger.debug(f"Query classification result: {classification}")

        if not classification.get("is_stock_related") or classification.get("confidence", 0) < 7:
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
                openai_key=openai_client.api_key
            ),
            "market_data": MarketDataAgent(),
            "analysis": AnalysisAgent()
        }

        # Execute agents in sequence with dependency tracking
        results = {}
        execution_order = sorted(parsed_tasks["agents"], key=lambda x: x["task_step"])

        for task in execution_order:
            if not task.get("is_required", False):
                continue

            agent_name = task["name"]
            agent = agents.get(agent_name)

            if not agent:
                logger.warning(f"Agent {agent_name} not found, skipping")
                continue

            # Execute agent with task configuration
            agent_result = await execute_agent(
                agent_name=agent_name,
                agent=agent,
                task_config=task,
                previous_results=results,
                logger=logger
            )

            if agent_result:
                results[agent_name] = agent_result
                logger.info(f"Agent {agent_name} completed successfully")
            else:
                logger.warning(f"No results from {agent_name}, evaluating impact on workflow")
                if task.get("is_required", True):
                    raise HTTPException(
                        status_code=500,
                        detail=f"Required agent {agent_name} failed to produce results"
                    )

        # Generate final response
        final_response = await master_agent.generate_final_response(results)
        logger.info("Analysis completed successfully")

        return StockAnalysisResponse(
            response=final_response.get("response", ""),
            confidence_score=final_response.get("confidence_score", 0.0),
            meta_information={
                "query_classification": classification,
                "execution_path": parsed_tasks["agents"],
                "timestamp": str(datetime.now()),
                "execution_summary": {
                    "completed_tasks": list(results.keys()),
                    "task_count": len(parsed_tasks["agents"]),
                    "successful_tasks": len(results)
                }
            },
            results=results
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in stock analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error in stock analysis: {str(e)}"
        )
