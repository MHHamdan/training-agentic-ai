"""
LangGraph Workflow for Comprehensive AI Assistant
Implements observable, traceable workflow for multi-service information gathering
Author: Mohammed Hamdan
"""

import asyncio
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import logging
import time
from datetime import datetime

# LangGraph and LangSmith imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback for when LangGraph is not available
    def traceable(func):
        return func

from config.settings import API_CONFIGS, WORKFLOW_CONFIG, SERVICE_CATEGORIES
from services.multi_api_service import MultiAPIService
from utils.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """State maintained throughout the workflow"""
    user_query: str
    user_location: Optional[str]
    user_preferences: Dict[str, Any]
    intent_analysis: Dict[str, Any]
    required_services: List[str]
    api_responses: Dict[str, Any]
    aggregated_data: Dict[str, Any]
    formatted_response: str
    workflow_metadata: Dict[str, Any]
    errors: List[str]
    processing_time: float
    confidence_score: float

@dataclass
class WorkflowStep:
    """Represents a step in the workflow"""
    name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    details: Dict[str, Any] = None
    error: Optional[str] = None

class ComprehensiveAIWorkflow:
    """
    Main workflow orchestrator for the comprehensive AI assistant
    Implements observable LangGraph workflow with full traceability
    """
    
    def __init__(self):
        """Initialize the workflow with services and graph"""
        self.multi_api_service = MultiAPIService()
        self.response_formatter = ResponseFormatter()
        self.workflow_steps: List[WorkflowStep] = []
        self.graph = self._build_graph()
        
        logger.info("🚀 Comprehensive AI Workflow initialized")
    
    def _build_graph(self) -> Optional[Any]:
        """Build the LangGraph workflow"""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("⚠️ LangGraph not available - using sequential processing")
            return None
        
        try:
            # Create the workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add nodes for each workflow step
            workflow.add_node("intent_analysis", self.analyze_intent)
            workflow.add_node("service_selection", self.select_services)
            workflow.add_node("data_collection", self.collect_data)
            workflow.add_node("data_aggregation", self.aggregate_data)
            workflow.add_node("response_formatting", self.format_response)
            workflow.add_node("quality_assessment", self.assess_quality)
            
            # Define the workflow edges
            workflow.set_entry_point("intent_analysis")
            workflow.add_edge("intent_analysis", "service_selection")
            workflow.add_edge("service_selection", "data_collection")
            workflow.add_edge("data_collection", "data_aggregation")
            workflow.add_edge("data_aggregation", "response_formatting")
            workflow.add_edge("response_formatting", "quality_assessment")
            workflow.add_edge("quality_assessment", END)
            
            return workflow.compile()
            
        except Exception as e:
            logger.error(f"❌ Failed to build LangGraph: {e}")
            return None
    
    def _add_step(self, name: str, status: str = "pending", details: Dict[str, Any] = None):
        """Add a workflow step for tracking"""
        step = WorkflowStep(
            name=name,
            status=status,
            start_time=time.time() if status == "running" else None,
            details=details or {}
        )
        self.workflow_steps.append(step)
        return step
    
    def _update_step(self, step: WorkflowStep, status: str, error: str = None, details: Dict[str, Any] = None):
        """Update a workflow step"""
        step.status = status
        step.end_time = time.time()
        if error:
            step.error = error
        if details:
            step.details.update(details)
    
    @traceable
    async def analyze_intent(self, state: WorkflowState) -> WorkflowState:
        """
        Analyze user intent and extract key information
        """
        step = self._add_step("Intent Analysis", "running")
        
        try:
            user_query = state["user_query"].lower()
            
            # Analyze intent using keyword matching and patterns
            intent_analysis = {
                "primary_intent": "information_request",
                "categories": [],
                "entities": {},
                "urgency": "normal",
                "location_required": False
            }
            
            # Check for different service categories
            if any(word in user_query for word in ["weather", "temperature", "rain", "forecast"]):
                intent_analysis["categories"].append("weather")
                intent_analysis["location_required"] = True
                
            if any(word in user_query for word in ["news", "headlines", "breaking", "latest"]):
                intent_analysis["categories"].append("news")
                
            if any(word in user_query for word in ["restaurant", "food", "eat", "dining"]):
                intent_analysis["categories"].append("places")
                intent_analysis["location_required"] = True
                
            if any(word in user_query for word in ["stock", "price", "market", "crypto", "bitcoin"]):
                intent_analysis["categories"].append("finance")
                
            if any(word in user_query for word in ["movie", "tv", "entertainment", "show"]):
                intent_analysis["categories"].append("entertainment")
                
            if any(word in user_query for word in ["recipe", "nutrition", "calories", "healthy"]):
                intent_analysis["categories"].append("health")
                
            if any(word in user_query for word in ["sports", "game", "score", "team"]):
                intent_analysis["categories"].append("sports")
                
            if any(word in user_query for word in ["buy", "price", "shop", "deal", "store"]):
                intent_analysis["categories"].append("shopping")
                
            # If no specific category found, default to news and weather
            if not intent_analysis["categories"]:
                intent_analysis["categories"] = ["news", "weather"]
                intent_analysis["location_required"] = True
            
            state["intent_analysis"] = intent_analysis
            
            self._update_step(step, "completed", details={
                "categories_identified": len(intent_analysis["categories"]),
                "location_required": intent_analysis["location_required"]
            })
            
            logger.info(f"✅ Intent analyzed: {intent_analysis['categories']}")
            
        except Exception as e:
            error_msg = f"Intent analysis failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    @traceable
    async def select_services(self, state: WorkflowState) -> WorkflowState:
        """
        Select appropriate services based on intent analysis
        """
        step = self._add_step("Service Selection", "running")
        
        try:
            intent_analysis = state["intent_analysis"]
            required_services = []
            
            # Map categories to services
            for category in intent_analysis["categories"]:
                services = SERVICE_CATEGORIES.get(category, [])
                required_services.extend(services)
            
            # Remove duplicates and prioritize
            required_services = list(set(required_services))
            
            # Limit concurrent services based on configuration
            max_services = WORKFLOW_CONFIG["max_concurrent_apis"]
            if len(required_services) > max_services:
                required_services = required_services[:max_services]
            
            state["required_services"] = required_services
            
            self._update_step(step, "completed", details={
                "services_selected": len(required_services),
                "service_list": required_services
            })
            
            logger.info(f"✅ Selected {len(required_services)} services: {required_services}")
            
        except Exception as e:
            error_msg = f"Service selection failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    @traceable
    async def collect_data(self, state: WorkflowState) -> WorkflowState:
        """
        Collect data from selected services concurrently
        """
        step = self._add_step("Data Collection", "running")
        
        try:
            required_services = state["required_services"]
            user_query = state["user_query"]
            user_location = state.get("user_location", "San Francisco, CA")
            
            # Collect data from all services concurrently
            api_responses = await self.multi_api_service.fetch_from_multiple_services(
                services=required_services,
                query=user_query,
                location=user_location
            )
            
            state["api_responses"] = api_responses
            
            # Count successful responses
            successful_responses = sum(1 for response in api_responses.values() 
                                    if response.get("success", False))
            
            self._update_step(step, "completed", details={
                "total_services": len(required_services),
                "successful_responses": successful_responses,
                "success_rate": successful_responses / len(required_services) if required_services else 0
            })
            
            logger.info(f"✅ Collected data from {successful_responses}/{len(required_services)} services")
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    @traceable
    async def aggregate_data(self, state: WorkflowState) -> WorkflowState:
        """
        Aggregate and process collected data
        """
        step = self._add_step("Data Aggregation", "running")
        
        try:
            api_responses = state["api_responses"]
            intent_analysis = state["intent_analysis"]
            
            aggregated_data = {
                "news": [],
                "weather": {},
                "places": [],
                "finance": {},
                "entertainment": [],
                "health": [],
                "sports": [],
                "shopping": [],
                "summary": {}
            }
            
            # Process each API response
            total_items = 0
            for service_name, response in api_responses.items():
                if not response.get("success", False):
                    continue
                
                data = response.get("data", {})
                
                # Categorize data based on service type
                if service_name in ["newsapi", "duckduckgo"]:
                    news_items = self._extract_news_items(data)
                    aggregated_data["news"].extend(news_items)
                    total_items += len(news_items)
                    
                elif service_name in ["openweather", "weatherapi"]:
                    weather_data = self._extract_weather_data(data)
                    aggregated_data["weather"].update(weather_data)
                    total_items += 1
                    
                elif service_name in ["foursquare", "yelp"]:
                    places_data = self._extract_places_data(data)
                    aggregated_data["places"].extend(places_data)
                    total_items += len(places_data)
                    
                elif service_name in ["alphavantage", "coingecko"]:
                    finance_data = self._extract_finance_data(data)
                    aggregated_data["finance"].update(finance_data)
                    total_items += 1
                    
                elif service_name == "tmdb":
                    entertainment_data = self._extract_entertainment_data(data)
                    aggregated_data["entertainment"].extend(entertainment_data)
                    total_items += len(entertainment_data)
            
            state["aggregated_data"] = aggregated_data
            
            self._update_step(step, "completed", details={
                "total_data_items": total_items,
                "categories_populated": len([k for k, v in aggregated_data.items() if v])
            })
            
            logger.info(f"✅ Aggregated {total_items} data items across multiple categories")
            
        except Exception as e:
            error_msg = f"Data aggregation failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    @traceable
    async def format_response(self, state: WorkflowState) -> WorkflowState:
        """
        Format the final response for the user
        """
        step = self._add_step("Response Formatting", "running")
        
        try:
            aggregated_data = state["aggregated_data"]
            intent_analysis = state["intent_analysis"]
            user_query = state["user_query"]
            
            # Format response using the response formatter
            formatted_response = await self.response_formatter.format_comprehensive_response(
                query=user_query,
                data=aggregated_data,
                intent=intent_analysis
            )
            
            state["formatted_response"] = formatted_response
            
            self._update_step(step, "completed", details={
                "response_length": len(formatted_response),
                "categories_included": len([k for k, v in aggregated_data.items() if v])
            })
            
            logger.info("✅ Response formatted successfully")
            
        except Exception as e:
            error_msg = f"Response formatting failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    @traceable
    async def assess_quality(self, state: WorkflowState) -> WorkflowState:
        """
        Assess the quality and completeness of the response
        """
        step = self._add_step("Quality Assessment", "running")
        
        try:
            aggregated_data = state["aggregated_data"]
            api_responses = state["api_responses"]
            required_services = state["required_services"]
            
            # Calculate metrics
            successful_apis = sum(1 for response in api_responses.values() 
                                if response.get("success", False))
            total_apis = len(required_services)
            success_rate = successful_apis / total_apis if total_apis > 0 else 0
            
            data_completeness = len([k for k, v in aggregated_data.items() if v]) / len(aggregated_data)
            
            # Calculate confidence score
            confidence_score = (success_rate * 0.6) + (data_completeness * 0.4)
            
            state["confidence_score"] = confidence_score
            state["workflow_metadata"] = {
                "successful_apis": successful_apis,
                "total_apis": total_apis,
                "success_rate": success_rate,
                "data_completeness": data_completeness,
                "workflow_steps": len(self.workflow_steps),
                "total_processing_time": time.time() - (self.workflow_steps[0].start_time if self.workflow_steps else time.time())
            }
            
            self._update_step(step, "completed", details={
                "confidence_score": confidence_score,
                "success_rate": success_rate,
                "data_completeness": data_completeness
            })
            
            logger.info(f"✅ Quality assessment completed - Confidence: {confidence_score:.2%}")
            
        except Exception as e:
            error_msg = f"Quality assessment failed: {str(e)}"
            self._update_step(step, "failed", error=error_msg)
            state["errors"].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return state
    
    # Helper methods for data extraction
    def _extract_news_items(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract news items from API response"""
        try:
            if "articles" in data:
                return data["articles"][:5]  # Limit to top 5
            return []
        except Exception:
            return []
    
    def _extract_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract weather data from API response"""
        try:
            return {
                "temperature": data.get("main", {}).get("temp"),
                "description": data.get("weather", [{}])[0].get("description"),
                "humidity": data.get("main", {}).get("humidity"),
                "location": data.get("name", "Unknown")
            }
        except Exception:
            return {}
    
    def _extract_places_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract places data from API response"""
        try:
            if "businesses" in data:
                return data["businesses"][:5]
            elif "results" in data:
                return data["results"][:5]
            return []
        except Exception:
            return []
    
    def _extract_finance_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial data from API response"""
        try:
            return {
                "stocks": data.get("Global Quote", {}),
                "crypto": data.get("prices", {}),
                "market_status": "open" if data else "closed"
            }
        except Exception:
            return {}
    
    def _extract_entertainment_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entertainment data from API response"""
        try:
            if "results" in data:
                return data["results"][:5]
            return []
        except Exception:
            return []
    
    async def process_request(self, user_query: str, user_location: str = None, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user requests
        """
        start_time = time.time()
        self.workflow_steps = []  # Reset steps for new request
        
        try:
            # Initialize state
            initial_state = WorkflowState(
                user_query=user_query,
                user_location=user_location,
                user_preferences=user_preferences or {},
                intent_analysis={},
                required_services=[],
                api_responses={},
                aggregated_data={},
                formatted_response="",
                workflow_metadata={},
                errors=[],
                processing_time=0.0,
                confidence_score=0.0
            )
            
            # Execute workflow
            if self.graph and LANGGRAPH_AVAILABLE:
                logger.info("🔄 Executing LangGraph workflow...")
                final_state = await self.graph.ainvoke(initial_state)
            else:
                logger.info("🔄 Executing sequential workflow...")
                final_state = await self._execute_sequential_workflow(initial_state)
            
            # Calculate final processing time
            final_state["processing_time"] = time.time() - start_time
            
            return {
                "success": True,
                "response": final_state["formatted_response"],
                "confidence": final_state["confidence_score"],
                "metadata": final_state["workflow_metadata"],
                "workflow_steps": [
                    {
                        "name": step.name,
                        "status": step.status,
                        "duration": (step.end_time - step.start_time) if step.end_time and step.start_time else 0,
                        "details": step.details,
                        "error": step.error
                    }
                    for step in self.workflow_steps
                ],
                "errors": final_state["errors"],
                "processing_time": final_state["processing_time"]
            }
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "success": False,
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "confidence": 0.0,
                "metadata": {},
                "workflow_steps": [],
                "errors": [str(e)],
                "processing_time": time.time() - start_time
            }
    
    async def _execute_sequential_workflow(self, state: WorkflowState) -> WorkflowState:
        """Execute workflow sequentially when LangGraph is not available"""
        state = await self.analyze_intent(state)
        state = await self.select_services(state)
        state = await self.collect_data(state)
        state = await self.aggregate_data(state)
        state = await self.format_response(state)
        state = await self.assess_quality(state)
        return state
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get workflow visualization data"""
        return {
            "nodes": [
                {"id": "intent_analysis", "label": "Intent Analysis", "type": "analysis"},
                {"id": "service_selection", "label": "Service Selection", "type": "routing"},
                {"id": "data_collection", "label": "Data Collection", "type": "api"},
                {"id": "data_aggregation", "label": "Data Aggregation", "type": "processing"},
                {"id": "response_formatting", "label": "Response Formatting", "type": "formatting"},
                {"id": "quality_assessment", "label": "Quality Assessment", "type": "validation"}
            ],
            "edges": [
                {"from": "intent_analysis", "to": "service_selection"},
                {"from": "service_selection", "to": "data_collection"},
                {"from": "data_collection", "to": "data_aggregation"},
                {"from": "data_aggregation", "to": "response_formatting"},
                {"from": "response_formatting", "to": "quality_assessment"}
            ],
            "current_steps": self.workflow_steps
        }