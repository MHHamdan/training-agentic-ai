"""
LangGraph Workflow Manager for Research Agent
Orchestrates multi-agent research workflow with Langfuse observability
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    print("Warning: LangGraph not installed. Using mock implementation.")
    StateGraph = object
    END = "END"
    MemorySaver = object

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    from langfuse.callback import CallbackHandler
except ImportError:
    print("Warning: Langfuse not installed. Observability will be limited.")
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None
    CallbackHandler = None
    Langfuse = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from graph.state import (
    ResearchState, ResearchPhase, create_initial_state,
    update_state_phase, add_error_to_state, calculate_overall_quality
)

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level))
logger = logging.getLogger(__name__)

class ResearchWorkflowManager:
    """
    Manages the LangGraph research workflow with Langfuse observability
    """
    
    def __init__(self):
        """Initialize workflow manager with Langfuse tracking"""
        self.config = config
        self.langfuse_client = None
        self.langfuse_handler = None
        self.workflow = None
        self.checkpointer = None
        
        # Initialize Langfuse if configured
        if self.config.langfuse.enabled and Langfuse:
            self._initialize_langfuse()
        
        # Build the workflow graph
        self._build_workflow()
    
    def _initialize_langfuse(self):
        """Initialize Langfuse client and handler"""
        try:
            self.langfuse_client = Langfuse(
                public_key=self.config.langfuse.public_key,
                secret_key=self.config.langfuse.secret_key,
                host=self.config.langfuse.host
            )
            
            if CallbackHandler:
                self.langfuse_handler = CallbackHandler(
                    public_key=self.config.langfuse.public_key,
                    secret_key=self.config.langfuse.secret_key,
                    host=self.config.langfuse.host
                )
            
            logger.info("Langfuse observability initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.config.langfuse.enabled = False
    
    def _build_workflow(self):
        """Build the LangGraph workflow"""
        try:
            # Create workflow graph
            workflow_builder = StateGraph(ResearchState)
            
            # Add nodes (agents)
            workflow_builder.add_node("search", self._search_node)
            workflow_builder.add_node("analyze", self._analyze_node)
            workflow_builder.add_node("synthesize", self._synthesize_node)
            workflow_builder.add_node("evaluate", self._evaluate_node)
            workflow_builder.add_node("fact_check", self._fact_check_node)
            
            # Define edges
            workflow_builder.set_entry_point("search")
            workflow_builder.add_edge("search", "analyze")
            workflow_builder.add_edge("analyze", "synthesize")
            workflow_builder.add_edge("synthesize", "fact_check")
            workflow_builder.add_edge("fact_check", "evaluate")
            workflow_builder.add_edge("evaluate", END)
            
            # Add conditional edges for error handling
            workflow_builder.add_conditional_edges(
                "search",
                self._check_search_results,
                {
                    "continue": "analyze",
                    "retry": "search",
                    "end": END
                }
            )
            
            # Compile workflow
            if MemorySaver and MemorySaver != object:
                self.checkpointer = MemorySaver()
                self.workflow = workflow_builder.compile(checkpointer=self.checkpointer)
            else:
                self.workflow = workflow_builder.compile()
            
            logger.info("Research workflow built successfully")
        except Exception as e:
            logger.error(f"Failed to build workflow: {e}")
            # Create a simple fallback workflow
            self.workflow = self._create_fallback_workflow()
    
    @observe(as_type="generation")
    async def _search_node(self, state: ResearchState) -> ResearchState:
        """Search agent node with Langfuse tracking"""
        try:
            state = update_state_phase(state, ResearchPhase.SEARCH)
            start_time = datetime.now()
            
            # Import search agent
            from agents.search_agent import SearchAgent
            search_agent = SearchAgent()
            
            # Execute search with tracking
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Search query: {state['query']}",
                    metadata={
                        "agent": "search_agent",
                        "phase": "search",
                        "research_id": state.get("research_id")
                    }
                )
            
            # Perform search
            search_results = await search_agent.search(
                query=state["query"],
                max_results=self.config.research.max_search_results
            )
            
            # Update state
            state["search_results"] = search_results
            state["search_metadata"] = {
                "total_results": len(search_results),
                "search_time": (datetime.now() - start_time).total_seconds(),
                "sources_used": search_agent.get_sources_used()
            }
            
            # Track processing time
            state["processing_time"]["search"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Search completed: {len(search_results)} results found")
            
            return state
            
        except Exception as e:
            logger.error(f"Search node error: {e}")
            return add_error_to_state(state, str(e), "search")
    
    @observe(as_type="generation")
    async def _analyze_node(self, state: ResearchState) -> ResearchState:
        """Analysis agent node with Langfuse tracking"""
        try:
            state = update_state_phase(state, ResearchPhase.ANALYSIS)
            start_time = datetime.now()
            
            # Import analyzer agent
            from agents.analyzer_agent import AnalyzerAgent
            analyzer = AnalyzerAgent()
            
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Analyzing {len(state['search_results'])} sources",
                    metadata={
                        "agent": "analyzer_agent",
                        "phase": "analysis",
                        "research_id": state.get("research_id"),
                        "source_count": len(state.get("search_results", []))
                    }
                )
            
            # Perform analysis
            analysis_results = await analyzer.analyze(
                search_results=state["search_results"],
                query=state["query"],
                depth=self.config.research.analysis_depth
            )
            
            # Update state
            state["analyzed_content"] = analysis_results["analyzed_content"]
            state["key_insights"] = analysis_results["key_insights"]
            state["extracted_facts"] = analysis_results["extracted_facts"]
            state["relevance_scores"] = analysis_results["relevance_scores"]
            
            # Track processing time
            state["processing_time"]["analysis"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Analysis completed: {len(state['key_insights'])} insights extracted")
            
            return state
            
        except Exception as e:
            logger.error(f"Analysis node error: {e}")
            return add_error_to_state(state, str(e), "analysis")
    
    @observe(as_type="generation")
    async def _synthesize_node(self, state: ResearchState) -> ResearchState:
        """Synthesis agent node with Langfuse tracking"""
        try:
            state = update_state_phase(state, ResearchPhase.SYNTHESIS)
            start_time = datetime.now()
            
            # Import synthesizer agent
            from agents.synthesizer_agent import SynthesizerAgent
            synthesizer = SynthesizerAgent()
            
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Synthesizing {len(state['analyzed_content'])} analyses",
                    metadata={
                        "agent": "synthesizer_agent",
                        "phase": "synthesis",
                        "research_id": state.get("research_id"),
                        "insight_count": len(state.get("key_insights", []))
                    }
                )
            
            # Perform synthesis
            synthesis_results = await synthesizer.synthesize(
                analyzed_content=state["analyzed_content"],
                key_insights=state["key_insights"],
                query=state["query"],
                max_length=self.config.performance.synthesis_max_length
            )
            
            # Update state
            state["synthesis"] = synthesis_results["synthesis"]
            state["executive_summary"] = synthesis_results["executive_summary"]
            state["detailed_findings"] = synthesis_results["detailed_findings"]
            state["recommendations"] = synthesis_results["recommendations"]
            state["citations"] = synthesis_results["citations"]
            state["bibliography"] = synthesis_results["bibliography"]
            
            # Track processing time
            state["processing_time"]["synthesis"] = (datetime.now() - start_time).total_seconds()
            
            logger.info("Synthesis completed successfully")
            
            return state
            
        except Exception as e:
            logger.error(f"Synthesis node error: {e}")
            return add_error_to_state(state, str(e), "synthesis")
    
    @observe(as_type="generation")
    async def _fact_check_node(self, state: ResearchState) -> ResearchState:
        """Fact checking node with Langfuse tracking"""
        try:
            start_time = datetime.now()
            
            # Import fact checker
            from tools.fact_checker import FactChecker
            fact_checker = FactChecker()
            
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Fact checking research claims",
                    metadata={
                        "agent": "fact_checker",
                        "phase": "fact_check",
                        "research_id": state.get("research_id")
                    }
                )
            
            # Perform fact checking
            fact_check_results = await fact_checker.verify_facts(
                claims=state.get("extracted_facts", []),
                sources=state.get("search_results", []),
                threshold=self.config.research.fact_check_threshold
            )
            
            # Update state
            state["fact_check_results"] = fact_check_results["results"]
            state["verified_claims"] = fact_check_results["verified"]
            state["disputed_claims"] = fact_check_results["disputed"]
            
            # Track processing time
            state["processing_time"]["fact_check"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Fact checking completed: {len(state['verified_claims'])} verified")
            
            return state
            
        except Exception as e:
            logger.error(f"Fact check node error: {e}")
            return add_error_to_state(state, str(e), "fact_check")
    
    @observe(as_type="evaluation")
    async def _evaluate_node(self, state: ResearchState) -> ResearchState:
        """Evaluation agent node with Langfuse tracking"""
        try:
            state = update_state_phase(state, ResearchPhase.EVALUATION)
            start_time = datetime.now()
            
            # Import evaluator agent
            from agents.evaluator_agent import EvaluatorAgent
            evaluator = EvaluatorAgent()
            
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Evaluating research quality",
                    metadata={
                        "agent": "evaluator_agent",
                        "phase": "evaluation",
                        "research_id": state.get("research_id")
                    }
                )
            
            # Perform evaluation
            evaluation_results = await evaluator.evaluate(
                state=state,
                evaluation_depth=self.config.performance.evaluation_depth
            )
            
            # Update state with evaluation metrics
            state["evaluation"] = evaluation_results
            state["quality_score"] = evaluation_results.get("overall_quality")
            state["accuracy_score"] = evaluation_results.get("accuracy")
            state["completeness_score"] = evaluation_results.get("completeness")
            state["bias_score"] = evaluation_results.get("bias_detection")
            state["reliability_score"] = evaluation_results.get("source_reliability")
            
            # Check academic compliance
            state["academic_standards_met"] = evaluation_results.get("academic_compliance", False)
            state["citation_compliance"] = evaluation_results.get("citation_quality", 0) > 0.8
            
            # Track processing time
            state["processing_time"]["evaluation"] = (datetime.now() - start_time).total_seconds()
            
            # Mark as complete
            state = update_state_phase(state, ResearchPhase.COMPLETE)
            
            # Log evaluation score to Langfuse
            if self.langfuse_client:
                self.langfuse_client.score(
                    trace_id=state.get("trace_id"),
                    name="research_quality",
                    value=state["quality_score"],
                    comment=f"Research completed for query: {state['query']}"
                )
            
            logger.info(f"Evaluation completed. Quality score: {state['quality_score']:.2f}")
            
            return state
            
        except Exception as e:
            logger.error(f"Evaluation node error: {e}")
            return add_error_to_state(state, str(e), "evaluation")
    
    def _check_search_results(self, state: ResearchState) -> str:
        """Conditional edge to check search results"""
        if not state.get("search_results"):
            if state.get("metadata", {}).get("retry_count", 0) < 3:
                return "retry"
            return "end"
        return "continue"
    
    def _create_fallback_workflow(self):
        """Create a simple fallback workflow if LangGraph is not available"""
        workflow_manager = self
        
        class FallbackWorkflow:
            async def invoke(self, state, config=None):
                # Simple sequential execution
                state = await workflow_manager._search_node(state)
                if state.get("search_results"):
                    state = await workflow_manager._analyze_node(state)
                    state = await workflow_manager._synthesize_node(state)
                    state = await workflow_manager._fact_check_node(state)
                    state = await workflow_manager._evaluate_node(state)
                return state
        
        return FallbackWorkflow()
    
    @observe(as_type="workflow")
    async def run_research(
        self,
        query: str,
        user_preferences: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ResearchState:
        """
        Execute the complete research workflow
        
        Args:
            query: Research query
            user_preferences: User preferences for research
            **kwargs: Additional parameters
        
        Returns:
            Completed ResearchState
        """
        try:
            # Create initial state
            state = create_initial_state(
                query=query,
                user_preferences=user_preferences or {},
                **kwargs
            )
            
            # Create Langfuse trace if enabled
            if self.langfuse_client:
                trace = self.langfuse_client.trace(
                    name="research_workflow",
                    input={"query": query},
                    metadata={
                        "project": self.config.langfuse.project,
                        "organization": self.config.langfuse.organization,
                        "environment": self.config.langfuse.environment
                    },
                    tags=self.config.langfuse.tags
                )
                state["trace_id"] = trace.id
            
            # Configure workflow execution
            workflow_config = {
                "recursion_limit": 10,
                "configurable": {
                    "thread_id": state["research_id"]
                }
            }
            
            # Add Langfuse callback if available
            if self.langfuse_handler:
                workflow_config["callbacks"] = [self.langfuse_handler]
            
            # Execute workflow using async API
            final_state = await self.workflow.ainvoke(state, workflow_config)
            
            # Calculate total processing time
            total_time = sum(final_state.get("processing_time", {}).values())
            final_state["metadata"]["total_processing_time"] = total_time
            
            # Log completion
            logger.info(f"Research completed in {total_time:.2f} seconds")
            
            return final_state
            
        except Exception as e:
            logger.error(f"Research workflow error: {e}")
            state = add_error_to_state(state, str(e), "workflow")
            state = update_state_phase(state, ResearchPhase.ERROR)
            return state
    
    async def get_workflow_status(self, research_id: str) -> Dict[str, Any]:
        """Get status of a research workflow"""
        if self.checkpointer:
            checkpoint = self.checkpointer.get({"configurable": {"thread_id": research_id}})
            if checkpoint:
                return {
                    "status": "in_progress",
                    "phase": checkpoint.get("phase"),
                    "progress": checkpoint.get("metadata", {}).get("progress", 0)
                }
        return {"status": "not_found"}

# Create global workflow manager instance
workflow_manager = ResearchWorkflowManager()