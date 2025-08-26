"""
Langfuse Observability Integration for Research Agent
Enterprise-grade monitoring and evaluation framework
"""

import os
import logging
import time
import functools
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create mock decorators for development
    def observe(as_type: str = "generation"):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    class MockLangfuseContext:
        def update_current_observation(self, **kwargs):
            pass
        def update_current_trace(self, **kwargs):
            pass
    
    langfuse_context = MockLangfuseContext()

from config import config

logger = logging.getLogger(__name__)

class LangfuseManager:
    """Centralized Langfuse management for research agent"""
    
    def __init__(self):
        self.enabled = LANGFUSE_AVAILABLE and config.langfuse.enabled
        self.client = None
        self.callback_handler = None
        self.current_session_id = None
        
        if self.enabled:
            self._initialize_langfuse()
        else:
            logger.warning("Langfuse not available or disabled. Running in mock mode.")
    
    def _initialize_langfuse(self):
        """Initialize Langfuse client and callback handler"""
        try:
            # Initialize Langfuse client
            self.client = Langfuse(
                public_key=config.langfuse.public_key,
                secret_key=config.langfuse.secret_key,
                host=config.langfuse.host,
                debug=config.debug
            )
            
            # Initialize callback handler for LangChain integration
            self.callback_handler = CallbackHandler(
                secret_key=config.langfuse.secret_key,
                public_key=config.langfuse.public_key,
                host=config.langfuse.host
            )
            
            logger.info("Langfuse observability initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {str(e)}")
            self.enabled = False
    
    def start_research_session(self, query: str, user_id: str = None) -> str:
        """Start a new research session with full tracking"""
        if not self.enabled:
            return "mock_session_id"
        
        try:
            # Create new trace for research session
            trace = self.client.trace(
                name="research_session",
                metadata={
                    "project": config.langfuse.project,
                    "organization": config.langfuse.organization,
                    "environment": config.langfuse.environment,
                    "agent_version": config.agent_version,
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                },
                tags=config.langfuse.tags + ["research_session"],
                user_id=user_id
            )
            
            self.current_session_id = trace.id
            logger.info(f"Started research session: {self.current_session_id}")
            
            return self.current_session_id
            
        except Exception as e:
            logger.error(f"Failed to start research session: {str(e)}")
            return "fallback_session_id"
    
    def end_research_session(self, session_id: str, results: Dict[str, Any]):
        """End research session with results tracking"""
        if not self.enabled:
            return
        
        try:
            if session_id and session_id != "mock_session_id":
                # Update trace with final results
                self.client.trace(
                    id=session_id,
                    output=results,
                    metadata={
                        "completion_timestamp": datetime.now().isoformat(),
                        "research_quality_score": results.get("quality_score"),
                        "citations_count": len(results.get("citations", [])),
                        "sources_used": len(results.get("search_results", []))
                    }
                )
                
            logger.info(f"Ended research session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to end research session: {str(e)}")
    
    def log_research_evaluation(self, session_id: str, evaluation_results: Dict[str, Any]):
        """Log research quality evaluation results"""
        if not self.enabled:
            return
        
        try:
            self.client.score(
                trace_id=session_id,
                name="research_quality",
                value=evaluation_results.get("overall_score", 0),
                comment=f"Research evaluation: {evaluation_results}",
                metadata={
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "evaluation_metrics": evaluation_results
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log research evaluation: {str(e)}")
    
    def log_model_performance(self, model_name: str, task_type: str, 
                            performance_metrics: Dict[str, Any]):
        """Log model performance for comparison"""
        if not self.enabled:
            return
        
        try:
            self.client.generation(
                name=f"model_performance_{model_name}",
                model=model_name,
                metadata={
                    "task_type": task_type,
                    "performance_metrics": performance_metrics,
                    "timestamp": datetime.now().isoformat()
                },
                tags=["model_performance", task_type]
            )
            
        except Exception as e:
            logger.error(f"Failed to log model performance: {str(e)}")
    
    def get_callback_handler(self):
        """Get Langfuse callback handler for LangChain integration"""
        return self.callback_handler if self.enabled else None

# Global Langfuse manager instance
langfuse_manager = LangfuseManager()

def track_research_operation(operation_type: str, metadata: Dict[str, Any] = None):
    """Decorator to track research operations with Langfuse"""
    def decorator(func):
        @functools.wraps(func)
        @observe(as_type=operation_type)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation_metadata = metadata or {}
            
            try:
                # Update Langfuse context with operation details
                if langfuse_manager.enabled:
                    langfuse_context.update_current_observation(
                        input=f"{func.__name__} called with args: {len(args)} kwargs: {len(kwargs)}",
                        metadata={
                            "operation_type": operation_type,
                            "function_name": func.__name__,
                            "agent_version": config.agent_version,
                            "timestamp": datetime.now().isoformat(),
                            **operation_metadata
                        },
                        tags=["research_operation", operation_type]
                    )
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Update context with results
                execution_time = time.time() - start_time
                if langfuse_manager.enabled:
                    langfuse_context.update_current_observation(
                        output=result,
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": True,
                            "result_type": type(result).__name__
                        }
                    )
                
                return result
                
            except Exception as e:
                # Log errors to Langfuse
                execution_time = time.time() - start_time
                if langfuse_manager.enabled:
                    langfuse_context.update_current_observation(
                        metadata={
                            "execution_time_seconds": execution_time,
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                raise
        
        return wrapper
    return decorator

class ResearchMetricsTracker:
    """Track and analyze research performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.langfuse_client = langfuse_manager.client
        
    @observe(as_type="metrics_tracking")
    def track_search_metrics(self, query: str, results_count: int, 
                           search_time: float, source_quality: float):
        """Track search operation metrics"""
        metrics = {
            "query_complexity": len(query.split()),
            "results_count": results_count,
            "search_time_seconds": search_time,
            "average_source_quality": source_quality,
            "timestamp": datetime.now().isoformat()
        }
        
        if langfuse_manager.enabled:
            langfuse_context.update_current_observation(
                input=f"Search query: {query}",
                output=f"Found {results_count} results",
                metadata=metrics
            )
        
        return metrics
    
    @observe(as_type="metrics_tracking")
    def track_analysis_metrics(self, content_length: int, analysis_depth: str,
                             processing_time: float, insight_count: int):
        """Track analysis operation metrics"""
        metrics = {
            "content_length": content_length,
            "analysis_depth": analysis_depth,
            "processing_time_seconds": processing_time,
            "insights_generated": insight_count,
            "processing_rate": content_length / processing_time if processing_time > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if langfuse_manager.enabled:
            langfuse_context.update_current_observation(
                input=f"Analyzing {content_length} characters",
                output=f"Generated {insight_count} insights",
                metadata=metrics
            )
        
        return metrics
    
    @observe(as_type="metrics_tracking") 
    def track_synthesis_metrics(self, source_count: int, synthesis_length: int,
                              synthesis_time: float, citation_count: int):
        """Track synthesis operation metrics"""
        metrics = {
            "sources_synthesized": source_count,
            "synthesis_length": synthesis_length,
            "synthesis_time_seconds": synthesis_time,
            "citations_generated": citation_count,
            "synthesis_efficiency": synthesis_length / synthesis_time if synthesis_time > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        if langfuse_manager.enabled:
            langfuse_context.update_current_observation(
                input=f"Synthesizing {source_count} sources",
                output=f"Created {synthesis_length} character synthesis",
                metadata=metrics
            )
        
        return metrics

class ResearchQualityEvaluator:
    """Evaluate and track research quality with Langfuse"""
    
    def __init__(self):
        self.langfuse_client = langfuse_manager.client
    
    @observe(as_type="quality_evaluation")
    def evaluate_research_quality(self, research_output: Dict[str, Any], 
                                original_query: str) -> Dict[str, float]:
        """Comprehensive research quality evaluation"""
        
        evaluation_results = {
            "relevance_score": self._calculate_relevance_score(research_output, original_query),
            "accuracy_score": self._calculate_accuracy_score(research_output),
            "completeness_score": self._calculate_completeness_score(research_output),
            "citation_quality_score": self._calculate_citation_quality(research_output),
            "bias_detection_score": self._calculate_bias_score(research_output),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate overall quality score
        weights = {
            "relevance_score": 0.25,
            "accuracy_score": 0.25,
            "completeness_score": 0.20,
            "citation_quality_score": 0.15,
            "bias_detection_score": 0.15
        }
        
        overall_score = sum(
            evaluation_results[metric] * weight 
            for metric, weight in weights.items()
            if metric in evaluation_results
        )
        
        evaluation_results["overall_quality_score"] = overall_score
        
        # Update Langfuse context
        if langfuse_manager.enabled:
            langfuse_context.update_current_observation(
                input=f"Evaluating research for query: {original_query}",
                output=f"Overall quality score: {overall_score:.2f}",
                metadata={
                    "evaluation_metrics": evaluation_results,
                    "quality_threshold_met": overall_score >= config.research.quality_score_minimum
                }
            )
        
        return evaluation_results
    
    def _calculate_relevance_score(self, research_output: Dict[str, Any], 
                                 original_query: str) -> float:
        """Calculate relevance score based on query-result alignment"""
        # Simplified relevance calculation
        synthesis = research_output.get("synthesis", "")
        query_words = set(original_query.lower().split())
        synthesis_words = set(synthesis.lower().split())
        
        if not query_words:
            return 0.0
        
        intersection = query_words.intersection(synthesis_words)
        return len(intersection) / len(query_words)
    
    def _calculate_accuracy_score(self, research_output: Dict[str, Any]) -> float:
        """Calculate accuracy score based on source reliability"""
        sources = research_output.get("search_results", [])
        if not sources:
            return 0.0
        
        # Simplified accuracy scoring based on source domains
        reliable_domains = ["edu", "gov", "org", "arxiv", "pubmed"]
        reliable_sources = 0
        
        for source in sources:
            url = source.get("url", "")
            if any(domain in url for domain in reliable_domains):
                reliable_sources += 1
        
        return reliable_sources / len(sources) if sources else 0.0
    
    def _calculate_completeness_score(self, research_output: Dict[str, Any]) -> float:
        """Calculate completeness score based on content coverage"""
        synthesis = research_output.get("synthesis", "")
        sources = research_output.get("search_results", [])
        citations = research_output.get("citations", [])
        
        # Scoring factors
        has_synthesis = len(synthesis) > 100
        has_sources = len(sources) >= 3
        has_citations = len(citations) > 0
        
        completeness_factors = [has_synthesis, has_sources, has_citations]
        return sum(completeness_factors) / len(completeness_factors)
    
    def _calculate_citation_quality(self, research_output: Dict[str, Any]) -> float:
        """Calculate citation quality score"""
        citations = research_output.get("citations", [])
        if not citations:
            return 0.0
        
        # Basic citation quality check
        valid_citations = sum(1 for citation in citations if len(citation) > 20)
        return valid_citations / len(citations)
    
    def _calculate_bias_score(self, research_output: Dict[str, Any]) -> float:
        """Calculate bias detection score (higher = less biased)"""
        synthesis = research_output.get("synthesis", "")
        
        # Simplified bias detection based on balanced language
        bias_indicators = ["always", "never", "all", "none", "definitely", "obviously"]
        bias_count = sum(1 for indicator in bias_indicators if indicator in synthesis.lower())
        
        # Return inverse of bias (higher score = less bias)
        max_bias = len(synthesis.split()) * 0.1  # Allow 10% bias words
        bias_ratio = bias_count / max_bias if max_bias > 0 else 0
        
        return max(0.0, 1.0 - bias_ratio)

# Global instances
research_metrics = ResearchMetricsTracker()
research_evaluator = ResearchQualityEvaluator()

# Export for easy imports
__all__ = [
    "langfuse_manager", "track_research_operation", "observe", "langfuse_context",
    "ResearchMetricsTracker", "ResearchQualityEvaluator", "research_metrics", "research_evaluator"
]