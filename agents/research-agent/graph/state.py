"""
Research State Management for LangGraph
Defines the state structure for multi-agent research workflow
"""

from typing import TypedDict, List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class ResearchPhase(Enum):
    """Research workflow phases"""
    INITIALIZATION = "initialization"
    SEARCH = "search"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    ERROR = "error"

class SourceQuality(Enum):
    """Source quality ratings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNRELIABLE = "unreliable"

class ResearchState(TypedDict):
    """
    State management for research workflow
    Tracks all information flow between agents
    """
    # Core research data
    query: str
    research_id: str
    timestamp: datetime
    phase: ResearchPhase
    
    # Search results
    search_results: List[Dict[str, Any]]
    search_metadata: Dict[str, Any]
    source_quality_scores: Dict[str, float]
    
    # Analysis outputs
    analyzed_content: List[Dict[str, Any]]
    key_insights: List[str]
    extracted_facts: List[Dict[str, Any]]
    relevance_scores: Dict[str, float]
    
    # Synthesis results
    synthesis: str
    executive_summary: str
    detailed_findings: Dict[str, Any]
    recommendations: List[str]
    
    # Citations and references
    citations: List[str]
    bibliography: List[Dict[str, Any]]
    citation_format: str
    
    # Evaluation metrics
    evaluation: Dict[str, Any]
    quality_score: Optional[float]
    accuracy_score: Optional[float]
    completeness_score: Optional[float]
    bias_score: Optional[float]
    reliability_score: Optional[float]
    
    # Fact checking
    fact_check_results: List[Dict[str, Any]]
    verified_claims: List[str]
    disputed_claims: List[str]
    
    # Model tracking
    models_used: Dict[str, List[str]]
    model_performance: Dict[str, Dict[str, Any]]
    model_comparisons: Optional[Dict[str, Any]]
    
    # Metadata and tracking
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    processing_time: Dict[str, float]
    
    # Langfuse tracking
    trace_id: Optional[str]
    session_id: Optional[str]
    observation_ids: Dict[str, str]
    
    # User preferences
    user_preferences: Dict[str, Any]
    output_format: str
    language: str
    
    # Academic compliance
    academic_standards_met: bool
    plagiarism_check: Optional[Dict[str, Any]]
    citation_compliance: bool

class SearchResult(TypedDict):
    """Structure for individual search results"""
    source: str
    title: str
    url: Optional[str]
    content: str
    snippet: str
    published_date: Optional[datetime]
    author: Optional[str]
    relevance_score: float
    quality_score: float
    metadata: Dict[str, Any]

class AnalysisResult(TypedDict):
    """Structure for analysis outputs"""
    source_id: str
    summary: str
    key_points: List[str]
    extracted_entities: List[Dict[str, str]]
    sentiment: Dict[str, float]
    topics: List[str]
    relevance_to_query: float
    confidence: float
    metadata: Dict[str, Any]

class Citation(TypedDict):
    """Structure for academic citations"""
    id: str
    authors: List[str]
    title: str
    publication: Optional[str]
    year: Optional[int]
    url: Optional[str]
    doi: Optional[str]
    citation_text: str
    format: str

class EvaluationMetrics(TypedDict):
    """Structure for research quality evaluation"""
    overall_quality: float
    accuracy: float
    completeness: float
    relevance: float
    bias_detection: float
    source_reliability: float
    citation_quality: float
    academic_compliance: float
    fact_check_score: float
    confidence: float
    recommendations: List[str]
    issues_found: List[str]

class ModelPerformance(TypedDict):
    """Structure for model performance tracking"""
    model_name: str
    task_type: str
    execution_time: float
    token_usage: Dict[str, int]
    cost_estimate: float
    quality_score: float
    errors: List[str]
    metadata: Dict[str, Any]

def create_initial_state(query: str, **kwargs) -> ResearchState:
    """
    Create initial research state
    
    Args:
        query: Research query
        **kwargs: Additional state parameters
    
    Returns:
        Initialized ResearchState
    """
    from datetime import datetime
    import uuid
    
    return ResearchState(
        query=query,
        research_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        phase=ResearchPhase.INITIALIZATION,
        
        # Initialize empty collections
        search_results=[],
        search_metadata={},
        source_quality_scores={},
        analyzed_content=[],
        key_insights=[],
        extracted_facts=[],
        relevance_scores={},
        
        # Initialize empty synthesis
        synthesis="",
        executive_summary="",
        detailed_findings={},
        recommendations=[],
        
        # Initialize citations
        citations=[],
        bibliography=[],
        citation_format=kwargs.get("citation_format", "APA"),
        
        # Initialize evaluation
        evaluation={},
        quality_score=None,
        accuracy_score=None,
        completeness_score=None,
        bias_score=None,
        reliability_score=None,
        
        # Initialize fact checking
        fact_check_results=[],
        verified_claims=[],
        disputed_claims=[],
        
        # Initialize model tracking
        models_used={},
        model_performance={},
        model_comparisons=None,
        
        # Initialize metadata
        metadata=kwargs.get("metadata", {}),
        errors=[],
        warnings=[],
        processing_time={},
        
        # Langfuse tracking
        trace_id=kwargs.get("trace_id"),
        session_id=kwargs.get("session_id"),
        observation_ids={},
        
        # User preferences
        user_preferences=kwargs.get("user_preferences", {}),
        output_format=kwargs.get("output_format", "detailed"),
        language=kwargs.get("language", "en"),
        
        # Academic compliance
        academic_standards_met=False,
        plagiarism_check=None,
        citation_compliance=False
    )

def update_state_phase(state: ResearchState, phase: ResearchPhase) -> ResearchState:
    """Update the research phase in state"""
    state["phase"] = phase
    state["metadata"]["phase_updated_at"] = datetime.now().isoformat()
    return state

def add_error_to_state(state: ResearchState, error: str, phase: str = None) -> ResearchState:
    """Add error to state tracking"""
    error_entry = {
        "error": error,
        "phase": phase or state.get("phase", "unknown"),
        "timestamp": datetime.now().isoformat()
    }
    state["errors"].append(error_entry)
    return state

def add_warning_to_state(state: ResearchState, warning: str) -> ResearchState:
    """Add warning to state tracking"""
    warning_entry = {
        "warning": warning,
        "timestamp": datetime.now().isoformat()
    }
    state["warnings"].append(warning_entry)
    return state

def calculate_overall_quality(state: ResearchState) -> float:
    """Calculate overall research quality score"""
    scores = []
    
    if state.get("accuracy_score"):
        scores.append(state["accuracy_score"])
    if state.get("completeness_score"):
        scores.append(state["completeness_score"])
    if state.get("reliability_score"):
        scores.append(state["reliability_score"])
    if state.get("bias_score"):
        scores.append(1.0 - state["bias_score"])  # Lower bias is better
    
    if scores:
        return sum(scores) / len(scores)
    return 0.0

def is_research_complete(state: ResearchState) -> bool:
    """Check if research workflow is complete"""
    return (
        state.get("phase") == ResearchPhase.COMPLETE and
        state.get("synthesis") and
        state.get("evaluation") and
        state.get("quality_score") is not None
    )

def get_state_summary(state: ResearchState) -> Dict[str, Any]:
    """Get summary of current research state"""
    return {
        "research_id": state.get("research_id"),
        "query": state.get("query"),
        "phase": state.get("phase"),
        "sources_found": len(state.get("search_results", [])),
        "insights_extracted": len(state.get("key_insights", [])),
        "quality_score": state.get("quality_score"),
        "errors": len(state.get("errors", [])),
        "warnings": len(state.get("warnings", [])),
        "is_complete": is_research_complete(state)
    }