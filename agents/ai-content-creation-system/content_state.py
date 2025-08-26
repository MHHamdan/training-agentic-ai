"""
Enhanced Content Creation State Management for LangGraph Multi-Agent System
Following patterns from financial_state.py for consistency
"""

from typing import List, Dict, Any, Optional, Literal, Annotated
from datetime import datetime
from langgraph.graph import MessagesState, add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class SEOMetrics(BaseModel):
    """SEO performance metrics"""
    focus_keywords: List[str] = []
    keyword_density: Dict[str, float] = {}
    readability_score: float = Field(default=0.0, ge=0.0, le=100.0)
    seo_score: float = Field(default=0.0, ge=0.0, le=100.0)
    meta_title: str = ""
    meta_description: str = ""
    word_count: int = 0
    heading_structure: Dict[str, int] = {}


class ContentAlert(BaseModel):
    """Content creation alert structure"""
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    timestamp: datetime
    source_agent: str
    content_type: str
    recommended_action: Optional[str] = None


class ContentResult(BaseModel):
    """Content creation result from an agent"""
    agent_name: str
    content_type: str
    timestamp: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)
    content_data: Dict[str, Any]
    suggestions: List[str] = []
    warnings: List[str] = []


class BrandGuidelines(BaseModel):
    """Brand guidelines and voice settings"""
    brand_name: str = ""
    tone: Literal["professional", "casual", "friendly", "authoritative", "conversational", "technical"] = "professional"
    voice: Literal["active", "passive", "mixed"] = "active"
    style_keywords: List[str] = []
    avoid_words: List[str] = []
    target_audience: str = ""
    industry: str = ""
    content_pillars: List[str] = []


class ContentTemplate(BaseModel):
    """Content template structure"""
    template_type: str
    sections: List[str] = []
    word_count_target: int = 0
    seo_requirements: Dict[str, Any] = {}
    formatting_rules: Dict[str, str] = {}


class ContentCreationState(MessagesState):
    """
    Enhanced state management for content creation multi-agent system.
    Extends MessagesState to maintain conversation history.
    """
    # Core content parameters
    content_topic: str = ""
    content_type: Literal[
        "blog_post", 
        "social_media", 
        "website_copy", 
        "product_description",
        "email_campaign",
        "white_paper",
        "case_study",
        "technical_documentation"
    ] = "blog_post"
    target_keywords: List[str] = []
    primary_keyword: str = ""
    secondary_keywords: List[str] = []
    
    # Client and brand requirements
    client_name: str = ""
    brand_guidelines: Optional[BrandGuidelines] = None
    content_template: Optional[ContentTemplate] = None
    custom_requirements: Dict[str, Any] = {}
    
    # SEO and audience parameters
    target_audience: str = ""
    target_word_count: int = 800
    seo_metrics: Optional[SEOMetrics] = None
    competitor_analysis: Dict[str, Any] = {}
    trending_topics: List[str] = []
    
    # Research data
    research_sources: List[Dict[str, Any]] = []
    keyword_research: Dict[str, Any] = {}
    content_gaps: List[str] = []
    market_insights: Dict[str, Any] = {}
    
    # Content workflow state
    last_active_agent: str = ""
    research_completed: bool = False
    strategy_completed: bool = False
    content_drafted: bool = False
    seo_optimized: bool = False
    quality_checked: bool = False
    
    # Generated content
    content_outline: str = ""
    content_sections: Dict[str, str] = {}
    draft_content: str = ""
    optimized_content: str = ""
    final_content: str = ""
    
    # Content variations
    title_variations: List[str] = []
    meta_descriptions: List[str] = []
    call_to_actions: List[str] = []
    social_media_snippets: List[str] = []
    
    # Quality metrics
    readability_score: float = 0.0
    seo_score: float = 0.0
    originality_score: float = 0.0
    brand_alignment_score: float = 0.0
    engagement_potential: float = 0.0
    
    # Workflow management
    completed_analyses: Dict[str, ContentResult] = {}
    content_alerts: List[ContentAlert] = []
    approval_required: bool = False
    approval_status: Optional[str] = None
    
    # Performance tracking
    creation_start_time: Optional[datetime] = None
    creation_end_time: Optional[datetime] = None
    total_api_calls: int = 0
    estimated_cost: float = 0.0
    
    # Advanced features
    use_ai_research: bool = True
    enable_competitor_analysis: bool = True
    auto_seo_optimization: bool = True
    plagiarism_check: bool = True
    
    # Multi-platform support
    platform_variations: Dict[str, str] = {}  # platform -> content
    export_formats: List[str] = ["markdown", "html", "docx"]
    
    # Audit trail
    audit_log: List[Dict[str, Any]] = []
    content_revisions: List[Dict[str, Any]] = []
    
    def add_content_result(self, result: ContentResult):
        """Add a content creation result to the state"""
        self.completed_analyses[result.agent_name] = result
        
    def add_content_alert(self, alert: ContentAlert):
        """Add a content alert"""
        self.content_alerts.append(alert)
        
    def update_seo_metrics(self, metrics: SEOMetrics):
        """Update SEO metrics"""
        self.seo_metrics = metrics
        
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an audit event"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "agent": self.last_active_agent
        })
        
    def add_content_revision(self, version: str, content: str, changes: str):
        """Track content revisions"""
        self.content_revisions.append({
            "version": version,
            "content": content,
            "changes": changes,
            "timestamp": datetime.now().isoformat(),
            "agent": self.last_active_agent
        })