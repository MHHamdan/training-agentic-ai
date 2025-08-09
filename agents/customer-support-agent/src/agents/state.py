"""State and memory schemas for the Customer Support Agent"""

from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, validator
import uuid


class UserProfile(BaseModel):
    """Schema for user personal information"""
    user_id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User full name")
    account_type: str = Field(default="standard", description="Account subscription level")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('email')
    def validate_email(cls, v):
        """Basic email validation"""
        if '@' not in v or '.' not in v:
            raise ValueError('Invalid email format')
        return v.lower()
    
    @validator('account_type')
    def validate_account_type(cls, v):
        """Validate account type"""
        valid_types = ['standard', 'premium', 'enterprise']
        if v.lower() not in valid_types:
            raise ValueError(f'Account type must be one of: {", ".join(valid_types)}')
        return v.lower()


class QueryHistory(BaseModel):
    """Schema for storing query history"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique query identifier")
    user_id: str = Field(..., description="Associated user ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    query_text: str = Field(..., description="Original query text")
    category: str = Field(..., description="Query category")
    resolution: Optional[str] = Field(None, description="Query resolution")
    escalated: bool = Field(default=False)
    resolved_by: Optional[str] = Field(None, description="Agent or human who resolved")
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5)
    response_time_seconds: Optional[float] = Field(None, description="Time to resolve in seconds")
    
    @validator('category')
    def validate_category(cls, v):
        """Validate query category"""
        valid_categories = ['technical', 'billing', 'feature', 'account', 'general']
        if v.lower() not in valid_categories:
            return 'general'  # Default to general if invalid
        return v.lower()


class ConversationContext(BaseModel):
    """Schema for conversation context"""
    thread_id: str = Field(..., description="Conversation thread ID")
    session_start: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    message_count: int = Field(default=0)
    context_summary: Optional[str] = Field(None, description="AI-generated context summary")
    sentiment: Optional[str] = Field(None, description="Overall conversation sentiment")
    priority: str = Field(default="normal", description="Conversation priority level")
    
    @validator('priority')
    def validate_priority(cls, v):
        """Validate priority level"""
        valid_priorities = ['low', 'normal', 'high', 'urgent']
        if v.lower() not in valid_priorities:
            return 'normal'
        return v.lower()


class EscalationInfo(BaseModel):
    """Schema for escalation information"""
    escalation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    escalated_at: datetime = Field(default_factory=datetime.now)
    reason: str = Field(..., description="Reason for escalation")
    urgency_level: str = Field(default="medium", description="Urgency level")
    assigned_agent: Optional[str] = Field(None, description="Human agent assigned")
    estimated_resolution_time: Optional[datetime] = Field(None)
    customer_notified: bool = Field(default=False)
    
    @validator('urgency_level')
    def validate_urgency(cls, v):
        """Validate urgency level"""
        valid_urgencies = ['low', 'medium', 'high', 'critical']
        if v.lower() not in valid_urgencies:
            return 'medium'
        return v.lower()


class AgentState(MessagesState):
    """Main agent state with comprehensive memory management"""
    
    # User identification
    user_id: str
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Current conversation state
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_query: Optional[str] = None
    query_category: Optional[str] = None
    
    # Memory references
    user_profile: Optional[UserProfile] = None
    query_history: List[QueryHistory] = Field(default_factory=list)
    conversation_context: Optional[ConversationContext] = None
    
    # HITL and escalation
    requires_human: bool = False
    escalation_info: Optional[EscalationInfo] = None
    human_agent_id: Optional[str] = None
    
    # Agent behavior configuration
    max_messages: int = 10  # For message trimming
    confidence_threshold: float = 0.7
    enable_auto_responses: bool = True
    
    # State management
    is_active: bool = True
    last_updated: datetime = Field(default_factory=datetime.now)
    state_version: str = "1.0.0"
    
    # Processing metadata
    processing_start_time: Optional[datetime] = None
    response_generated_at: Optional[datetime] = None
    total_processing_time: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('user_id', 'thread_id', 'session_id')
    def validate_ids(cls, v):
        """Validate required ID fields"""
        if not v or not isinstance(v, str):
            raise ValueError("ID must be a non-empty string")
        return v
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        """Validate confidence threshold"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v
    
    @validator('max_messages')
    def validate_max_messages(cls, v):
        """Validate max messages"""
        if v < 1:
            raise ValueError("Max messages must be at least 1")
        return min(v, 50)  # Cap at 50 for performance


class MessageMetadata(BaseModel):
    """Schema for message metadata"""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(..., description="Source of message (user, agent, system, human)")
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    processing_time: Optional[float] = Field(None, description="Time to generate response")
    auto_response: bool = Field(default=False)
    escalated: bool = Field(default=False)
    human_reviewed: bool = Field(default=False)
    feedback_score: Optional[int] = Field(None, ge=1, le=5)
    tags: List[str] = Field(default_factory=list)
    
    @validator('source')
    def validate_source(cls, v):
        """Validate message source"""
        valid_sources = ['user', 'agent', 'system', 'human']
        if v.lower() not in valid_sources:
            raise ValueError(f'Source must be one of: {", ".join(valid_sources)}')
        return v.lower()


class AgentMetrics(BaseModel):
    """Schema for agent performance metrics"""
    session_id: str = Field(..., description="Session identifier")
    total_queries: int = Field(default=0)
    resolved_queries: int = Field(default=0)
    escalated_queries: int = Field(default=0)
    average_response_time: float = Field(default=0.0)
    average_confidence_score: float = Field(default=0.0)
    user_satisfaction_score: Optional[float] = Field(None, ge=1, le=5)
    session_duration: Optional[float] = Field(None, description="Session duration in seconds")
    
    @property
    def resolution_rate(self) -> float:
        """Calculate resolution rate"""
        if self.total_queries == 0:
            return 0.0
        return self.resolved_queries / self.total_queries
    
    @property
    def escalation_rate(self) -> float:
        """Calculate escalation rate"""
        if self.total_queries == 0:
            return 0.0
        return self.escalated_queries / self.total_queries


class KnowledgeBaseEntry(BaseModel):
    """Schema for knowledge base entries"""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., description="Entry title")
    content: str = Field(..., description="Entry content")
    category: str = Field(..., description="Entry category")
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = Field(default=0)
    effectiveness_score: float = Field(default=0.0, ge=0, le=1)
    author: Optional[str] = Field(None, description="Entry author")
    approved: bool = Field(default=False)


# Type aliases for better code readability
StateDict = Dict[str, Any]
MessageList = List[Dict[str, Any]]
MetadataDict = Dict[str, Any]

# Constants for state management
class StateConstants:
    """Constants used throughout the state management system"""
    
    # Message roles
    ROLE_USER = "user"
    ROLE_AGENT = "agent"
    ROLE_SYSTEM = "system"
    ROLE_HUMAN = "human"
    
    # Query categories
    CATEGORY_TECHNICAL = "technical"
    CATEGORY_BILLING = "billing"
    CATEGORY_FEATURE = "feature"
    CATEGORY_ACCOUNT = "account"
    CATEGORY_GENERAL = "general"
    
    # Escalation reasons
    ESCALATION_COMPLEXITY = "query_complexity"
    ESCALATION_LOW_CONFIDENCE = "low_confidence"
    ESCALATION_USER_REQUEST = "user_request"
    ESCALATION_TIMEOUT = "timeout"
    ESCALATION_ERROR = "processing_error"
    
    # Priority levels
    PRIORITY_LOW = "low"
    PRIORITY_NORMAL = "normal"
    PRIORITY_HIGH = "high"
    PRIORITY_URGENT = "urgent"
    
    # Account types
    ACCOUNT_STANDARD = "standard"
    ACCOUNT_PREMIUM = "premium"
    ACCOUNT_ENTERPRISE = "enterprise"


def create_initial_state(user_id: str, **kwargs) -> AgentState:
    """Create an initial agent state with default values"""
    return AgentState(
        user_id=user_id,
        thread_id=kwargs.get('thread_id', str(uuid.uuid4())),
        session_id=kwargs.get('session_id', str(uuid.uuid4())),
        messages=[],
        **kwargs
    )


def update_state_timestamp(state: AgentState) -> AgentState:
    """Update the state's last_updated timestamp"""
    state.last_updated = datetime.now()
    return state


def validate_state(state: AgentState) -> bool:
    """Validate the current state"""
    try:
        # Check required fields
        if not state.user_id or not state.thread_id:
            return False
        
        # Validate message count
        if len(state.messages) > state.max_messages * 2:  # Allow some buffer
            return False
        
        # Validate confidence threshold
        if not 0 <= state.confidence_threshold <= 1:
            return False
        
        return True
    except Exception:
        return False
