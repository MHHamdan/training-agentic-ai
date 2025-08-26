"""
Enhanced Financial State Management for LangGraph Multi-Agent System
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class MarketConditions(BaseModel):
    """Market conditions tracking"""
    vix: float = Field(default=15.0, description="VIX volatility index")
    trend: Literal["bullish", "bearish", "neutral"] = "neutral"
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    volume_ratio: float = Field(default=1.0, description="Volume vs average")
    market_cap_weighted_pe: float = Field(default=20.0)


class RiskAlert(BaseModel):
    """Risk alert structure"""
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    timestamp: datetime
    source_agent: str
    affected_symbols: List[str]
    recommended_action: Optional[str] = None


class AnalysisResult(BaseModel):
    """Analysis result from an agent"""
    agent_name: str
    analysis_type: str
    timestamp: datetime
    confidence_score: float = Field(ge=0.0, le=1.0)
    data: Dict[str, Any]
    recommendations: List[str] = []
    warnings: List[str] = []


class PortfolioPosition(BaseModel):
    """Portfolio position details"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    pnl: float
    pnl_percentage: float
    allocation_percentage: float


class FinancialAnalysisState(MessagesState):
    """
    Enhanced state management for financial multi-agent system.
    Extends MessagesState to maintain conversation history.
    """
    # Core analysis parameters
    target_symbols: List[str] = []
    analysis_type: Literal[
        "fundamental", 
        "technical", 
        "risk", 
        "portfolio", 
        "sentiment",
        "comprehensive"
    ] = "comprehensive"
    time_horizon: Literal["intraday", "short", "medium", "long"] = "medium"
    risk_tolerance: Literal["conservative", "moderate", "aggressive"] = "moderate"
    
    # Market context
    market_conditions: Optional[MarketConditions] = None
    current_prices: Dict[str, float] = {}
    market_data_timestamp: Optional[datetime] = None
    
    # Portfolio state
    portfolio_positions: List[PortfolioPosition] = []
    total_portfolio_value: float = 0.0
    available_cash: float = 100000.0  # Default starting cash
    
    # Analysis workflow state
    last_active_agent: str = ""
    active_analysis_threads: List[str] = []
    completed_analyses: Dict[str, AnalysisResult] = {}
    pending_approvals: List[str] = []
    workflow_id: str = ""
    
    # User interaction state
    user_preferences: Dict[str, Any] = {}
    notification_settings: Dict[str, bool] = {
        "critical_alerts": True,
        "trade_confirmations": True,
        "analysis_complete": True,
        "market_updates": False
    }
    approval_required: bool = False
    approval_status: Optional[str] = None
    
    # Results and alerts
    analysis_results: Dict[str, Any] = {}
    risk_alerts: List[RiskAlert] = []
    recommendations: List[Dict[str, Any]] = []
    compliance_status: Literal["approved", "review_required", "rejected", "pending"] = "pending"
    
    # Performance tracking
    analysis_start_time: Optional[datetime] = None
    analysis_end_time: Optional[datetime] = None
    total_api_calls: int = 0
    estimated_cost: float = 0.0
    
    # Alert management
    alert_acknowledged: bool = False
    critical_alert_active: bool = False
    
    # Advanced features
    use_real_time_data: bool = True
    enable_backtesting: bool = False
    backtesting_period: Optional[str] = None
    
    # Multi-threading support
    concurrent_analyses: Dict[str, str] = {}  # thread_id -> agent_name
    analysis_priority: Literal["normal", "high", "critical"] = "normal"
    
    # Audit trail
    audit_log: List[Dict[str, Any]] = []
    compliance_checks: List[Dict[str, Any]] = []
    
    def add_analysis_result(self, result: AnalysisResult):
        """Add an analysis result to the state"""
        self.completed_analyses[result.agent_name] = result
        self.analysis_results[result.analysis_type] = result.data
        
    def add_risk_alert(self, alert: RiskAlert):
        """Add a risk alert and update critical status"""
        self.risk_alerts.append(alert)
        if alert.severity == "critical":
            self.critical_alert_active = True
            
    def update_market_conditions(self, conditions: MarketConditions):
        """Update market conditions"""
        self.market_conditions = conditions
        self.market_data_timestamp = datetime.now()
        
    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log an audit event"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "agent": self.last_active_agent
        })