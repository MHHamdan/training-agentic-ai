import agentops
import logging
import time
import json
from typing import Dict, Any, Optional, List
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, asdict

from config import config

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance tracking for agents and tools"""
    agent_name: str
    task_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_estimate: Optional[float] = None
    model_used: Optional[str] = None
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Mark the metric as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error_message = error_message

@dataclass 
class ComplianceLog:
    """Compliance and audit trail logging"""
    timestamp: datetime
    agent_name: str
    action_type: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    risk_level: str
    compliance_status: str
    decision_reasoning: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AgentOpsManager:
    """Centralized AgentOps integration and observability management"""
    
    def __init__(self):
        self.session = None
        self.performance_metrics: List[PerformanceMetrics] = []
        self.compliance_logs: List[ComplianceLog] = []
        self.active_metrics: Dict[str, PerformanceMetrics] = {}
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize AgentOps session with configuration"""
        try:
            if not config.agentops_api_key:
                logger.warning("AgentOps API key not found. Running without observability.")
                return
            
            self.session = agentops.init(
                api_key=config.agentops_api_key,
                project_name=config.agentops.project_name,
                tags=config.agentops.tags,
                default_tags=["stock-analysis", "financial", "crewai"]
            )
            
            logger.info(f"AgentOps session initialized: {self.session.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AgentOps: {str(e)}")
            self.session = None
    
    def start_agent_tracking(self, agent_name: str, task_type: str, 
                           input_data: Dict[str, Any] = None) -> str:
        """Start tracking an agent's performance"""
        tracking_id = f"{agent_name}_{int(time.time())}"
        
        metric = PerformanceMetrics(
            agent_name=agent_name,
            task_type=task_type,
            start_time=time.time()
        )
        
        self.active_metrics[tracking_id] = metric
        
        if self.session:
            try:
                # Log agent start event
                agentops.record(
                    event_type="agent_start",
                    event_data={
                        "agent_name": agent_name,
                        "task_type": task_type,
                        "tracking_id": tracking_id,
                        "input_data": input_data or {}
                    }
                )
            except Exception as e:
                logger.error(f"AgentOps logging error: {str(e)}")
        
        return tracking_id
    
    def end_agent_tracking(self, tracking_id: str, success: bool = True, 
                          error_message: Optional[str] = None,
                          output_data: Dict[str, Any] = None,
                          model_used: Optional[str] = None):
        """End tracking for an agent"""
        if tracking_id not in self.active_metrics:
            logger.warning(f"Tracking ID {tracking_id} not found")
            return
        
        metric = self.active_metrics[tracking_id]
        metric.complete(success, error_message)
        metric.model_used = model_used
        
        self.performance_metrics.append(metric)
        del self.active_metrics[tracking_id]
        
        if self.session:
            try:
                agentops.record(
                    event_type="agent_complete",
                    event_data={
                        "tracking_id": tracking_id,
                        "duration": metric.duration,
                        "success": success,
                        "error_message": error_message,
                        "output_data": output_data or {},
                        "model_used": model_used
                    }
                )
            except Exception as e:
                logger.error(f"AgentOps logging error: {str(e)}")
    
    def log_tool_usage(self, tool_name: str, input_data: Dict[str, Any],
                      output_data: Dict[str, Any], duration: float,
                      success: bool = True):
        """Log tool usage for observability"""
        if self.session:
            try:
                agentops.record(
                    event_type="tool_usage",
                    event_data={
                        "tool_name": tool_name,
                        "input_data": input_data,
                        "output_data": output_data,
                        "duration": duration,
                        "success": success,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"AgentOps tool logging error: {str(e)}")
    
    def log_compliance_action(self, agent_name: str, action_type: str,
                            input_data: Dict[str, Any], output_data: Dict[str, Any],
                            risk_level: str, decision_reasoning: str,
                            user_id: Optional[str] = None):
        """Log compliance-related actions for audit trail"""
        compliance_log = ComplianceLog(
            timestamp=datetime.now(),
            agent_name=agent_name,
            action_type=action_type,
            input_data=input_data,
            output_data=output_data,
            risk_level=risk_level,
            compliance_status="compliant",
            decision_reasoning=decision_reasoning,
            user_id=user_id,
            session_id=self.session.session_id if self.session else None
        )
        
        self.compliance_logs.append(compliance_log)
        
        if self.session:
            try:
                agentops.record(
                    event_type="compliance_action",
                    event_data=asdict(compliance_log)
                )
            except Exception as e:
                logger.error(f"AgentOps compliance logging error: {str(e)}")
    
    def log_model_comparison(self, models_tested: List[str], 
                           performance_results: Dict[str, Any],
                           winner: str, task_type: str):
        """Log model comparison results"""
        if self.session:
            try:
                agentops.record(
                    event_type="model_comparison",
                    event_data={
                        "models_tested": models_tested,
                        "performance_results": performance_results,
                        "winner": winner,
                        "task_type": task_type,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                logger.error(f"AgentOps model comparison logging error: {str(e)}")
    
    def get_session_metrics(self) -> Dict[str, Any]:
        """Get current session performance metrics"""
        total_tasks = len(self.performance_metrics)
        successful_tasks = sum(1 for m in self.performance_metrics if m.success)
        average_duration = sum(m.duration or 0 for m in self.performance_metrics) / total_tasks if total_tasks > 0 else 0
        
        return {
            "session_id": self.session.session_id if self.session else None,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_duration": average_duration,
            "total_compliance_logs": len(self.compliance_logs),
            "active_tracking": len(self.active_metrics)
        }
    
    def export_audit_trail(self, format: str = "json") -> str:
        """Export compliance audit trail"""
        if format == "json":
            audit_data = {
                "session_id": self.session.session_id if self.session else None,
                "export_timestamp": datetime.now().isoformat(),
                "compliance_logs": [asdict(log) for log in self.compliance_logs],
                "performance_metrics": [asdict(metric) for metric in self.performance_metrics]
            }
            return json.dumps(audit_data, indent=2, default=str)
        
        # Add other formats as needed
        return ""
    
    def end_session(self):
        """End the AgentOps session"""
        if self.session:
            try:
                agentops.end_session("Success")
                logger.info("AgentOps session ended successfully")
            except Exception as e:
                logger.error(f"Error ending AgentOps session: {str(e)}")

def track_agent_performance(agent_name: str, task_type: str):
    """Decorator to automatically track agent performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            obs = get_observability_manager()
            tracking_id = obs.start_agent_tracking(agent_name, task_type, kwargs)
            
            try:
                result = await func(*args, **kwargs)
                obs.end_agent_tracking(tracking_id, success=True, output_data={"result": str(result)})
                return result
            except Exception as e:
                obs.end_agent_tracking(tracking_id, success=False, error_message=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            obs = get_observability_manager()
            tracking_id = obs.start_agent_tracking(agent_name, task_type, kwargs)
            
            try:
                result = func(*args, **kwargs)
                obs.end_agent_tracking(tracking_id, success=True, output_data={"result": str(result)})
                return result
            except Exception as e:
                obs.end_agent_tracking(tracking_id, success=False, error_message=str(e))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def track_tool_usage(tool_name: str):
    """Decorator to track tool usage"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            obs = get_observability_manager()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                obs.log_tool_usage(
                    tool_name=tool_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    output_data={"result": str(result)[:1000]},  # Truncate long outputs
                    duration=duration,
                    success=True
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                obs.log_tool_usage(
                    tool_name=tool_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    output_data={"error": str(e)},
                    duration=duration,
                    success=False
                )
                raise
        
        return wrapper
    return decorator

# Global observability manager instance
_observability_manager: Optional[AgentOpsManager] = None

def get_observability_manager() -> AgentOpsManager:
    """Get the global observability manager instance"""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = AgentOpsManager()
    return _observability_manager

def initialize_observability() -> AgentOpsManager:
    """Initialize the observability system"""
    global _observability_manager
    _observability_manager = AgentOpsManager()
    return _observability_manager