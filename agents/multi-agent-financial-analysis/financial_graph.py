"""
Main LangGraph Implementation for Multi-Agent Financial Analysis System
"""

from typing import Dict, List, Any, Literal
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import uuid

from financial_state import FinancialAnalysisState, RiskAlert, MarketConditions
from financial_agents import (
    call_market_research_agent,
    call_technical_analysis_agent,
    call_risk_assessment_agent,
    call_sentiment_analysis_agent,
    call_portfolio_optimization_agent,
    call_compliance_agent,
    call_report_generation_agent
)


def route_initial_request(state: FinancialAnalysisState) -> str:
    """
    Route initial user request to appropriate starting agent based on query analysis
    """
    if not state.get("messages"):
        return "market_research_agent"  # Default start
    
    user_query = state["messages"][-1].content.lower() if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1]).lower()
    
    # Analyze query to determine best starting point
    routing_keywords = {
        "risk_assessment_agent": ["risk", "volatility", "var", "downside", "protection", "hedge", "stress test"],
        "technical_analysis_agent": ["technical", "chart", "pattern", "indicator", "rsi", "macd", "trend", "support", "resistance"],
        "sentiment_analysis_agent": ["sentiment", "news", "social", "twitter", "reddit", "opinion", "buzz"],
        "portfolio_optimization_agent": ["portfolio", "allocation", "rebalance", "diversification", "optimize", "weight"],
        "compliance_agent": ["compliance", "regulatory", "esg", "ethical", "restriction", "insider"],
        "market_research_agent": ["fundamental", "valuation", "earnings", "revenue", "pe", "growth", "analysis"]
    }
    
    # Count keyword matches for each agent
    scores = {}
    for agent, keywords in routing_keywords.items():
        score = sum(1 for keyword in keywords if keyword in user_query)
        if score > 0:
            scores[agent] = score
    
    # Return agent with highest score, or default to market research
    if scores:
        return max(scores, key=scores.get)
    
    # Check analysis type if specified
    analysis_type = state.get("analysis_type", "comprehensive")
    if analysis_type == "technical":
        return "technical_analysis_agent"
    elif analysis_type == "risk":
        return "risk_assessment_agent"
    elif analysis_type == "sentiment":
        return "sentiment_analysis_agent"
    elif analysis_type == "portfolio":
        return "portfolio_optimization_agent"
    elif analysis_type == "fundamental":
        return "market_research_agent"
    
    return "market_research_agent"  # Default


def route_based_on_market_conditions(state: FinancialAnalysisState) -> str:
    """
    Smart routing based on current market volatility and conditions
    """
    market_conditions = state.get("market_conditions")
    
    # Check for critical alerts first
    if state.get("critical_alert_active", False):
        return "market_alert_node"
    
    # Check market volatility
    if market_conditions:
        vix = market_conditions.vix
        
        # High volatility scenario - prioritize risk
        if vix > 30:
            if "risk_analysis" not in state.get("completed_analyses", {}):
                return "risk_assessment_agent"
            elif state.get("approval_required", False):
                return "human_approval_node"
        
        # Very high volatility - require human oversight
        if vix > 40:
            return "human_approval_node"
    
    # Check compliance requirements
    if state.get("compliance_status") == "review_required":
        if "compliance_analysis" not in state.get("completed_analyses", {}):
            return "compliance_agent"
        else:
            return "human_approval_node"
    
    # Check if human approval is required
    if state.get("approval_required", False):
        return "human_approval_node"
    
    # Route based on analysis completeness
    return route_next_agent(state)


def route_next_agent(state: FinancialAnalysisState) -> str:
    """
    Determine next agent based on workflow and completed analyses
    """
    completed = set(state.get("completed_analyses", {}).keys())
    analysis_type = state.get("analysis_type", "comprehensive")
    
    # Comprehensive analysis workflow
    if analysis_type == "comprehensive":
        workflow_sequence = [
            ("fundamental_analysis", "market_research_agent"),
            ("technical_analysis", "technical_analysis_agent"),
            ("sentiment_analysis", "sentiment_analysis_agent"),
            ("risk_analysis", "risk_assessment_agent"),
            ("portfolio_analysis", "portfolio_optimization_agent"),
            ("compliance_analysis", "compliance_agent"),
            ("final_report", "report_generation_agent")
        ]
        
        for analysis_name, agent_name in workflow_sequence:
            if analysis_name not in completed:
                return agent_name
    
    # Quick scan workflow
    elif analysis_type == "technical":
        if "technical_analysis" not in completed:
            return "technical_analysis_agent"
        if "risk_analysis" not in completed:
            return "risk_assessment_agent"
    
    # Risk-focused workflow
    elif analysis_type == "risk":
        if "risk_analysis" not in completed:
            return "risk_assessment_agent"
        if "portfolio_analysis" not in completed:
            return "portfolio_optimization_agent"
    
    # Default to report generation if all analyses complete
    if "final_report" not in completed:
        return "report_generation_agent"
    
    return "human_interaction"


def human_interaction_node(state: FinancialAnalysisState, config) -> Command:
    """
    Handle human interaction and user input
    """
    # Get user input
    user_input = interrupt(value="Analysis ready. Please provide your input or 'continue' to proceed.")
    
    if isinstance(user_input, str):
        # Parse user commands
        if user_input.lower() in ["continue", "proceed", "next"]:
            # Continue with workflow
            next_agent = route_based_on_market_conditions(state)
            return Command(goto=next_agent)
        
        elif user_input.lower() in ["report", "generate report", "finish"]:
            # Jump to report generation
            return Command(goto="report_generation_agent")
        
        elif user_input.lower().startswith("analyze"):
            # Parse new analysis request
            return Command(
                update={"messages": [{"role": "human", "content": user_input}]},
                goto=route_initial_request(state)
            )
        
        else:
            # Add as message and route appropriately
            return Command(
                update={"messages": [{"role": "human", "content": user_input}]},
                goto=state.get("last_active_agent", "market_research_agent")
            )
    
    # Default continuation
    return Command(goto=route_based_on_market_conditions(state))


def human_approval_node(state: FinancialAnalysisState, config) -> Command:
    """
    Handle human approval for high-risk recommendations
    """
    # Compile items requiring approval
    approval_items = []
    
    # Check for high-risk recommendations
    for rec in state.get("recommendations", []):
        if isinstance(rec, dict) and rec.get("risk_level") == "high":
            approval_items.append(rec)
    
    # Check for compliance issues
    if state.get("compliance_status") == "review_required":
        approval_items.append({
            "type": "compliance",
            "message": "Compliance review required for proposed trades"
        })
    
    # Check for large portfolio changes
    if state.get("approval_required", False):
        approval_items.append({
            "type": "portfolio_change",
            "message": "Large portfolio rebalancing requires approval"
        })
    
    if approval_items:
        approval_message = f"""
        APPROVAL REQUIRED:
        
        The following items require your approval:
        {approval_items}
        
        Please respond with 'approve', 'reject', or 'modify' followed by your instructions.
        """
        
        user_decision = interrupt(value=approval_message)
        
        if isinstance(user_decision, str):
            decision_lower = user_decision.lower()
            
            if "approve" in decision_lower:
                # Approved - continue workflow
                update = {
                    "approval_status": "approved",
                    "approval_required": False,
                    "compliance_status": "approved" if state.get("compliance_status") == "review_required" else state.get("compliance_status")
                }
                
                # Log approval
                audit_log = state.get("audit_log", [])
                audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "approval_granted",
                    "details": {
                        "items": approval_items,
                        "approver": "user",
                        "timestamp": datetime.now().isoformat()
                    },
                    "agent": "human_approval_node"
                })
                update["audit_log"] = audit_log
                
                return Command(update=update, goto=route_next_agent(state))
            
            elif "reject" in decision_lower:
                # Rejected - re-evaluate with lower risk tolerance
                update = {
                    "approval_status": "rejected",
                    "approval_required": False,
                    "risk_tolerance": "conservative"  # Lower risk tolerance
                }
                
                # Log rejection
                audit_log = state.get("audit_log", [])
                audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "approval_rejected",
                    "details": {
                        "items": approval_items,
                        "timestamp": datetime.now().isoformat()
                    },
                    "agent": "human_approval_node"
                })
                update["audit_log"] = audit_log
                
                # Re-run risk assessment with new parameters
                return Command(update=update, goto="risk_assessment_agent")
            
            elif "modify" in decision_lower:
                # Parse modifications and update state
                return Command(
                    update={"messages": [{"role": "human", "content": user_decision}]},
                    goto="portfolio_optimization_agent"
                )
    
    # No approval needed, continue
    return Command(goto=route_next_agent(state))


def market_alert_node(state: FinancialAnalysisState, config) -> Command:
    """
    Handle real-time market alerts and critical events
    """
    # Get critical alerts
    critical_alerts = [
        alert for alert in state.get("risk_alerts", [])
        if alert.severity == "critical"
    ]
    
    if critical_alerts:
        # Format alert message
        alert_messages = []
        for alert in critical_alerts:
            alert_messages.append(f"""
            🚨 CRITICAL ALERT: {alert.message}
            Source: {alert.source_agent}
            Affected: {', '.join(alert.affected_symbols)}
            Recommended Action: {alert.recommended_action or 'Review immediately'}
            """)
        
        full_message = "\n".join(alert_messages) + "\n\nPress Enter to acknowledge and continue with risk assessment."
        
        # Wait for user acknowledgment
        user_response = interrupt(value=full_message)
        
        # Update state
        update = {
            "alert_acknowledged": True,
            "critical_alert_active": False
        }
        
        # Log alert acknowledgment
        audit_log = state.get("audit_log", [])
        audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": "critical_alert_acknowledged",
            "details": {
                "alerts": [alert.dict() for alert in critical_alerts],
                "timestamp": datetime.now().isoformat()
            },
            "agent": "market_alert_node"
        })
        update["audit_log"] = audit_log
        
        # Route to risk assessment for immediate action
        return Command(update=update, goto="risk_assessment_agent")
    
    # No critical alerts, continue normal flow
    return Command(goto=route_next_agent(state))


def supervisor_node(state: FinancialAnalysisState) -> Command:
    """
    Supervisor node for high-level orchestration and decision making
    """
    # Check if all required analyses are complete
    completed = set(state.get("completed_analyses", {}).keys())
    analysis_type = state.get("analysis_type", "comprehensive")
    
    required_analyses = {
        "comprehensive": ["fundamental_analysis", "technical_analysis", "sentiment_analysis", 
                         "risk_analysis", "portfolio_analysis", "compliance_analysis"],
        "technical": ["technical_analysis", "risk_analysis"],
        "risk": ["risk_analysis", "portfolio_analysis"],
        "sentiment": ["sentiment_analysis", "market_research"],
        "portfolio": ["portfolio_analysis", "risk_analysis"],
        "fundamental": ["fundamental_analysis", "market_research"]
    }
    
    required = set(required_analyses.get(analysis_type, []))
    missing = required - completed
    
    if missing:
        # Route to next missing analysis
        agent_map = {
            "fundamental_analysis": "market_research_agent",
            "technical_analysis": "technical_analysis_agent",
            "sentiment_analysis": "sentiment_analysis_agent",
            "risk_analysis": "risk_assessment_agent",
            "portfolio_analysis": "portfolio_optimization_agent",
            "compliance_analysis": "compliance_agent"
        }
        
        for analysis in missing:
            if analysis in agent_map:
                return Command(goto=agent_map[analysis])
    
    # All analyses complete, generate report
    if "final_report" not in completed:
        return Command(goto="report_generation_agent")
    
    # Everything complete
    return Command(goto=END)


def build_financial_analysis_graph() -> StateGraph:
    """
    Construct the complete financial analysis LangGraph with all agents and routing
    """
    
    # Initialize the graph with our custom state
    builder = StateGraph(FinancialAnalysisState)
    
    # Add all agent nodes
    builder.add_node("market_research_agent", call_market_research_agent)
    builder.add_node("technical_analysis_agent", call_technical_analysis_agent)
    builder.add_node("risk_assessment_agent", call_risk_assessment_agent)
    builder.add_node("sentiment_analysis_agent", call_sentiment_analysis_agent)
    builder.add_node("portfolio_optimization_agent", call_portfolio_optimization_agent)
    builder.add_node("compliance_agent", call_compliance_agent)
    builder.add_node("report_generation_agent", call_report_generation_agent)
    
    # Add interaction nodes
    builder.add_node("human_interaction", human_interaction_node)
    builder.add_node("human_approval_node", human_approval_node)
    builder.add_node("market_alert_node", market_alert_node)
    builder.add_node("supervisor", supervisor_node)
    
    # Add conditional entry point routing
    builder.add_conditional_edges(
        START,
        route_initial_request,
        {
            "market_research_agent": "market_research_agent",
            "technical_analysis_agent": "technical_analysis_agent",
            "risk_assessment_agent": "risk_assessment_agent",
            "sentiment_analysis_agent": "sentiment_analysis_agent",
            "portfolio_optimization_agent": "portfolio_optimization_agent",
            "compliance_agent": "compliance_agent"
        }
    )
    
    # Add edges from human interaction with market condition routing
    builder.add_conditional_edges(
        "human_interaction",
        route_based_on_market_conditions,
        {
            "market_research_agent": "market_research_agent",
            "technical_analysis_agent": "technical_analysis_agent",
            "risk_assessment_agent": "risk_assessment_agent",
            "sentiment_analysis_agent": "sentiment_analysis_agent",
            "portfolio_optimization_agent": "portfolio_optimization_agent",
            "compliance_agent": "compliance_agent",
            "report_generation_agent": "report_generation_agent",
            "human_approval_node": "human_approval_node",
            "market_alert_node": "market_alert_node"
        }
    )
    
    # Add edges from each agent to human interaction
    agents = [
        "market_research_agent",
        "technical_analysis_agent",
        "risk_assessment_agent",
        "sentiment_analysis_agent",
        "portfolio_optimization_agent",
        "compliance_agent",
        "report_generation_agent"
    ]
    
    for agent in agents:
        builder.add_edge(agent, "human_interaction")
    
    # Add edges from special nodes
    builder.add_edge("human_approval_node", "supervisor")
    builder.add_edge("market_alert_node", "risk_assessment_agent")
    builder.add_edge("supervisor", END)
    
    # Add checkpointing for conversation persistence
    checkpointer = MemorySaver()
    
    # Compile the graph
    return builder.compile(checkpointer=checkpointer)


def create_financial_analysis_session(
    symbols: List[str],
    analysis_type: str = "comprehensive",
    risk_tolerance: str = "moderate"
) -> Dict[str, Any]:
    """
    Create a new financial analysis session
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize state
    initial_state = {
        "target_symbols": symbols,
        "analysis_type": analysis_type,
        "risk_tolerance": risk_tolerance,
        "workflow_id": session_id,
        "analysis_start_time": datetime.now(),
        "market_conditions": MarketConditions(),  # Will be updated by agents
        "messages": []
    }
    
    # Create thread configuration
    thread_config = {
        "configurable": {
            "thread_id": session_id
        }
    }
    
    return {
        "session_id": session_id,
        "initial_state": initial_state,
        "thread_config": thread_config
    }


# Export the main graph
financial_graph = build_financial_analysis_graph()