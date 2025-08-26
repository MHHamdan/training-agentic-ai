"""
Main LangGraph Implementation for Multi-Agent Content Creation System
Following patterns from financial_graph.py for consistency
"""

from typing import Dict, List, Any, Literal
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import uuid

from content_state import ContentCreationState, ContentAlert, BrandGuidelines
from content_agents import (
    call_topic_research_agent,
    call_content_strategist_agent,
    call_content_writer_agent,
    call_seo_specialist_agent,
    call_quality_assurance_agent,
    call_content_editor_agent,
    call_content_publisher_agent
)


def route_initial_content_request(state: ContentCreationState) -> str:
    """
    Route initial content creation request to appropriate starting agent based on request analysis
    """
    if not state.get("messages"):
        return "topic_research_agent"  # Default start
    
    user_query = state["messages"][-1].content.lower() if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1]).lower()
    content_type = state.get("content_type", "blog_post")
    
    # Analyze query to determine best starting point
    routing_keywords = {
        "topic_research_agent": ["research", "trends", "keywords", "competitors", "market", "analysis", "data"],
        "content_strategist_agent": ["strategy", "plan", "outline", "structure", "approach", "format"],
        "content_writer_agent": ["write", "create", "draft", "content", "article", "copy"],
        "seo_specialist_agent": ["seo", "optimize", "ranking", "search", "meta", "keywords"],
        "quality_assurance_agent": ["review", "check", "quality", "compliance", "brand", "guidelines"]
    }
    
    # Content type specific routing
    if content_type in ["blog_post", "white_paper", "case_study"]:
        # Complex content starts with research
        if any(keyword in user_query for keyword in ["existing", "have content", "review"]):
            return "quality_assurance_agent"
        else:
            return "topic_research_agent"
    elif content_type in ["social_media", "email_campaign"]:
        # Short content can start with strategy
        return "content_strategist_agent"
    elif content_type in ["product_description", "website_copy"]:
        # Marketing copy benefits from SEO focus
        return "seo_specialist_agent" if "seo" in user_query else "content_strategist_agent"
    
    # Count keyword matches for each agent
    scores = {}
    for agent, keywords in routing_keywords.items():
        score = sum(1 for keyword in keywords if keyword in user_query)
        if score > 0:
            scores[agent] = score
    
    # Return agent with highest score, or default to research
    if scores:
        best_agent = max(scores.keys(), key=lambda k: scores[k])
        return best_agent
    
    return "topic_research_agent"


def route_based_on_content_type(state: ContentCreationState) -> str:
    """
    Route based on content type and current workflow state
    """
    content_type = state.get("content_type", "blog_post")
    last_agent = state.get("last_active_agent", "")
    
    # Content type specific workflows
    content_workflows = {
        "blog_post": [
            "topic_research_agent",
            "content_strategist_agent", 
            "content_writer_agent",
            "seo_specialist_agent",
            "quality_assurance_agent",
            "content_editor_agent",
            "content_publisher_agent"
        ],
        "social_media": [
            "content_strategist_agent",
            "content_writer_agent", 
            "quality_assurance_agent",
            "content_publisher_agent"
        ],
        "website_copy": [
            "seo_specialist_agent",
            "content_strategist_agent",
            "content_writer_agent",
            "quality_assurance_agent",
            "content_editor_agent",
            "content_publisher_agent"
        ],
        "white_paper": [
            "topic_research_agent",
            "content_strategist_agent",
            "content_writer_agent",
            "quality_assurance_agent",
            "content_editor_agent",
            "content_publisher_agent"
        ]
    }
    
    workflow = content_workflows.get(content_type, content_workflows["blog_post"])
    
    if last_agent in workflow:
        current_index = workflow.index(last_agent)
        if current_index + 1 < len(workflow):
            return workflow[current_index + 1]
    
    return "content_publisher_agent"


def route_next_agent(state: ContentCreationState) -> str:
    """
    Determine the next agent in the workflow based on current state
    """
    last_agent = state.get("last_active_agent", "")
    content_type = state.get("content_type", "blog_post")
    
    # Check if we need human approval (temporarily disabled)
    # if state.get("approval_required", False):
    #     return "human_approval_node"
    
    # Check for quality alerts that need immediate attention
    alerts = state.get("content_alerts", [])
    critical_alerts = [alert for alert in alerts if alert.get("severity") == "critical"]
    if critical_alerts:
        return "content_alert_node"
    
    # Standard workflow routing
    workflow_map = {
        "topic_research_agent": "content_strategist_agent",
        "content_strategist_agent": "content_writer_agent", 
        "content_writer_agent": "seo_specialist_agent",
        "seo_specialist_agent": "quality_assurance_agent",
        "quality_assurance_agent": "content_editor_agent",
        "content_editor_agent": "content_publisher_agent",
        "content_publisher_agent": "__end__"
    }
    
    # Content type specific modifications
    if content_type == "social_media":
        workflow_map.update({
            "content_writer_agent": "quality_assurance_agent",
            "quality_assurance_agent": "content_publisher_agent"
        })
    elif content_type == "white_paper":
        workflow_map.update({
            "quality_assurance_agent": "content_editor_agent",
            "content_editor_agent": "content_publisher_agent"  # Skip approval for now, go directly to publisher
        })
    
    next_agent = workflow_map.get(last_agent, "content_publisher_agent")
    return next_agent


def human_interaction_node(state: ContentCreationState) -> Command:
    """
    Handle human interaction and workflow continuation
    """
    last_agent = state.get("last_active_agent", "")
    completed_analyses = state.get("completed_analyses", {})
    
    # Create summary of completed work
    summary_message = f"""
## Content Creation Progress Summary

**Last Active Agent:** {last_agent.replace('_', ' ').title()}
**Completed Analyses:** {len(completed_analyses)}
**Content Type:** {state.get('content_type', 'Unknown')}
**Topic:** {state.get('content_topic', 'Not specified')}

### Completed Steps:
{chr(10).join([f"✅ {name.replace('_', ' ').title()}" for name in completed_analyses.keys()])}

### Current Status:
- Research Completed: {'✅' if state.get('research_completed') else '❌'}
- Strategy Completed: {'✅' if state.get('strategy_completed') else '❌'}
- Content Drafted: {'✅' if state.get('content_drafted') else '❌'}
- SEO Optimized: {'✅' if state.get('seo_optimized') else '❌'}
- Quality Checked: {'✅' if state.get('quality_checked') else '❌'}

**Next:** Continue workflow or provide feedback for adjustments.
"""
    
    # Check if workflow is complete
    if state.get("creation_end_time") or last_agent == "content_publisher_agent":
        summary_message += "\n\n🎉 **Content creation workflow completed!**"
        return Command(update={"workflow_status": "completed"})
    
    # Continue workflow
    next_agent = route_next_agent(state)
    if next_agent == "human_interaction" or next_agent == "__end__":
        return Command(update={"workflow_status": "awaiting_input"})
    
    return Command(goto=next_agent)


def human_approval_node(state: ContentCreationState) -> Command:
    """
    Handle human approval workflow for content publication
    """
    # Get approval items
    approval_items = []
    if state.get("content_drafted"):
        approval_items.append("Content Draft")
    if state.get("seo_optimized"):
        approval_items.append("SEO Optimization")
    if state.get("quality_checked"):
        approval_items.append("Quality Assurance")
    
    approval_message = f"""
## Content Approval Required

The following content elements are ready for review:

{chr(10).join([f"📋 {item}" for item in approval_items])}

**Content Type:** {state.get('content_type', 'Unknown')}
**Topic:** {state.get('content_topic', 'Not specified')}
**Target Keywords:** {', '.join(state.get('target_keywords', []))}

Please review the content and respond with:
- 'approve' to proceed with publication
- 'reject' to send back for revisions
- 'modify [instructions]' to request specific changes

**Quality Scores:**
- SEO Score: {state.get('seo_score', 'Not calculated')}
- Readability Score: {state.get('readability_score', 'Not calculated')}
"""
    
    # Wait for human decision
    try:
        user_response = interrupt(value=approval_message)
        
        if user_response:
            decision_lower = user_response.lower()
            
            if "approve" in decision_lower:
                # Approved - continue to publication
                update = {
                    "approval_status": "approved",
                    "approval_required": False
                }
                
                # Log approval
                audit_log = state.get("audit_log", [])
                audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "content_approved",
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
                # Rejected - send back for revisions
                update = {
                    "approval_status": "rejected",
                    "approval_required": False
                }
                
                # Log rejection
                audit_log = state.get("audit_log", [])
                audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "content_rejected",
                    "details": {
                        "items": approval_items,
                        "timestamp": datetime.now().isoformat()
                    },
                    "agent": "human_approval_node"
                })
                update["audit_log"] = audit_log
                
                # Route back to content editor for revisions
                return Command(update=update, goto="content_editor_agent")
            
            elif "modify" in decision_lower:
                # Modification requested
                modification_request = user_response.replace("modify", "").strip()
                update = {
                    "approval_status": "modifications_requested",
                    "modification_request": modification_request,
                    "approval_required": False
                }
                
                return Command(update=update, goto="content_editor_agent")
    
    except Exception:
        # Handle interrupt timeout or error
        pass
    
    # Default: continue without approval
    return Command(update={"approval_required": False}, goto=route_next_agent(state))


def content_alert_node(state: ContentCreationState) -> Command:
    """
    Handle critical content alerts that require immediate attention
    """
    alerts = state.get("content_alerts", [])
    critical_alerts = [alert for alert in alerts if alert.get("severity") in ["critical", "high"]]
    
    if not critical_alerts:
        # No critical alerts, continue normal flow
        return Command(goto=route_next_agent(state))
    
    # Format alert messages
    alert_messages = []
    for alert in critical_alerts:
        alert_messages.append(f"""
🚨 CONTENT ALERT: {alert.get('message', 'Unknown issue')}
Source: {alert.get('source_agent', 'Unknown')}
Content Type: {alert.get('content_type', 'Unknown')}
Recommended Action: {alert.get('recommended_action', 'Review immediately')}
""")
    
    full_message = "\n".join(alert_messages) + "\n\nPress Enter to acknowledge and continue with content review."
    
    # Wait for user acknowledgment
    try:
        user_response = interrupt(value=full_message)
    except Exception:
        pass
    
    # Update state
    update = {
        "alert_acknowledged": True,
        "critical_alert_active": False
    }
    
    # Log alert acknowledgment
    audit_log = state.get("audit_log", [])
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "event_type": "content_alert_acknowledged",
        "details": {
            "alerts": [alert for alert in critical_alerts],
            "timestamp": datetime.now().isoformat()
        },
        "agent": "content_alert_node"
    })
    update["audit_log"] = audit_log
    
    # Route to quality assurance for immediate review
    return Command(update=update, goto="quality_assurance_agent")


def supervisor_node(state: ContentCreationState) -> Command:
    """
    Supervisor node for workflow coordination and decision making
    """
    # Analyze current state and determine best next action
    completed_analyses = state.get("completed_analyses", {})
    
    # Check workflow completeness
    required_steps = ["topic_research", "content_strategy", "content_creation", "seo_optimization", "quality_assurance"]
    completed_steps = [step for step in required_steps if any(step in analysis for analysis in completed_analyses.keys())]
    
    if len(completed_steps) >= 4:  # Most steps complete
        return Command(goto="content_publisher_agent")
    elif len(completed_steps) >= 2:  # Mid-workflow
        return Command(goto="quality_assurance_agent")
    else:  # Early workflow
        return Command(goto="content_strategist_agent")


# Build the content creation graph
def build_content_creation_graph():
    """Construct the complete content creation LangGraph"""
    
    builder = StateGraph(ContentCreationState)
    
    # Add all content creation agent nodes
    builder.add_node("topic_research_agent", call_topic_research_agent)
    builder.add_node("content_strategist_agent", call_content_strategist_agent)
    builder.add_node("content_writer_agent", call_content_writer_agent)
    builder.add_node("seo_specialist_agent", call_seo_specialist_agent)
    builder.add_node("quality_assurance_agent", call_quality_assurance_agent)
    builder.add_node("content_editor_agent", call_content_editor_agent)
    builder.add_node("content_publisher_agent", call_content_publisher_agent)
    builder.add_node("human_interaction", human_interaction_node)
    builder.add_node("human_approval_node", human_approval_node)
    builder.add_node("content_alert_node", content_alert_node)
    builder.add_node("supervisor", supervisor_node)
    
    # Add conditional entry points
    builder.add_conditional_edges(
        START,
        route_initial_content_request,
        {
            "topic_research_agent": "topic_research_agent",
            "content_strategist_agent": "content_strategist_agent",
            "content_writer_agent": "content_writer_agent",
            "seo_specialist_agent": "seo_specialist_agent",
            "quality_assurance_agent": "quality_assurance_agent"
        }
    )
    
    # Add workflow routing edges
    for agent in ["topic_research_agent", "content_strategist_agent", "content_writer_agent", 
                  "seo_specialist_agent", "quality_assurance_agent", "content_editor_agent",
                  "content_publisher_agent"]:
        builder.add_conditional_edges(
            agent,
            route_next_agent,
            {
                "topic_research_agent": "topic_research_agent",
                "content_strategist_agent": "content_strategist_agent",
                "content_writer_agent": "content_writer_agent", 
                "seo_specialist_agent": "seo_specialist_agent",
                "quality_assurance_agent": "quality_assurance_agent",
                "content_editor_agent": "content_editor_agent",
                "content_publisher_agent": "content_publisher_agent",
                "human_interaction": "human_interaction",
                "human_approval_node": "human_approval_node",
                "content_alert_node": "content_alert_node",
                "supervisor": "supervisor",
                "__end__": END
            }
        )
    
    # Add human interaction routing
    builder.add_conditional_edges(
        "human_interaction",
        route_next_agent,
        {
            "topic_research_agent": "topic_research_agent",
            "content_strategist_agent": "content_strategist_agent",
            "content_writer_agent": "content_writer_agent",
            "seo_specialist_agent": "seo_specialist_agent", 
            "quality_assurance_agent": "quality_assurance_agent",
            "content_editor_agent": "content_editor_agent",
            "content_publisher_agent": "content_publisher_agent",
            "human_approval_node": "human_approval_node",
            "content_alert_node": "content_alert_node",
            "supervisor": "supervisor",
            "__end__": END
        }
    )
    
    # Add approval node routing
    builder.add_conditional_edges(
        "human_approval_node",
        route_next_agent,
        {
            "content_editor_agent": "content_editor_agent",
            "content_publisher_agent": "content_publisher_agent",
            "quality_assurance_agent": "quality_assurance_agent",
            "__end__": END
        }
    )
    
    # Add alert node routing
    builder.add_conditional_edges(
        "content_alert_node",
        route_next_agent,
        {
            "quality_assurance_agent": "quality_assurance_agent",
            "content_editor_agent": "content_editor_agent",
            "supervisor": "supervisor"
        }
    )
    
    # Add supervisor routing
    builder.add_conditional_edges(
        "supervisor",
        route_next_agent,
        {
            "content_strategist_agent": "content_strategist_agent",
            "quality_assurance_agent": "quality_assurance_agent", 
            "content_publisher_agent": "content_publisher_agent",
            "__end__": END
        }
    )
    
    # Set up checkpointing for state persistence
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def create_content_creation_session(
    topic: str,
    content_type: str = "blog_post",
    target_keywords: List[str] = None,
    brand_guidelines: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a new content creation session
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    # Initialize state
    initial_state = {
        "content_topic": topic,
        "content_type": content_type,
        "target_keywords": target_keywords or [],
        "brand_guidelines": BrandGuidelines(**brand_guidelines) if brand_guidelines else None,
        "creation_start_time": datetime.now(),
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


# Create the compiled graph instance
content_creation_graph = build_content_creation_graph()