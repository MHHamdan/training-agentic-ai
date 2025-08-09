"""Escalation node for Human-in-the-Loop (HITL) functionality"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..agents.state import AgentState, EscalationInfo, StateConstants


class EscalationManager:
    """Manage escalation to human agents"""
    
    def __init__(self):
        self.escalation_thresholds = {
            'complexity_score': 0.7,
            'confidence_threshold': 0.5,
            'max_auto_attempts': 3,
            'user_frustration_keywords': [
                'frustrated', 'angry', 'terrible', 'awful', 'horrible',
                'speak to human', 'human agent', 'manager', 'supervisor'
            ]
        }
        
        self.escalation_reasons = {
            'complexity': "Query complexity exceeds AI capabilities",
            'low_confidence': "AI confidence level below threshold",
            'user_request': "User explicitly requested human assistance",
            'repeated_failures': "Multiple failed resolution attempts",
            'account_type': "Premium/Enterprise customer priority",
            'technical_issue': "Complex technical issue requiring human expertise",
            'billing_dispute': "Billing dispute requiring human verification",
            'security_concern': "Security-related query requiring human review"
        }
    
    def check_escalation_needed(self, state: AgentState) -> AgentState:
        """Check if escalation to human agent is needed"""
        escalation_needed = False
        escalation_reason = None
        urgency_level = "medium"
        
        # Check complexity score
        complexity_score = state.metadata.get('complexity_score', 0.0)
        if complexity_score >= self.escalation_thresholds['complexity_score']:
            escalation_needed = True
            escalation_reason = self.escalation_reasons['complexity']
            urgency_level = "high" if complexity_score > 0.8 else "medium"
        
        # Check for explicit user request for human
        if state.current_query:
            query_lower = state.current_query.lower()
            for keyword in self.escalation_thresholds['user_frustration_keywords']:
                if keyword in query_lower:
                    escalation_needed = True
                    escalation_reason = self.escalation_reasons['user_request']
                    urgency_level = "high"
                    break
        
        # Check user history for repeated issues
        if state.query_history:
            recent_escalations = [q for q in state.query_history[-5:] if q.escalated]
            if len(recent_escalations) >= 2:
                escalation_needed = True
                escalation_reason = self.escalation_reasons['repeated_failures']
                urgency_level = "high"
        
        # Check account type priority
        if (state.user_profile and 
            state.user_profile.account_type in ['premium', 'enterprise']):
            # Lower threshold for premium customers
            if complexity_score >= 0.5:
                escalation_needed = True
                escalation_reason = self.escalation_reasons['account_type']
                urgency_level = "high"
        
        # Check for specific categories that often need human intervention
        high_escalation_categories = ['billing', 'account']
        if (state.query_category in high_escalation_categories and 
            complexity_score >= 0.5):
            escalation_needed = True
            escalation_reason = self.escalation_reasons[
                'billing_dispute' if state.query_category == 'billing' 
                else 'technical_issue'
            ]
        
        # Check conversation context for signs of frustration
        if state.conversation_context:
            if (state.conversation_context.message_count > 5 and 
                not any('resolved' in msg.get('content', '').lower() 
                       for msg in state.messages[-3:])):
                escalation_needed = True
                escalation_reason = self.escalation_reasons['repeated_failures']
                urgency_level = "high"
        
        # Update state
        state.requires_human = escalation_needed
        
        if escalation_needed:
            state.escalation_info = self._create_escalation_info(
                escalation_reason, urgency_level, state
            )
        
        return state
    
    def _create_escalation_info(self, reason: str, urgency: str, state: AgentState) -> EscalationInfo:
        """Create escalation information"""
        # Estimate resolution time based on urgency and account type
        resolution_time_hours = {
            'critical': 1,
            'high': 4,
            'medium': 24,
            'low': 72
        }
        
        hours_to_resolve = resolution_time_hours.get(urgency, 24)
        
        # Adjust for account type
        if state.user_profile and state.user_profile.account_type == 'enterprise':
            hours_to_resolve = max(1, hours_to_resolve // 2)
        elif state.user_profile and state.user_profile.account_type == 'premium':
            hours_to_resolve = max(2, int(hours_to_resolve * 0.75))
        
        estimated_resolution = datetime.now() + timedelta(hours=hours_to_resolve)
        
        return EscalationInfo(
            reason=reason,
            urgency_level=urgency,
            estimated_resolution_time=estimated_resolution
        )
    
    def create_escalation_ticket(self, state: AgentState) -> Dict[str, Any]:
        """Create a detailed escalation ticket for human agents"""
        if not state.escalation_info:
            return {}
        
        # Gather context for human agent
        context_summary = self._prepare_context_for_human(state)
        
        ticket = {
            'escalation_id': state.escalation_info.escalation_id,
            'timestamp': state.escalation_info.escalated_at.isoformat(),
            'user_info': {
                'user_id': state.user_id,
                'name': state.user_profile.name if state.user_profile else 'Unknown',
                'email': state.user_profile.email if state.user_profile else 'Unknown',
                'account_type': state.user_profile.account_type if state.user_profile else 'standard'
            },
            'query_info': {
                'current_query': state.current_query,
                'category': state.query_category,
                'complexity_score': state.metadata.get('complexity_score', 0.0),
                'entities': state.metadata.get('entities', {}),
                'priority': state.conversation_context.priority if state.conversation_context else 'normal'
            },
            'escalation_details': {
                'reason': state.escalation_info.reason,
                'urgency_level': state.escalation_info.urgency_level,
                'estimated_resolution': state.escalation_info.estimated_resolution_time.isoformat(),
                'customer_notified': state.escalation_info.customer_notified
            },
            'conversation_history': self._prepare_conversation_history(state),
            'user_history_summary': state.metadata.get('history_summary', ''),
            'user_insights': state.metadata.get('user_insights', {}),
            'context_summary': context_summary,
            'suggested_actions': self._get_suggested_actions(state)
        }
        
        return ticket
    
    def _prepare_context_for_human(self, state: AgentState) -> str:
        """Prepare context summary for human agent"""
        context_parts = []
        
        # User background
        if state.user_profile:
            context_parts.append(
                f"User: {state.user_profile.name} ({state.user_profile.account_type} account)"
            )
        
        # Current situation
        context_parts.append(f"Current query: {state.current_query}")
        context_parts.append(f"Category: {state.query_category}")
        
        # Complexity indicators
        complexity_score = state.metadata.get('complexity_score', 0.0)
        context_parts.append(f"Complexity score: {complexity_score:.2f}")
        
        # Previous attempts
        if state.query_history:
            recent_queries = [q for q in state.query_history[-3:]]
            if recent_queries:
                context_parts.append(
                    f"Recent queries: {len(recent_queries)} in last interactions"
                )
        
        # Escalation reason
        if state.escalation_info:
            context_parts.append(f"Escalation reason: {state.escalation_info.reason}")
        
        return "; ".join(context_parts)
    
    def _prepare_conversation_history(self, state: AgentState) -> List[Dict[str, Any]]:
        """Prepare conversation history for human agent"""
        # Get last 10 messages for context
        recent_messages = state.messages[-10:] if len(state.messages) > 10 else state.messages
        
        formatted_history = []
        for msg in recent_messages:
            formatted_msg = {
                'role': msg.get('role', 'unknown'),
                'content': msg.get('content', ''),
                'timestamp': msg.get('timestamp', datetime.now().isoformat()),
                'metadata': msg.get('metadata', {})
            }
            formatted_history.append(formatted_msg)
        
        return formatted_history
    
    def _get_suggested_actions(self, state: AgentState) -> List[str]:
        """Get suggested actions for human agent based on context"""
        suggestions = []
        
        # Category-specific suggestions
        if state.query_category == 'billing':
            suggestions.extend([
                "Verify account billing information",
                "Check recent payment history",
                "Review subscription status"
            ])
        elif state.query_category == 'technical':
            suggestions.extend([
                "Check system logs for errors",
                "Verify user permissions",
                "Test user's configuration"
            ])
        elif state.query_category == 'account':
            suggestions.extend([
                "Verify user identity",
                "Check account security settings",
                "Review access permissions"
            ])
        
        # Urgency-specific suggestions
        if state.escalation_info and state.escalation_info.urgency_level in ['high', 'critical']:
            suggestions.insert(0, "Priority handling required - respond within 1 hour")
        
        # Account type specific suggestions
        if state.user_profile and state.user_profile.account_type in ['premium', 'enterprise']:
            suggestions.insert(0, "Premium customer - expedited handling")
        
        # History-based suggestions
        if state.query_history:
            escalated_queries = [q for q in state.query_history if q.escalated]
            if len(escalated_queries) > 1:
                suggestions.append("Multiple escalations - consider account review")
        
        return suggestions
    
    def send_escalation_notification(self, state: AgentState) -> AgentState:
        """Send notification about escalation to customer"""
        if not state.escalation_info:
            return state
        
        # Create customer notification message
        notification_message = self._create_customer_notification(state)
        
        # Add notification to conversation
        notification_msg = {
            'role': StateConstants.ROLE_SYSTEM,
            'content': notification_message,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'escalation_notice': True,
                'escalation_id': state.escalation_info.escalation_id,
                'estimated_resolution': state.escalation_info.estimated_resolution_time.isoformat()
            }
        }
        
        state.messages.append(notification_msg)
        
        # Mark customer as notified
        state.escalation_info.customer_notified = True
        
        return state
    
    def _create_customer_notification(self, state: AgentState) -> str:
        """Create customer notification message"""
        urgency = state.escalation_info.urgency_level
        estimated_time = state.escalation_info.estimated_resolution_time
        
        # Format the estimated time
        now = datetime.now()
        time_diff = estimated_time - now
        
        if time_diff.total_seconds() <= 3600:  # Less than 1 hour
            time_str = "within the next hour"
        elif time_diff.total_seconds() <= 86400:  # Less than 24 hours
            hours = int(time_diff.total_seconds() // 3600)
            time_str = f"within {hours} hour{'s' if hours > 1 else ''}"
        else:
            days = int(time_diff.total_seconds() // 86400)
            time_str = f"within {days} day{'s' if days > 1 else ''}"
        
        # Base message
        message = "I've escalated your query to one of our human agents who will be better equipped to help you."
        
        # Add urgency-specific information
        if urgency in ['high', 'critical']:
            message += " Due to the nature of your request, this has been marked as high priority."
        
        # Add time estimate
        message += f" You can expect a response {time_str}."
        
        # Add account-specific information
        if state.user_profile and state.user_profile.account_type in ['premium', 'enterprise']:
            message += " As a valued premium customer, your query will receive expedited handling."
        
        # Add reference information
        message += f" Your escalation reference ID is: {state.escalation_info.escalation_id}"
        
        return message


# Function to use in LangGraph node
def check_escalation_node(state: AgentState) -> AgentState:
    """LangGraph node function for escalation checking"""
    manager = EscalationManager()
    updated_state = manager.check_escalation_needed(state)
    
    if updated_state.requires_human:
        # Create escalation ticket (would be sent to human agent system)
        ticket = manager.create_escalation_ticket(updated_state)
        updated_state.metadata['escalation_ticket'] = ticket
        
        # Send notification to customer
        updated_state = manager.send_escalation_notification(updated_state)
    
    return updated_state
