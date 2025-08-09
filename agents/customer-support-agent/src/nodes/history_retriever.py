"""History retrieval node for customer support agent"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from ..agents.state import AgentState, UserProfile, QueryHistory, ConversationContext
from ..memory.database import DatabaseManager


class HistoryRetriever:
    """Retrieve and manage user history and conversation context"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        self.max_history_items = 10
        self.context_window_days = 30
    
    def retrieve_history(self, state: AgentState) -> AgentState:
        """Main history retrieval function"""
        # Retrieve user profile
        state.user_profile = self._get_user_profile(state.user_id)
        
        # Retrieve query history
        state.query_history = self._get_query_history(state.user_id)
        
        # Update conversation context
        state.conversation_context = self._get_or_create_conversation_context(
            state.thread_id, state.user_id
        )
        
        # Add relevant context to metadata
        state.metadata['history_summary'] = self._create_history_summary(state)
        state.metadata['user_insights'] = self._generate_user_insights(state)
        
        return state
    
    def _get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from database"""
        try:
            profile_data = self.db_manager.get_user_profile(user_id)
            if profile_data:
                return UserProfile(
                    user_id=profile_data.user_id,
                    email=profile_data.email,
                    name=profile_data.name,
                    account_type=profile_data.account_type,
                    created_at=profile_data.created_at,
                    metadata=self._parse_metadata(profile_data.metadata)
                )
        except Exception as e:
            print(f"Error retrieving user profile: {e}")
        
        return None
    
    def _get_query_history(self, user_id: str) -> List[QueryHistory]:
        """Retrieve recent query history for the user"""
        try:
            # Get queries from the last 30 days
            cutoff_date = datetime.now() - timedelta(days=self.context_window_days)
            
            history_data = self.db_manager.get_query_history(
                user_id, 
                limit=self.max_history_items,
                since=cutoff_date
            )
            
            return [
                QueryHistory(
                    query_id=item.query_id,
                    user_id=item.user_id,
                    timestamp=item.timestamp,
                    query_text=item.query_text,
                    category=item.category,
                    resolution=item.resolution,
                    escalated=item.escalated,
                    resolved_by=item.resolved_by,
                    satisfaction_rating=item.satisfaction_rating
                ) for item in history_data
            ]
        except Exception as e:
            print(f"Error retrieving query history: {e}")
            return []
    
    def _get_or_create_conversation_context(self, thread_id: str, user_id: str) -> ConversationContext:
        """Get existing conversation context or create new one"""
        try:
            # Try to get existing context
            context_data = self.db_manager.get_conversation_context(thread_id)
            
            if context_data:
                # Update last activity
                context = ConversationContext(
                    thread_id=context_data.thread_id,
                    session_start=context_data.session_start,
                    last_activity=datetime.now(),
                    message_count=context_data.message_count + 1,
                    context_summary=context_data.context_summary
                )
            else:
                # Create new context
                context = ConversationContext(
                    thread_id=thread_id,
                    session_start=datetime.now(),
                    last_activity=datetime.now(),
                    message_count=1
                )
            
            # Save updated context
            self.db_manager.save_conversation_context(context)
            return context
            
        except Exception as e:
            print(f"Error managing conversation context: {e}")
            # Return default context
            return ConversationContext(
                thread_id=thread_id,
                session_start=datetime.now(),
                last_activity=datetime.now(),
                message_count=1
            )
    
    def _create_history_summary(self, state: AgentState) -> str:
        """Create a summary of the user's query history"""
        if not state.query_history:
            return "No previous interactions found."
        
        # Analyze query patterns
        total_queries = len(state.query_history)
        categories = [q.category for q in state.query_history]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        
        escalated_count = sum(1 for q in state.query_history if q.escalated)
        resolved_count = sum(1 for q in state.query_history if q.resolution)
        
        # Recent query analysis
        recent_queries = sorted(state.query_history, key=lambda x: x.timestamp, reverse=True)[:3]
        recent_categories = [q.category for q in recent_queries]
        
        # Build summary
        summary_parts = [
            f"User has {total_queries} previous interactions in the last {self.context_window_days} days."
        ]
        
        if category_counts:
            most_common_category = max(category_counts, key=category_counts.get)
            summary_parts.append(f"Most common query type: {most_common_category} ({category_counts[most_common_category]} queries).")
        
        if escalated_count > 0:
            summary_parts.append(f"{escalated_count} queries required human intervention.")
        
        if resolved_count > 0:
            resolution_rate = (resolved_count / total_queries) * 100
            summary_parts.append(f"Resolution rate: {resolution_rate:.1f}%.")
        
        if recent_queries:
            summary_parts.append(f"Recent query topics: {', '.join(set(recent_categories))}.")
        
        return " ".join(summary_parts)
    
    def _generate_user_insights(self, state: AgentState) -> Dict[str, Any]:
        """Generate insights about the user based on their history"""
        insights = {
            'is_new_user': len(state.query_history) == 0,
            'is_frequent_user': len(state.query_history) > 5,
            'needs_escalation_likely': False,
            'preferred_resolution_time': 'standard',
            'communication_style': 'formal',
            'satisfaction_trend': 'neutral'
        }
        
        if not state.query_history:
            return insights
        
        # Analyze escalation patterns
        recent_escalations = sum(1 for q in state.query_history[-5:] if q.escalated)
        if recent_escalations >= 2:
            insights['needs_escalation_likely'] = True
        
        # Analyze account type for priority
        if state.user_profile and state.user_profile.account_type in ['premium', 'enterprise']:
            insights['preferred_resolution_time'] = 'priority'
        
        # Analyze satisfaction ratings
        ratings = [q.satisfaction_rating for q in state.query_history if q.satisfaction_rating]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            if avg_rating >= 4:
                insights['satisfaction_trend'] = 'positive'
            elif avg_rating <= 2:
                insights['satisfaction_trend'] = 'negative'
            else:
                insights['satisfaction_trend'] = 'neutral'
        
        # Analyze query complexity patterns
        complex_categories = ['technical', 'feature']
        complex_queries = sum(1 for q in state.query_history if q.category in complex_categories)
        if complex_queries > len(state.query_history) * 0.6:
            insights['user_type'] = 'technical'
        else:
            insights['user_type'] = 'general'
        
        return insights
    
    def _parse_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Parse metadata string to dictionary"""
        if not metadata_str:
            return {}
        
        try:
            import json
            return json.loads(metadata_str)
        except Exception:
            return {}
    
    def get_relevant_context(self, state: AgentState, current_query: str) -> List[str]:
        """Get relevant context from history based on current query"""
        if not state.query_history:
            return []
        
        relevant_context = []
        current_query_lower = current_query.lower()
        
        # Find similar queries
        for query in state.query_history:
            query_lower = query.query_text.lower()
            
            # Simple keyword matching (could be enhanced with embeddings)
            common_words = set(current_query_lower.split()) & set(query_lower.split())
            if len(common_words) >= 2:  # At least 2 common words
                if query.resolution:
                    relevant_context.append(
                        f"Similar previous query: '{query.query_text}' "
                        f"was resolved with: '{query.resolution}'"
                    )
        
        # Find queries in the same category
        current_category = state.query_category
        if current_category:
            category_queries = [q for q in state.query_history if q.category == current_category]
            if category_queries:
                successful_resolutions = [q for q in category_queries if q.resolution and not q.escalated]
                if successful_resolutions:
                    latest_resolution = max(successful_resolutions, key=lambda x: x.timestamp)
                    relevant_context.append(
                        f"Previous {current_category} query was resolved: '{latest_resolution.resolution}'"
                    )
        
        return relevant_context[:3]  # Limit to top 3 relevant items
    
    def update_conversation_stats(self, state: AgentState) -> AgentState:
        """Update conversation statistics"""
        if state.conversation_context:
            state.conversation_context.last_activity = datetime.now()
            state.conversation_context.message_count = len(state.messages)
            
            # Save updated context
            try:
                self.db_manager.save_conversation_context(state.conversation_context)
            except Exception as e:
                print(f"Error updating conversation stats: {e}")
        
        return state


# Function to use in LangGraph node
def retrieve_history_node(state: AgentState) -> AgentState:
    """LangGraph node function for history retrieval"""
    retriever = HistoryRetriever()
    updated_state = retriever.retrieve_history(state)
    return retriever.update_conversation_stats(updated_state)
