"""Long-term memory management for customer support agent"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from .database import DatabaseManager
from .short_term import ShortTermMemory
from ..agents.state import UserProfile, QueryHistory, ConversationContext


class LongTermMemory:
    """Manage long-term persistent memory for the customer support agent"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, 
                 cache_manager: Optional[ShortTermMemory] = None):
        """Initialize long-term memory manager"""
        self.db_manager = db_manager or DatabaseManager()
        self.cache_manager = cache_manager or ShortTermMemory()
        self.retention_days = 365  # Data retention period
    
    # User Profile Management
    def store_user_profile(self, profile: UserProfile) -> bool:
        """Store user profile with caching"""
        try:
            # Save to database
            success = self.db_manager.save_user_profile(profile)
            
            if success:
                # Cache for quick access
                profile_data = {
                    'user_id': profile.user_id,
                    'email': profile.email,
                    'name': profile.name,
                    'account_type': profile.account_type,
                    'created_at': profile.created_at.isoformat(),
                    'metadata': profile.metadata
                }
                self.cache_manager.cache_user_state(profile.user_id, profile_data)
            
            return success
        except Exception as e:
            print(f"Error storing user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile with cache fallback"""
        try:
            # Try cache first
            cached_profile = self.cache_manager.get_cached_user_state(user_id)
            if cached_profile:
                return UserProfile(
                    user_id=cached_profile['user_id'],
                    email=cached_profile['email'],
                    name=cached_profile['name'],
                    account_type=cached_profile['account_type'],
                    created_at=datetime.fromisoformat(cached_profile['created_at']),
                    metadata=cached_profile.get('metadata', {})
                )
            
            # Fallback to database
            db_profile = self.db_manager.get_user_profile(user_id)
            if db_profile:
                profile = UserProfile(
                    user_id=db_profile.user_id,
                    email=db_profile.email,
                    name=db_profile.name,
                    account_type=db_profile.account_type,
                    created_at=db_profile.created_at,
                    metadata=self._parse_json_metadata(db_profile.metadata)
                )
                
                # Cache for future use
                self.store_user_profile(profile)
                return profile
            
            return None
        except Exception as e:
            print(f"Error retrieving user profile: {e}")
            return None
    
    def update_user_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> bool:
        """Update user's last interaction and relevant metadata"""
        try:
            # Update database
            success = self.db_manager.update_user_last_interaction(user_id)
            
            if success:
                # Update cache if profile exists
                cached_profile = self.cache_manager.get_cached_user_state(user_id)
                if cached_profile:
                    cached_profile['last_interaction'] = datetime.now().isoformat()
                    cached_profile['metadata'].update(interaction_data)
                    self.cache_manager.cache_user_state(user_id, cached_profile)
            
            return success
        except Exception as e:
            print(f"Error updating user interaction: {e}")
            return False
    
    # Query History Management
    def store_query_history(self, query: QueryHistory) -> bool:
        """Store query history entry"""
        try:
            return self.db_manager.save_query_history(query)
        except Exception as e:
            print(f"Error storing query history: {e}")
            return False
    
    def get_user_query_history(self, user_id: str, limit: int = 10, 
                               days: int = 30) -> List[QueryHistory]:
        """Get user's query history with intelligent filtering"""
        try:
            since_date = datetime.now() - timedelta(days=days)
            db_queries = self.db_manager.get_query_history(user_id, limit, since_date)
            
            return [
                QueryHistory(
                    query_id=q.query_id,
                    user_id=q.user_id,
                    timestamp=q.timestamp,
                    query_text=q.query_text,
                    category=q.category,
                    resolution=q.resolution,
                    escalated=q.escalated,
                    resolved_by=q.resolved_by,
                    satisfaction_rating=q.satisfaction_rating,
                    response_time_seconds=q.response_time_seconds
                ) for q in db_queries
            ]
        except Exception as e:
            print(f"Error retrieving query history: {e}")
            return []
    
    def get_category_trends(self, user_id: str, days: int = 90) -> Dict[str, Any]:
        """Analyze query category trends for a user"""
        try:
            all_queries = self.get_user_query_history(user_id, limit=100, days=days)
            
            if not all_queries:
                return {'trends': {}, 'most_common': None, 'escalation_patterns': {}}
            
            # Category frequency
            category_counts = {}
            escalation_by_category = {}
            resolution_times = {}
            
            for query in all_queries:
                category = query.category or 'general'
                category_counts[category] = category_counts.get(category, 0) + 1
                
                if query.escalated:
                    escalation_by_category[category] = escalation_by_category.get(category, 0) + 1
                
                if query.response_time_seconds:
                    if category not in resolution_times:
                        resolution_times[category] = []
                    resolution_times[category].append(query.response_time_seconds)
            
            # Calculate averages
            avg_resolution_times = {}
            for category, times in resolution_times.items():
                avg_resolution_times[category] = sum(times) / len(times)
            
            most_common = max(category_counts, key=category_counts.get) if category_counts else None
            
            return {
                'trends': category_counts,
                'most_common': most_common,
                'escalation_patterns': escalation_by_category,
                'average_resolution_times': avg_resolution_times,
                'total_queries': len(all_queries)
            }
        except Exception as e:
            print(f"Error analyzing category trends: {e}")
            return {}
    
    # Conversation Context Management
    def store_conversation_context(self, context: ConversationContext, user_id: str) -> bool:
        """Store conversation context"""
        try:
            # Add user_id to context for database storage
            context_dict = context.dict()
            context_dict['user_id'] = user_id
            
            success = self.db_manager.save_conversation_context(context)
            
            if success:
                # Cache the context
                self.cache_manager.cache_conversation_context(context.thread_id, context)
            
            return success
        except Exception as e:
            print(f"Error storing conversation context: {e}")
            return False
    
    def get_conversation_context(self, thread_id: str) -> Optional[ConversationContext]:
        """Get conversation context with cache fallback"""
        try:
            # Try cache first
            cached_context = self.cache_manager.get_cached_context(thread_id)
            if cached_context:
                return ConversationContext(
                    thread_id=cached_context['thread_id'],
                    session_start=datetime.fromisoformat(cached_context['session_start']),
                    last_activity=datetime.fromisoformat(cached_context['last_activity']),
                    message_count=cached_context['message_count'],
                    context_summary=cached_context.get('context_summary'),
                    sentiment=cached_context.get('sentiment'),
                    priority=cached_context.get('priority', 'normal')
                )
            
            # Fallback to database
            db_context = self.db_manager.get_conversation_context(thread_id)
            if db_context:
                context = ConversationContext(
                    thread_id=db_context.thread_id,
                    session_start=db_context.session_start,
                    last_activity=db_context.last_activity,
                    message_count=db_context.message_count,
                    context_summary=db_context.context_summary,
                    sentiment=db_context.sentiment,
                    priority=db_context.priority or 'normal'
                )
                
                # Cache for future use
                self.cache_manager.cache_conversation_context(thread_id, context)
                return context
            
            return None
        except Exception as e:
            print(f"Error retrieving conversation context: {e}")
            return None
    
    # Pattern Recognition and Learning
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze user behavior patterns for personalization"""
        try:
            query_history = self.get_user_query_history(user_id, limit=50, days=180)
            
            if not query_history:
                return {'patterns': {}, 'insights': {}, 'recommendations': []}
            
            # Time patterns
            time_patterns = self._analyze_time_patterns(query_history)
            
            # Category preferences
            category_analysis = self.get_category_trends(user_id, days=180)
            
            # Escalation patterns
            escalation_analysis = self._analyze_escalation_patterns(query_history)
            
            # Response satisfaction
            satisfaction_analysis = self._analyze_satisfaction_patterns(query_history)
            
            # Generate insights
            insights = self._generate_user_insights(
                time_patterns, category_analysis, escalation_analysis, satisfaction_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(insights)
            
            return {
                'patterns': {
                    'time': time_patterns,
                    'categories': category_analysis,
                    'escalations': escalation_analysis,
                    'satisfaction': satisfaction_analysis
                },
                'insights': insights,
                'recommendations': recommendations
            }
        except Exception as e:
            print(f"Error analyzing user patterns: {e}")
            return {}
    
    def _analyze_time_patterns(self, queries: List[QueryHistory]) -> Dict[str, Any]:
        """Analyze when user typically contacts support"""
        if not queries:
            return {}
        
        hours = [q.timestamp.hour for q in queries]
        days = [q.timestamp.weekday() for q in queries]  # 0=Monday
        
        # Most common hours
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Most common days
        day_counts = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days:
            day_name = day_names[day]
            day_counts[day_name] = day_counts.get(day_name, 0) + 1
        
        return {
            'peak_hours': sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'peak_days': sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            'total_queries': len(queries)
        }
    
    def _analyze_escalation_patterns(self, queries: List[QueryHistory]) -> Dict[str, Any]:
        """Analyze escalation patterns"""
        if not queries:
            return {}
        
        escalated_queries = [q for q in queries if q.escalated]
        escalation_rate = len(escalated_queries) / len(queries)
        
        # Escalation by category
        escalation_categories = {}
        for query in escalated_queries:
            category = query.category or 'general'
            escalation_categories[category] = escalation_categories.get(category, 0) + 1
        
        # Recent escalation trend
        recent_queries = sorted(queries, key=lambda x: x.timestamp, reverse=True)[:10]
        recent_escalations = sum(1 for q in recent_queries if q.escalated)
        recent_escalation_rate = recent_escalations / len(recent_queries) if recent_queries else 0
        
        return {
            'overall_rate': escalation_rate,
            'recent_rate': recent_escalation_rate,
            'categories': escalation_categories,
            'trend': 'increasing' if recent_escalation_rate > escalation_rate else 'stable'
        }
    
    def _analyze_satisfaction_patterns(self, queries: List[QueryHistory]) -> Dict[str, Any]:
        """Analyze user satisfaction patterns"""
        rated_queries = [q for q in queries if q.satisfaction_rating]
        
        if not rated_queries:
            return {'average_rating': None, 'trend': 'unknown', 'ratings_count': 0}
        
        ratings = [q.satisfaction_rating for q in rated_queries]
        avg_rating = sum(ratings) / len(ratings)
        
        # Recent satisfaction trend
        recent_rated = sorted(rated_queries, key=lambda x: x.timestamp, reverse=True)[:5]
        if len(recent_rated) >= 2:
            recent_avg = sum(q.satisfaction_rating for q in recent_rated) / len(recent_rated)
            trend = 'improving' if recent_avg > avg_rating else 'declining' if recent_avg < avg_rating else 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'average_rating': avg_rating,
            'recent_average': sum(q.satisfaction_rating for q in recent_rated) / len(recent_rated) if recent_rated else None,
            'trend': trend,
            'ratings_count': len(rated_queries),
            'distribution': {i: ratings.count(i) for i in range(1, 6)}
        }
    
    def _generate_user_insights(self, time_patterns: Dict, category_analysis: Dict, 
                               escalation_analysis: Dict, satisfaction_analysis: Dict) -> Dict[str, Any]:
        """Generate actionable insights from user patterns"""
        insights = {}
        
        # Communication preferences
        if time_patterns.get('peak_hours'):
            peak_hour = time_patterns['peak_hours'][0][0]
            if 9 <= peak_hour <= 17:
                insights['communication_preference'] = 'business_hours'
            elif 18 <= peak_hour <= 22:
                insights['communication_preference'] = 'evening'
            else:
                insights['communication_preference'] = 'off_hours'
        
        # User type classification
        escalation_rate = escalation_analysis.get('overall_rate', 0)
        if escalation_rate > 0.3:
            insights['user_type'] = 'high_maintenance'
        elif escalation_rate < 0.1:
            insights['user_type'] = 'self_sufficient'
        else:
            insights['user_type'] = 'standard'
        
        # Satisfaction trend
        sat_trend = satisfaction_analysis.get('trend', 'unknown')
        insights['satisfaction_trend'] = sat_trend
        
        # Most problematic category
        most_common = category_analysis.get('most_common')
        if most_common:
            insights['primary_concern'] = most_common
        
        return insights
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on user insights"""
        recommendations = []
        
        user_type = insights.get('user_type')
        if user_type == 'high_maintenance':
            recommendations.append("Consider proactive outreach for this user")
            recommendations.append("Prepare detailed responses and escalation paths")
        elif user_type == 'self_sufficient':
            recommendations.append("Provide comprehensive self-service resources")
            recommendations.append("Focus on quick, accurate responses")
        
        satisfaction_trend = insights.get('satisfaction_trend')
        if satisfaction_trend == 'declining':
            recommendations.append("Schedule follow-up to address concerns")
            recommendations.append("Consider human agent intervention")
        elif satisfaction_trend == 'improving':
            recommendations.append("Continue current approach")
        
        primary_concern = insights.get('primary_concern')
        if primary_concern == 'technical':
            recommendations.append("Have technical documentation ready")
            recommendations.append("Consider escalation to technical team")
        elif primary_concern == 'billing':
            recommendations.append("Verify account information")
            recommendations.append("Have billing specialist available")
        
        return recommendations
    
    # Utility methods
    def _parse_json_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Parse JSON metadata string"""
        if not metadata_str:
            return {}
        try:
            import json
            return json.loads(metadata_str)
        except Exception:
            return {}
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        return self.db_manager.cleanup_old_data(self.retention_days)
    
    def get_memory_health(self) -> Dict[str, Any]:
        """Get health status of long-term memory system"""
        return {
            'database_healthy': self.db_manager.health_check(),
            'cache_healthy': self.cache_manager.health_check(),
            'retention_days': self.retention_days
        }
