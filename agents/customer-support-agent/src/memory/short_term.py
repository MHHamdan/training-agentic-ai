"""Short-term memory management for customer support agent"""

import json
import redis
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..agents.state import AgentState, ConversationContext


class ShortTermMemory:
    """Manage short-term memory using Redis for session-based storage"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        """Initialize short-term memory manager
        
        Args:
            redis_url: Redis connection URL
            ttl: Time-to-live for cached data in seconds (default: 1 hour)
        """
        self.ttl = ttl
        self.redis_client = self._connect_redis(redis_url)
        self.key_prefix = "customer_support:"
    
    def _connect_redis(self, redis_url: str) -> Optional[redis.Redis]:
        """Connect to Redis with fallback to in-memory storage"""
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            client.ping()
            return client
        except Exception as e:
            print(f"Redis connection failed: {e}. Using in-memory fallback.")
            return None
    
    def _get_key(self, key_type: str, identifier: str) -> str:
        """Generate Redis key with prefix"""
        return f"{self.key_prefix}{key_type}:{identifier}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for storage"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        return str(data)
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from storage"""
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    
    # Session Management
    def store_session_data(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Store session data"""
        try:
            if self.redis_client:
                key = self._get_key("session", session_id)
                serialized_data = self._serialize_data(data)
                self.redis_client.setex(key, self.ttl, serialized_data)
            else:
                # Fallback to in-memory storage (not persistent)
                self._memory_store = getattr(self, '_memory_store', {})
                self._memory_store[session_id] = data
            return True
        except Exception as e:
            print(f"Error storing session data: {e}")
            return False
    
    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        try:
            if self.redis_client:
                key = self._get_key("session", session_id)
                data = self.redis_client.get(key)
                return self._deserialize_data(data) if data else None
            else:
                # Fallback to in-memory storage
                memory_store = getattr(self, '_memory_store', {})
                return memory_store.get(session_id)
        except Exception as e:
            print(f"Error retrieving session data: {e}")
            return None
    
    def extend_session(self, session_id: str) -> bool:
        """Extend session TTL"""
        try:
            if self.redis_client:
                key = self._get_key("session", session_id)
                return bool(self.redis_client.expire(key, self.ttl))
            return True  # In-memory storage doesn't expire
        except Exception as e:
            print(f"Error extending session: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session data"""
        try:
            if self.redis_client:
                key = self._get_key("session", session_id)
                return bool(self.redis_client.delete(key))
            else:
                memory_store = getattr(self, '_memory_store', {})
                return memory_store.pop(session_id, None) is not None
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    # Conversation Context Caching
    def cache_conversation_context(self, thread_id: str, context: ConversationContext) -> bool:
        """Cache conversation context for quick access"""
        try:
            context_data = {
                'thread_id': context.thread_id,
                'session_start': context.session_start.isoformat(),
                'last_activity': context.last_activity.isoformat(),
                'message_count': context.message_count,
                'context_summary': context.context_summary,
                'sentiment': context.sentiment,
                'priority': context.priority
            }
            
            if self.redis_client:
                key = self._get_key("context", thread_id)
                serialized_data = self._serialize_data(context_data)
                self.redis_client.setex(key, self.ttl, serialized_data)
            else:
                self._context_store = getattr(self, '_context_store', {})
                self._context_store[thread_id] = context_data
            
            return True
        except Exception as e:
            print(f"Error caching conversation context: {e}")
            return False
    
    def get_cached_context(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached conversation context"""
        try:
            if self.redis_client:
                key = self._get_key("context", thread_id)
                data = self.redis_client.get(key)
                return self._deserialize_data(data) if data else None
            else:
                context_store = getattr(self, '_context_store', {})
                return context_store.get(thread_id)
        except Exception as e:
            print(f"Error retrieving cached context: {e}")
            return None
    
    # Message History Caching
    def cache_recent_messages(self, thread_id: str, messages: List[Dict[str, Any]], limit: int = 10) -> bool:
        """Cache recent messages for a conversation"""
        try:
            # Keep only the most recent messages
            recent_messages = messages[-limit:] if len(messages) > limit else messages
            
            if self.redis_client:
                key = self._get_key("messages", thread_id)
                serialized_data = self._serialize_data(recent_messages)
                self.redis_client.setex(key, self.ttl, serialized_data)
            else:
                self._messages_store = getattr(self, '_messages_store', {})
                self._messages_store[thread_id] = recent_messages
            
            return True
        except Exception as e:
            print(f"Error caching messages: {e}")
            return False
    
    def get_cached_messages(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieve cached messages"""
        try:
            if self.redis_client:
                key = self._get_key("messages", thread_id)
                data = self.redis_client.get(key)
                return self._deserialize_data(data) if data else []
            else:
                messages_store = getattr(self, '_messages_store', {})
                return messages_store.get(thread_id, [])
        except Exception as e:
            print(f"Error retrieving cached messages: {e}")
            return []
    
    # User State Caching
    def cache_user_state(self, user_id: str, state_data: Dict[str, Any]) -> bool:
        """Cache user state information"""
        try:
            # Add timestamp
            state_data['cached_at'] = datetime.now().isoformat()
            
            if self.redis_client:
                key = self._get_key("user_state", user_id)
                serialized_data = self._serialize_data(state_data)
                self.redis_client.setex(key, self.ttl, serialized_data)
            else:
                self._user_state_store = getattr(self, '_user_state_store', {})
                self._user_state_store[user_id] = state_data
            
            return True
        except Exception as e:
            print(f"Error caching user state: {e}")
            return False
    
    def get_cached_user_state(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached user state"""
        try:
            if self.redis_client:
                key = self._get_key("user_state", user_id)
                data = self.redis_client.get(key)
                return self._deserialize_data(data) if data else None
            else:
                user_state_store = getattr(self, '_user_state_store', {})
                return user_state_store.get(user_id)
        except Exception as e:
            print(f"Error retrieving cached user state: {e}")
            return None
    
    # Analytics Caching
    def cache_analytics_data(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache analytics data with optional custom TTL"""
        try:
            cache_ttl = ttl or self.ttl
            
            if self.redis_client:
                cache_key = self._get_key("analytics", key)
                serialized_data = self._serialize_data(data)
                self.redis_client.setex(cache_key, cache_ttl, serialized_data)
            else:
                self._analytics_store = getattr(self, '_analytics_store', {})
                self._analytics_store[key] = data
            
            return True
        except Exception as e:
            print(f"Error caching analytics data: {e}")
            return False
    
    def get_cached_analytics(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analytics data"""
        try:
            if self.redis_client:
                cache_key = self._get_key("analytics", key)
                data = self.redis_client.get(cache_key)
                return self._deserialize_data(data) if data else None
            else:
                analytics_store = getattr(self, '_analytics_store', {})
                return analytics_store.get(key)
        except Exception as e:
            print(f"Error retrieving cached analytics: {e}")
            return None
    
    # Rate Limiting
    def check_rate_limit(self, user_id: str, limit: int = 10, window: int = 60) -> bool:
        """Check if user is within rate limits"""
        try:
            if self.redis_client:
                key = self._get_key("rate_limit", user_id)
                current = self.redis_client.get(key)
                
                if current is None:
                    # First request in window
                    self.redis_client.setex(key, window, 1)
                    return True
                elif int(current) < limit:
                    # Within limit
                    self.redis_client.incr(key)
                    return True
                else:
                    # Exceeded limit
                    return False
            else:
                # No rate limiting for in-memory fallback
                return True
                
        except Exception as e:
            print(f"Error checking rate limit: {e}")
            return True  # Allow on error
    
    def get_rate_limit_status(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get current rate limit status for user"""
        try:
            if self.redis_client:
                key = self._get_key("rate_limit", user_id)
                current = self.redis_client.get(key)
                ttl = self.redis_client.ttl(key)
                
                if current is None:
                    return {'requests': 0, 'limit': limit, 'reset_in': 0, 'remaining': limit}
                
                requests = int(current)
                return {
                    'requests': requests,
                    'limit': limit,
                    'reset_in': ttl if ttl > 0 else 0,
                    'remaining': max(0, limit - requests)
                }
            else:
                return {'requests': 0, 'limit': limit, 'reset_in': 0, 'remaining': limit}
                
        except Exception as e:
            print(f"Error getting rate limit status: {e}")
            return {'requests': 0, 'limit': limit, 'reset_in': 0, 'remaining': limit}
    
    # Cleanup and Maintenance
    def cleanup_expired_keys(self) -> int:
        """Clean up expired keys (Redis handles this automatically)"""
        try:
            if self.redis_client:
                # Redis handles expiration automatically
                return 0
            else:
                # For in-memory stores, we don't have expiration
                # In a real implementation, you'd track timestamps
                return 0
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return 0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            if self.redis_client:
                info = self.redis_client.info('memory')
                return {
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'max_memory': info.get('maxmemory', 0),
                    'keys_count': self.redis_client.dbsize()
                }
            else:
                # Estimate in-memory usage
                stores = ['_memory_store', '_context_store', '_messages_store', '_user_state_store', '_analytics_store']
                total_keys = sum(len(getattr(self, store, {})) for store in stores)
                return {
                    'used_memory': 0,
                    'used_memory_human': 'Unknown',
                    'max_memory': 0,
                    'keys_count': total_keys
                }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if memory system is healthy"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            else:
                # In-memory fallback is always "healthy"
                return True
        except Exception as e:
            print(f"Memory health check failed: {e}")
            return False
