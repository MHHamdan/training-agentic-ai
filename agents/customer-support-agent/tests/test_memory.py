"""Tests for Memory Management System"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.database import DatabaseManager
from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.agents.state import UserProfile, QueryHistory, ConversationContext


class TestDatabaseManager:
    """Test cases for DatabaseManager"""
    
    @pytest.fixture
    def db_manager(self):
        """Create database manager with in-memory SQLite"""
        return DatabaseManager('sqlite:///:memory:')
    
    @pytest.fixture
    def sample_user_profile(self):
        """Create sample user profile"""
        return UserProfile(
            user_id="test_user_123",
            email="test@example.com",
            name="Test User",
            account_type="standard"
        )
    
    @pytest.fixture
    def sample_query_history(self):
        """Create sample query history"""
        return QueryHistory(
            user_id="test_user_123",
            query_text="How do I reset my password?",
            category="account"
        )
    
    @pytest.fixture
    def sample_conversation_context(self):
        """Create sample conversation context"""
        return ConversationContext(
            thread_id="test_thread_123"
        )
    
    def test_save_and_get_user_profile(self, db_manager, sample_user_profile):
        """Test saving and retrieving user profile"""
        # Save profile
        success = db_manager.save_user_profile(sample_user_profile)
        assert success is True
        
        # Retrieve profile
        retrieved_profile = db_manager.get_user_profile(sample_user_profile.user_id)
        assert retrieved_profile is not None
        assert retrieved_profile.user_id == sample_user_profile.user_id
        assert retrieved_profile.email == sample_user_profile.email
        assert retrieved_profile.name == sample_user_profile.name
    
    def test_save_and_get_query_history(self, db_manager, sample_query_history):
        """Test saving and retrieving query history"""
        # Save query
        success = db_manager.save_query_history(sample_query_history)
        assert success is True
        
        # Retrieve queries
        queries = db_manager.get_query_history(sample_query_history.user_id)
        assert len(queries) == 1
        assert queries[0].query_text == sample_query_history.query_text
        assert queries[0].category == sample_query_history.category
    
    def test_save_and_get_conversation_context(self, db_manager, sample_conversation_context):
        """Test saving and retrieving conversation context"""
        # Save context
        success = db_manager.save_conversation_context(sample_conversation_context)
        assert success is True
        
        # Retrieve context
        retrieved_context = db_manager.get_conversation_context(sample_conversation_context.thread_id)
        assert retrieved_context is not None
        assert retrieved_context.thread_id == sample_conversation_context.thread_id
    
    def test_get_recent_queries_by_category(self, db_manager, sample_user_profile):
        """Test retrieving queries by category"""
        # Create multiple queries
        queries = [
            QueryHistory(user_id=sample_user_profile.user_id, query_text="Account question 1", category="account"),
            QueryHistory(user_id=sample_user_profile.user_id, query_text="Billing question 1", category="billing"),
            QueryHistory(user_id=sample_user_profile.user_id, query_text="Account question 2", category="account")
        ]
        
        for query in queries:
            db_manager.save_query_history(query)
        
        # Retrieve account queries
        account_queries = db_manager.get_recent_queries_by_category(sample_user_profile.user_id, "account")
        assert len(account_queries) == 2
        
        # Retrieve billing queries
        billing_queries = db_manager.get_recent_queries_by_category(sample_user_profile.user_id, "billing")
        assert len(billing_queries) == 1
    
    def test_cleanup_old_contexts(self, db_manager):
        """Test cleanup of old conversation contexts"""
        # Create old and new contexts
        old_context = ConversationContext(
            thread_id="old_thread",
            last_activity=datetime.now() - timedelta(days=35)
        )
        new_context = ConversationContext(
            thread_id="new_thread",
            last_activity=datetime.now()
        )
        
        db_manager.save_conversation_context(old_context)
        db_manager.save_conversation_context(new_context)
        
        # Cleanup contexts older than 30 days
        deleted_count = db_manager.cleanup_old_contexts(days=30)
        assert deleted_count == 1
        
        # Verify new context still exists
        retrieved_new = db_manager.get_conversation_context("new_thread")
        assert retrieved_new is not None
        
        # Verify old context was deleted
        retrieved_old = db_manager.get_conversation_context("old_thread")
        assert retrieved_old is None
    
    def test_health_check(self, db_manager):
        """Test database health check"""
        health = db_manager.health_check()
        assert health is True
    
    def test_metrics_operations(self, db_manager):
        """Test metrics saving and retrieval"""
        metrics_data = {
            'total_queries': 100,
            'resolved_queries': 85,
            'escalated_queries': 15,
            'average_response_time': 2.5,
            'average_confidence_score': 0.8
        }
        
        # Save metrics
        success = db_manager.save_daily_metrics(metrics_data)
        assert success is True
        
        # Get metrics summary
        summary = db_manager.get_metrics_summary(days=1)
        assert summary['total_queries'] == 100
        assert summary['resolution_rate'] == 85.0
        assert summary['escalation_rate'] == 15.0


class TestShortTermMemory:
    """Test cases for ShortTermMemory"""
    
    @pytest.fixture
    def short_term_memory(self):
        """Create short-term memory with mock Redis"""
        with patch('redis.from_url') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client
            
            memory = ShortTermMemory(ttl=60)
            memory.redis_client = mock_client
            return memory
    
    def test_store_and_get_session_data(self, short_term_memory):
        """Test session data storage and retrieval"""
        session_id = "test_session_123"
        session_data = {"user_id": "test_user", "messages": []}
        
        # Mock Redis operations
        short_term_memory.redis_client.setex.return_value = True
        short_term_memory.redis_client.get.return_value = '{"user_id": "test_user", "messages": []}'
        
        # Store data
        success = short_term_memory.store_session_data(session_id, session_data)
        assert success is True
        
        # Retrieve data
        retrieved_data = short_term_memory.get_session_data(session_id)
        assert retrieved_data is not None
        assert retrieved_data['user_id'] == 'test_user'
    
    def test_cache_conversation_context(self, short_term_memory):
        """Test conversation context caching"""
        thread_id = "test_thread_123"
        context = ConversationContext(thread_id=thread_id)
        
        # Mock Redis operations
        short_term_memory.redis_client.setex.return_value = True
        
        success = short_term_memory.cache_conversation_context(thread_id, context)
        assert success is True
        
        short_term_memory.redis_client.setex.assert_called_once()
    
    def test_rate_limiting(self, short_term_memory):
        """Test rate limiting functionality"""
        user_id = "test_user_123"
        
        # Mock Redis operations for rate limiting
        short_term_memory.redis_client.get.return_value = None  # First request
        short_term_memory.redis_client.setex.return_value = True
        
        # First request should be allowed
        allowed = short_term_memory.check_rate_limit(user_id, limit=5, window=60)
        assert allowed is True
        
        # Mock subsequent requests
        short_term_memory.redis_client.get.return_value = '4'  # 4 requests made
        short_term_memory.redis_client.incr.return_value = 5
        
        # Should still be allowed (5th request)
        allowed = short_term_memory.check_rate_limit(user_id, limit=5, window=60)
        assert allowed is True
        
        # Mock exceeding limit
        short_term_memory.redis_client.get.return_value = '5'  # 5 requests made
        
        # Should be denied (6th request)
        allowed = short_term_memory.check_rate_limit(user_id, limit=5, window=60)
        assert allowed is False
    
    def test_memory_stats(self, short_term_memory):
        """Test memory statistics retrieval"""
        # Mock Redis info response
        short_term_memory.redis_client.info.return_value = {
            'used_memory': 1024000,
            'used_memory_human': '1M',
            'maxmemory': 0
        }
        short_term_memory.redis_client.dbsize.return_value = 100
        
        stats = short_term_memory.get_memory_stats()
        assert stats['used_memory'] == 1024000
        assert stats['used_memory_human'] == '1M'
        assert stats['keys_count'] == 100
    
    def test_health_check(self, short_term_memory):
        """Test health check"""
        short_term_memory.redis_client.ping.return_value = True
        
        health = short_term_memory.health_check()
        assert health is True
        
        # Test failure case
        short_term_memory.redis_client.ping.side_effect = Exception("Connection failed")
        
        health = short_term_memory.health_check()
        assert health is False


class TestLongTermMemory:
    """Test cases for LongTermMemory"""
    
    @pytest.fixture
    def long_term_memory(self):
        """Create long-term memory with mocked dependencies"""
        with patch('src.memory.long_term.DatabaseManager') as mock_db, \
             patch('src.memory.long_term.ShortTermMemory') as mock_cache:
            
            db_manager = Mock()
            cache_manager = Mock()
            
            mock_db.return_value = db_manager
            mock_cache.return_value = cache_manager
            
            memory = LongTermMemory(db_manager, cache_manager)
            return memory, db_manager, cache_manager
    
    def test_store_and_get_user_profile(self, long_term_memory):
        """Test user profile storage and retrieval"""
        memory, db_manager, cache_manager = long_term_memory
        
        profile = UserProfile(
            user_id="test_user",
            email="test@example.com",
            name="Test User",
            account_type="standard"
        )
        
        # Mock successful database save
        db_manager.save_user_profile.return_value = True
        
        # Store profile
        success = memory.store_user_profile(profile)
        assert success is True
        
        db_manager.save_user_profile.assert_called_once_with(profile)
        cache_manager.cache_user_state.assert_called_once()
    
    def test_get_user_profile_from_cache(self, long_term_memory):
        """Test user profile retrieval from cache"""
        memory, db_manager, cache_manager = long_term_memory
        
        # Mock cache hit
        cached_profile = {
            'user_id': 'test_user',
            'email': 'test@example.com',
            'name': 'Test User',
            'account_type': 'standard',
            'created_at': datetime.now().isoformat(),
            'metadata': {}
        }
        cache_manager.get_cached_user_state.return_value = cached_profile
        
        profile = memory.get_user_profile('test_user')
        
        assert profile is not None
        assert profile.user_id == 'test_user'
        assert profile.email == 'test@example.com'
        
        # Database should not be called
        db_manager.get_user_profile.assert_not_called()
    
    def test_get_user_profile_from_database(self, long_term_memory):
        """Test user profile retrieval from database when cache misses"""
        memory, db_manager, cache_manager = long_term_memory
        
        # Mock cache miss
        cache_manager.get_cached_user_state.return_value = None
        
        # Mock database hit
        db_profile = Mock()
        db_profile.user_id = 'test_user'
        db_profile.email = 'test@example.com'
        db_profile.name = 'Test User'
        db_profile.account_type = 'standard'
        db_profile.created_at = datetime.now()
        db_profile.metadata = '{}'
        
        db_manager.get_user_profile.return_value = db_profile
        
        profile = memory.get_user_profile('test_user')
        
        assert profile is not None
        assert profile.user_id == 'test_user'
        
        # Both cache and database should be called
        cache_manager.get_cached_user_state.assert_called_once()
        db_manager.get_user_profile.assert_called_once()
    
    def test_analyze_user_patterns(self, long_term_memory):
        """Test user pattern analysis"""
        memory, db_manager, cache_manager = long_term_memory
        
        # Mock query history
        mock_queries = [
            Mock(timestamp=datetime.now() - timedelta(hours=i), 
                 category='technical', escalated=False, satisfaction_rating=4)
            for i in range(5)
        ]
        
        with patch.object(memory, 'get_user_query_history', return_value=mock_queries):
            patterns = memory.analyze_user_patterns('test_user')
            
            assert 'patterns' in patterns
            assert 'insights' in patterns
            assert 'recommendations' in patterns
    
    def test_get_category_trends(self, long_term_memory):
        """Test category trends analysis"""
        memory, db_manager, cache_manager = long_term_memory
        
        # Mock query history with different categories
        mock_queries = [
            Mock(category='technical', escalated=False, response_time_seconds=5.0),
            Mock(category='technical', escalated=True, response_time_seconds=10.0),
            Mock(category='billing', escalated=False, response_time_seconds=3.0),
        ]
        
        with patch.object(memory, 'get_user_query_history', return_value=mock_queries):
            trends = memory.get_category_trends('test_user')
            
            assert trends['trends']['technical'] == 2
            assert trends['trends']['billing'] == 1
            assert trends['most_common'] == 'technical'
            assert trends['escalation_patterns']['technical'] == 1
    
    def test_cleanup_old_data(self, long_term_memory):
        """Test old data cleanup"""
        memory, db_manager, cache_manager = long_term_memory
        
        # Mock cleanup results
        db_manager.cleanup_old_data.return_value = {
            'deleted_queries': 50,
            'deleted_contexts': 10,
            'deleted_escalations': 5
        }
        
        results = memory.cleanup_old_data()
        
        assert results['deleted_queries'] == 50
        assert results['deleted_contexts'] == 10
        assert results['deleted_escalations'] == 5
        
        db_manager.cleanup_old_data.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
