"""Database management for customer support agent memory"""

import os
import json
import sqlite3
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from ..agents.state import UserProfile, QueryHistory, ConversationContext

Base = declarative_base()


class UserProfileDB(Base):
    """SQLAlchemy model for user profiles"""
    __tablename__ = 'user_profiles'
    
    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    account_type = Column(String, default='standard')
    created_at = Column(DateTime, default=datetime.now)
    last_interaction = Column(DateTime)
    metadata = Column(Text)  # JSON string


class QueryHistoryDB(Base):
    """SQLAlchemy model for query history"""
    __tablename__ = 'query_history'
    
    query_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    query_text = Column(Text, nullable=False)
    category = Column(String)
    resolution = Column(Text)
    escalated = Column(Boolean, default=False)
    resolved_by = Column(String)
    satisfaction_rating = Column(Integer)
    response_time_seconds = Column(Float)


class ConversationContextDB(Base):
    """SQLAlchemy model for conversation contexts"""
    __tablename__ = 'conversation_contexts'
    
    thread_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    session_start = Column(DateTime, default=datetime.now)
    last_activity = Column(DateTime, default=datetime.now)
    message_count = Column(Integer, default=0)
    context_summary = Column(Text)
    sentiment = Column(String)
    priority = Column(String, default='normal')


class EscalationDB(Base):
    """SQLAlchemy model for escalations"""
    __tablename__ = 'escalations'
    
    escalation_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    thread_id = Column(String, nullable=False)
    escalated_at = Column(DateTime, default=datetime.now)
    reason = Column(String, nullable=False)
    urgency_level = Column(String, default='medium')
    assigned_agent = Column(String)
    estimated_resolution_time = Column(DateTime)
    resolved_at = Column(DateTime)
    customer_notified = Column(Boolean, default=False)
    resolution_notes = Column(Text)


class AgentMetricsDB(Base):
    """SQLAlchemy model for agent metrics"""
    __tablename__ = 'agent_metrics'
    
    id = Column(String, primary_key=True)
    date = Column(DateTime, default=datetime.now)
    total_queries = Column(Integer, default=0)
    resolved_queries = Column(Integer, default=0)
    escalated_queries = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    average_confidence_score = Column(Float, default=0.0)
    user_satisfaction_score = Column(Float)


class DatabaseManager:
    """Manage database operations for the customer support agent"""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager"""
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'sqlite:///customer_support.db'
        )
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True
        )
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with proper cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # User Profile Operations
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save or update user profile"""
        try:
            with self.get_session() as session:
                db_profile = UserProfileDB(
                    user_id=profile.user_id,
                    email=profile.email,
                    name=profile.name,
                    account_type=profile.account_type,
                    created_at=profile.created_at,
                    last_interaction=datetime.now(),
                    metadata=json.dumps(profile.metadata) if profile.metadata else '{}'
                )
                
                # Use merge to handle both insert and update
                session.merge(db_profile)
                return True
                
        except Exception as e:
            print(f"Error saving user profile: {e}")
            return False
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfileDB]:
        """Get user profile by ID"""
        try:
            with self.get_session() as session:
                return session.query(UserProfileDB).filter_by(user_id=user_id).first()
        except Exception as e:
            print(f"Error retrieving user profile: {e}")
            return None
    
    def update_user_last_interaction(self, user_id: str) -> bool:
        """Update user's last interaction timestamp"""
        try:
            with self.get_session() as session:
                profile = session.query(UserProfileDB).filter_by(user_id=user_id).first()
                if profile:
                    profile.last_interaction = datetime.now()
                    return True
                return False
        except Exception as e:
            print(f"Error updating user interaction: {e}")
            return False
    
    # Query History Operations
    def save_query_history(self, query: QueryHistory) -> bool:
        """Save query history entry"""
        try:
            with self.get_session() as session:
                db_query = QueryHistoryDB(
                    query_id=query.query_id,
                    user_id=query.user_id,
                    timestamp=query.timestamp,
                    query_text=query.query_text,
                    category=query.category,
                    resolution=query.resolution,
                    escalated=query.escalated,
                    resolved_by=query.resolved_by,
                    satisfaction_rating=query.satisfaction_rating,
                    response_time_seconds=query.response_time_seconds
                )
                
                session.add(db_query)
                return True
                
        except Exception as e:
            print(f"Error saving query history: {e}")
            return False
    
    def get_query_history(
        self, 
        user_id: str, 
        limit: int = 10, 
        since: Optional[datetime] = None
    ) -> List[QueryHistoryDB]:
        """Get query history for a user"""
        try:
            with self.get_session() as session:
                query = session.query(QueryHistoryDB).filter_by(user_id=user_id)
                
                if since:
                    query = query.filter(QueryHistoryDB.timestamp >= since)
                
                return query.order_by(QueryHistoryDB.timestamp.desc()).limit(limit).all()
                
        except Exception as e:
            print(f"Error retrieving query history: {e}")
            return []
    
    def get_recent_queries_by_category(
        self, 
        user_id: str, 
        category: str, 
        days: int = 30
    ) -> List[QueryHistoryDB]:
        """Get recent queries by category"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                return session.query(QueryHistoryDB)\
                    .filter_by(user_id=user_id, category=category)\
                    .filter(QueryHistoryDB.timestamp >= cutoff_date)\
                    .order_by(QueryHistoryDB.timestamp.desc())\
                    .all()
                    
        except Exception as e:
            print(f"Error retrieving queries by category: {e}")
            return []
    
    # Conversation Context Operations
    def save_conversation_context(self, context: ConversationContext) -> bool:
        """Save or update conversation context"""
        try:
            with self.get_session() as session:
                db_context = ConversationContextDB(
                    thread_id=context.thread_id,
                    user_id=getattr(context, 'user_id', ''),  # May need to add this to schema
                    session_start=context.session_start,
                    last_activity=context.last_activity,
                    message_count=context.message_count,
                    context_summary=context.context_summary,
                    sentiment=context.sentiment,
                    priority=context.priority
                )
                
                session.merge(db_context)
                return True
                
        except Exception as e:
            print(f"Error saving conversation context: {e}")
            return False
    
    def get_conversation_context(self, thread_id: str) -> Optional[ConversationContextDB]:
        """Get conversation context by thread ID"""
        try:
            with self.get_session() as session:
                return session.query(ConversationContextDB)\
                    .filter_by(thread_id=thread_id).first()
        except Exception as e:
            print(f"Error retrieving conversation context: {e}")
            return None
    
    def cleanup_old_contexts(self, days: int = 30) -> int:
        """Clean up old conversation contexts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                deleted = session.query(ConversationContextDB)\
                    .filter(ConversationContextDB.last_activity < cutoff_date)\
                    .delete()
                return deleted
                
        except Exception as e:
            print(f"Error cleaning up contexts: {e}")
            return 0
    
    # Escalation Operations
    def save_escalation(self, escalation_data: Dict[str, Any]) -> bool:
        """Save escalation information"""
        try:
            with self.get_session() as session:
                db_escalation = EscalationDB(
                    escalation_id=escalation_data.get('escalation_id'),
                    user_id=escalation_data.get('user_id'),
                    thread_id=escalation_data.get('thread_id'),
                    reason=escalation_data.get('reason'),
                    urgency_level=escalation_data.get('urgency_level', 'medium'),
                    assigned_agent=escalation_data.get('assigned_agent'),
                    estimated_resolution_time=escalation_data.get('estimated_resolution_time'),
                    customer_notified=escalation_data.get('customer_notified', False)
                )
                
                session.add(db_escalation)
                return True
                
        except Exception as e:
            print(f"Error saving escalation: {e}")
            return False
    
    def get_pending_escalations(self) -> List[EscalationDB]:
        """Get all pending escalations"""
        try:
            with self.get_session() as session:
                return session.query(EscalationDB)\
                    .filter(EscalationDB.resolved_at.is_(None))\
                    .order_by(EscalationDB.urgency_level.desc(), EscalationDB.escalated_at)\
                    .all()
        except Exception as e:
            print(f"Error retrieving pending escalations: {e}")
            return []
    
    def resolve_escalation(
        self, 
        escalation_id: str, 
        resolution_notes: str, 
        resolved_by: str
    ) -> bool:
        """Mark escalation as resolved"""
        try:
            with self.get_session() as session:
                escalation = session.query(EscalationDB)\
                    .filter_by(escalation_id=escalation_id).first()
                
                if escalation:
                    escalation.resolved_at = datetime.now()
                    escalation.resolution_notes = resolution_notes
                    escalation.assigned_agent = resolved_by
                    return True
                return False
                
        except Exception as e:
            print(f"Error resolving escalation: {e}")
            return False
    
    # Analytics and Metrics
    def save_daily_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Save daily agent metrics"""
        try:
            with self.get_session() as session:
                metrics_id = f"{datetime.now().date()}_metrics"
                
                db_metrics = AgentMetricsDB(
                    id=metrics_id,
                    date=datetime.now(),
                    total_queries=metrics_data.get('total_queries', 0),
                    resolved_queries=metrics_data.get('resolved_queries', 0),
                    escalated_queries=metrics_data.get('escalated_queries', 0),
                    average_response_time=metrics_data.get('average_response_time', 0.0),
                    average_confidence_score=metrics_data.get('average_confidence_score', 0.0),
                    user_satisfaction_score=metrics_data.get('user_satisfaction_score')
                )
                
                session.merge(db_metrics)
                return True
                
        except Exception as e:
            print(f"Error saving metrics: {e}")
            return False
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get metrics summary for the last N days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_session() as session:
                metrics = session.query(AgentMetricsDB)\
                    .filter(AgentMetricsDB.date >= cutoff_date)\
                    .all()
                
                if not metrics:
                    return {
                        'total_queries': 0,
                        'resolved_queries': 0,
                        'escalated_queries': 0,
                        'average_response_time': 0.0,
                        'resolution_rate': 0.0,
                        'escalation_rate': 0.0,
                        'user_satisfaction': 0.0
                    }
                
                total_queries = sum(m.total_queries for m in metrics)
                resolved_queries = sum(m.resolved_queries for m in metrics)
                escalated_queries = sum(m.escalated_queries for m in metrics)
                
                return {
                    'total_queries': total_queries,
                    'resolved_queries': resolved_queries,
                    'escalated_queries': escalated_queries,
                    'average_response_time': sum(m.average_response_time for m in metrics) / len(metrics),
                    'resolution_rate': (resolved_queries / total_queries * 100) if total_queries > 0 else 0,
                    'escalation_rate': (escalated_queries / total_queries * 100) if total_queries > 0 else 0,
                    'user_satisfaction': sum(m.user_satisfaction_score or 0 for m in metrics) / len([m for m in metrics if m.user_satisfaction_score])
                }
                
        except Exception as e:
            print(f"Error retrieving metrics summary: {e}")
            return {}
    
    # Utility methods
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute('SELECT 1')
                return True
        except Exception as e:
            print(f"Database health check failed: {e}")
            return False
    
    def cleanup_old_data(self, days: int = 365) -> Dict[str, int]:
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cleanup_results = {}
        
        try:
            with self.get_session() as session:
                # Clean up old query history
                deleted_queries = session.query(QueryHistoryDB)\
                    .filter(QueryHistoryDB.timestamp < cutoff_date)\
                    .delete()
                cleanup_results['deleted_queries'] = deleted_queries
                
                # Clean up old conversation contexts
                deleted_contexts = session.query(ConversationContextDB)\
                    .filter(ConversationContextDB.last_activity < cutoff_date)\
                    .delete()
                cleanup_results['deleted_contexts'] = deleted_contexts
                
                # Clean up resolved escalations
                deleted_escalations = session.query(EscalationDB)\
                    .filter(EscalationDB.resolved_at < cutoff_date)\
                    .delete()
                cleanup_results['deleted_escalations'] = deleted_escalations
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            cleanup_results['error'] = str(e)
        
        return cleanup_results
    
    def backup_database(self, backup_path: str) -> bool:
        """Create database backup (SQLite only)"""
        if not self.database_url.startswith('sqlite'):
            print("Backup only supported for SQLite databases")
            return False
        
        try:
            import shutil
            db_path = self.database_url.replace('sqlite:///', '')
            shutil.copy2(db_path, backup_path)
            return True
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
