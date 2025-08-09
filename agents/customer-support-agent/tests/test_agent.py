"""Tests for Customer Support Agent"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.support_agent import CustomerSupportAgent
from src.agents.state import AgentState, create_initial_state, StateConstants
from src.utils.config import TestConfig


class TestCustomerSupportAgent:
    """Test cases for CustomerSupportAgent"""
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing"""
        return "test_api_key_123"
    
    @pytest.fixture
    def agent(self, mock_api_key):
        """Create agent instance for testing"""
        with patch('src.agents.support_agent.ChatGoogleGenerativeAI'):
            agent = CustomerSupportAgent(api_key=mock_api_key)
            return agent
    
    @pytest.fixture
    def sample_state(self):
        """Create sample agent state for testing"""
        return create_initial_state(
            user_id="test_user_123",
            thread_id="test_thread_123"
        )
    
    def test_agent_initialization(self, mock_api_key):
        """Test agent initialization"""
        with patch('src.agents.support_agent.ChatGoogleGenerativeAI'):
            agent = CustomerSupportAgent(api_key=mock_api_key)
            assert agent.api_key == mock_api_key
            assert agent.graph is not None
            assert agent.compiled_graph is not None
    
    def test_agent_initialization_without_api_key(self):
        """Test agent initialization without API key"""
        with patch('src.agents.support_agent.ChatGoogleGenerativeAI'):
            agent = CustomerSupportAgent(api_key=None)
            assert agent.llm is None
    
    def test_intake_query_node(self, agent, sample_state):
        """Test query intake functionality"""
        # Add a message to the state
        sample_state.messages = [{
            'role': StateConstants.ROLE_USER,
            'content': 'I need help with my account',
            'timestamp': datetime.now().isoformat()
        }]
        
        result = agent._intake_query(sample_state)
        
        assert result.processing_start_time is not None
        assert result.user_id == "test_user_123"
        assert result.last_updated is not None
    
    def test_route_after_escalation_check(self, agent, sample_state):
        """Test escalation routing logic"""
        # Test escalation case
        sample_state.requires_human = True
        result = agent._route_after_escalation_check(sample_state)
        assert result == "escalate"
        
        # Test normal response case
        sample_state.requires_human = False
        result = agent._route_after_escalation_check(sample_state)
        assert result == "respond"
    
    def test_trim_messages(self, agent, sample_state):
        """Test message trimming functionality"""
        # Create more messages than the limit
        sample_state.max_messages = 3
        sample_state.messages = [
            {'role': StateConstants.ROLE_USER, 'content': f'Message {i}', 'timestamp': datetime.now().isoformat()}
            for i in range(10)
        ]
        
        result = agent._trim_messages(sample_state)
        
        # Should keep only the most recent messages plus any system messages
        assert len(result.messages) <= sample_state.max_messages + 1  # +1 for summary message
    
    def test_extract_resolution(self, agent, sample_state):
        """Test resolution extraction"""
        sample_state.messages = [
            {'role': StateConstants.ROLE_USER, 'content': 'Help me', 'timestamp': datetime.now().isoformat()},
            {'role': StateConstants.ROLE_AGENT, 'content': 'Here is the solution', 'timestamp': datetime.now().isoformat()}
        ]
        
        resolution = agent._extract_resolution(sample_state)
        assert resolution == 'Here is the solution'
    
    def test_extract_agent_response(self, agent, sample_state):
        """Test agent response extraction"""
        sample_state.messages = [
            {'role': StateConstants.ROLE_USER, 'content': 'Help me', 'timestamp': datetime.now().isoformat()},
            {'role': StateConstants.ROLE_AGENT, 'content': 'I can help you', 'timestamp': datetime.now().isoformat()}
        ]
        
        response = agent._extract_agent_response(sample_state)
        assert response == 'I can help you'
    
    def test_extract_agent_response_no_messages(self, agent, sample_state):
        """Test agent response extraction with no messages"""
        sample_state.messages = []
        
        response = agent._extract_agent_response(sample_state)
        assert "couldn't process" in response.lower()
    
    @pytest.mark.asyncio
    async def test_process_message_integration(self, agent):
        """Test complete message processing integration"""
        with patch.object(agent.compiled_graph, 'ainvoke') as mock_ainvoke:
            # Mock the graph response
            mock_result = AgentState(
                user_id="test_user",
                thread_id="test_thread",
                messages=[
                    {'role': StateConstants.ROLE_USER, 'content': 'Test message', 'timestamp': datetime.now().isoformat()},
                    {'role': StateConstants.ROLE_AGENT, 'content': 'Test response', 'timestamp': datetime.now().isoformat()}
                ],
                requires_human=False,
                escalation_info=None,
                total_processing_time=0.5,
                metadata={'test': True}
            )
            mock_ainvoke.return_value = mock_result
            
            result = await agent.process_message(
                user_id="test_user",
                message="Test message"
            )
            
            assert result['response'] == 'Test response'
            assert result['escalated'] is False
            assert result['processing_time'] == 0.5
            assert 'metadata' in result
    
    def test_handle_human_response(self, agent):
        """Test human response handling"""
        with patch.object(agent.compiled_graph, 'get_state') as mock_get_state, \
             patch.object(agent.compiled_graph, 'update_state') as mock_update_state:
            
            # Mock state snapshot
            mock_state = AgentState(
                user_id="test_user",
                thread_id="test_thread",
                messages=[],
                escalation_info=None
            )
            mock_snapshot = Mock()
            mock_snapshot.values = mock_state
            mock_get_state.return_value = mock_snapshot
            
            result = agent.handle_human_response(
                thread_id="test_thread",
                human_response="Human agent response",
                agent_id="agent_123"
            )
            
            assert result['success'] is True
            mock_update_state.assert_called_once()
    
    def test_handle_human_response_thread_not_found(self, agent):
        """Test human response handling when thread not found"""
        with patch.object(agent.compiled_graph, 'get_state') as mock_get_state:
            mock_get_state.return_value = None
            
            result = agent.handle_human_response(
                thread_id="nonexistent_thread",
                human_response="Response",
                agent_id="agent_123"
            )
            
            assert 'error' in result
            assert 'Thread not found' in result['error']


class TestAgentWorkflow:
    """Test the complete agent workflow"""
    
    @pytest.fixture
    def agent_with_mock_llm(self):
        """Create agent with mocked LLM"""
        with patch('src.agents.support_agent.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm = Mock()
            mock_llm.invoke.return_value.content = "Mocked LLM response"
            mock_llm_class.return_value = mock_llm
            
            agent = CustomerSupportAgent(api_key="test_key")
            return agent
    
    def test_workflow_state_progression(self, agent_with_mock_llm):
        """Test that state progresses correctly through workflow"""
        initial_state = create_initial_state(
            user_id="test_user",
            current_query="I need help with billing"
        )
        
        # Test intake
        state_after_intake = agent_with_mock_llm._intake_query(initial_state)
        assert state_after_intake.processing_start_time is not None
        
        # Test finalize
        state_after_finalize = agent_with_mock_llm._finalize_response(state_after_intake)
        assert state_after_finalize.total_processing_time is not None
        assert state_after_finalize.metadata.get('workflow_completed') is True


class TestAgentConfiguration:
    """Test agent configuration and setup"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = TestConfig()
        errors = config.validate()
        
        # Test config should have minimal errors
        assert isinstance(errors, dict)
    
    def test_agent_with_test_config(self):
        """Test agent initialization with test configuration"""
        with patch('src.agents.support_agent.ChatGoogleGenerativeAI'):
            with patch('src.utils.config.Config', TestConfig):
                agent = CustomerSupportAgent(api_key="test_key")
                assert agent is not None


if __name__ == "__main__":
    pytest.main([__file__])
