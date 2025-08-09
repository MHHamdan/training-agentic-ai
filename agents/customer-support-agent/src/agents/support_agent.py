"""Main Customer Support Agent with LangGraph workflow"""

import os
import asyncio
from typing import Literal, Dict, Any, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState, create_initial_state, update_state_timestamp, StateConstants
from ..nodes.query_processor import process_query_node
from ..nodes.history_retriever import retrieve_history_node
from ..nodes.escalation import check_escalation_node
from ..nodes.response_generator import ResponseGenerator
from ..memory.database import DatabaseManager
from ..utils.config import Config


class CustomerSupportAgent:
    """Main customer support agent orchestrating the workflow with LangGraph"""
    
    def __init__(self, api_key: Optional[str] = None, db_manager: Optional[DatabaseManager] = None):
        """Initialize the customer support agent"""
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.db_manager = db_manager or DatabaseManager()
        self.memory_store = MemorySaver()
        
        # Initialize LLM
        self.llm = self._setup_llm()
        
        # Initialize response generator with LLM
        self.response_generator = ResponseGenerator(self.llm)
        
        # Build the LangGraph workflow
        self.graph = self._build_graph()
        
        # Compile the graph with checkpointer for memory
        self.compiled_graph = self.graph.compile(
            checkpointer=self.memory_store,
            interrupt_before=[],  # Can add nodes here to create breakpoints
            interrupt_after=[]
        )
    
    def _setup_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """Setup Google Gemini LLM"""
        if not self.api_key:
            print("Warning: No Google API key provided. Using fallback responses.")
            return None
        
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,
                max_output_tokens=2048,
                convert_system_message_to_human=True
            )
        except Exception as e:
            print(f"Error setting up LLM: {e}")
            return None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add all nodes
        workflow.add_node("intake", self._intake_query)
        workflow.add_node("check_history", retrieve_history_node)
        workflow.add_node("process_query", process_query_node)
        workflow.add_node("check_escalation", check_escalation_node)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("handle_escalation", self._handle_escalation)
        workflow.add_node("trim_messages", self._trim_messages)
        workflow.add_node("save_interaction", self._save_interaction)
        workflow.add_node("finalize", self._finalize_response)
        
        # Set entry point
        workflow.set_entry_point("intake")
        
        # Define the workflow edges
        workflow.add_edge("intake", "check_history")
        workflow.add_edge("check_history", "process_query")
        workflow.add_edge("process_query", "check_escalation")
        
        # Conditional routing after escalation check
        workflow.add_conditional_edges(
            "check_escalation",
            self._route_after_escalation_check,
            {
                "escalate": "handle_escalation",
                "respond": "generate_response"
            }
        )
        
        # Both paths lead to message trimming
        workflow.add_edge("generate_response", "trim_messages")
        workflow.add_edge("handle_escalation", "trim_messages")
        
        # Continue to save and finalize
        workflow.add_edge("trim_messages", "save_interaction")
        workflow.add_edge("save_interaction", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def _intake_query(self, state: AgentState) -> AgentState:
        """Initial query intake and preprocessing"""
        # Update timestamp
        state = update_state_timestamp(state)
        
        # Set processing start time
        state.processing_start_time = datetime.now()
        
        # Validate state
        if not state.user_id:
            raise ValueError("User ID is required")
        
        # Initialize metadata if not present
        if not state.metadata:
            state.metadata = {}
        
        # Log the query intake
        print(f"Processing query for user {state.user_id}: {state.current_query}")
        
        return state
    
    def _route_after_escalation_check(self, state: AgentState) -> Literal["escalate", "respond"]:
        """Route based on escalation decision"""
        return "escalate" if state.requires_human else "respond"
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate AI response"""
        return self.response_generator.generate_response(state)
    
    def _handle_escalation(self, state: AgentState) -> AgentState:
        """Handle escalation to human agents"""
        if not state.escalation_info:
            return state
        
        # Create escalation ticket (in a real system, this would be sent to a ticketing system)
        escalation_ticket = state.metadata.get('escalation_ticket', {})
        
        # Log escalation
        print(f"Escalating query {state.escalation_info.escalation_id} to human agent")
        
        # In production, you would:
        # - Send ticket to human agent queue
        # - Update CRM system
        # - Send notifications to appropriate teams
        # - Schedule follow-up reminders
        
        # For now, we'll just add to metadata
        state.metadata['escalation_handled'] = True
        state.metadata['escalation_timestamp'] = datetime.now().isoformat()
        
        return state
    
    def _trim_messages(self, state: AgentState) -> AgentState:
        """Trim messages to manage context window and memory"""
        max_messages = state.max_messages
        
        if len(state.messages) <= max_messages:
            return state
        
        # Keep system messages and recent messages
        system_messages = [
            msg for msg in state.messages 
            if msg.get('role') == StateConstants.ROLE_SYSTEM
        ]
        
        # Get recent non-system messages
        non_system_messages = [
            msg for msg in state.messages 
            if msg.get('role') != StateConstants.ROLE_SYSTEM
        ]
        
        # Keep the most recent messages
        recent_messages = non_system_messages[-max_messages:]
        
        # Combine system messages and recent messages
        state.messages = system_messages + recent_messages
        
        # Add a summary message if we trimmed messages
        if len(non_system_messages) > max_messages:
            trimmed_count = len(non_system_messages) - max_messages
            summary_msg = {
                'role': StateConstants.ROLE_SYSTEM,
                'content': f"[Context: {trimmed_count} earlier messages have been summarized for efficiency]",
                'timestamp': datetime.now().isoformat(),
                'metadata': {'trimmed_messages': trimmed_count}
            }
            state.messages.insert(0, summary_msg)
        
        return state
    
    def _save_interaction(self, state: AgentState) -> AgentState:
        """Save interaction to long-term memory"""
        try:
            # Save query to database
            if state.current_query:
                from ..agents.state import QueryHistory
                
                query_entry = QueryHistory(
                    user_id=state.user_id,
                    query_text=state.current_query,
                    category=state.query_category or 'general',
                    escalated=state.requires_human,
                    resolution=self._extract_resolution(state),
                    response_time_seconds=self._calculate_response_time(state)
                )
                
                self.db_manager.save_query_history(query_entry)
            
            # Update user profile last interaction
            if state.user_profile:
                state.user_profile.metadata['last_interaction'] = datetime.now().isoformat()
                self.db_manager.save_user_profile(state.user_profile)
            
            # Update conversation context
            if state.conversation_context:
                state.conversation_context.last_activity = datetime.now()
                state.conversation_context.message_count = len(state.messages)
                self.db_manager.save_conversation_context(state.conversation_context)
            
        except Exception as e:
            print(f"Error saving interaction: {e}")
            # Don't fail the workflow if saving fails
            state.metadata['save_error'] = str(e)
        
        return state
    
    def _finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response and clean up"""
        # Calculate total processing time
        if state.processing_start_time:
            state.total_processing_time = (
                datetime.now() - state.processing_start_time
            ).total_seconds()
        
        # Update final timestamp
        state = update_state_timestamp(state)
        
        # Add completion metadata
        state.metadata['workflow_completed'] = True
        state.metadata['completion_timestamp'] = datetime.now().isoformat()
        
        return state
    
    def _extract_resolution(self, state: AgentState) -> Optional[str]:
        """Extract resolution from the conversation"""
        if not state.messages:
            return None
        
        # Get the last agent message as resolution
        for msg in reversed(state.messages):
            if msg.get('role') == StateConstants.ROLE_AGENT:
                return msg.get('content', '')
        
        return None
    
    def _calculate_response_time(self, state: AgentState) -> Optional[float]:
        """Calculate response time in seconds"""
        if state.processing_start_time and state.response_generated_at:
            return (state.response_generated_at - state.processing_start_time).total_seconds()
        return None
    
    async def process_message(self, user_id: str, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a single message from a user"""
        # Create or update state
        state = create_initial_state(
            user_id=user_id,
            thread_id=thread_id
        )
        
        # Add user message
        user_msg = {
            'role': StateConstants.ROLE_USER,
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'metadata': {}
        }
        state.messages = [user_msg]
        state.current_query = message
        
        # Process through workflow
        config = {"configurable": {"thread_id": state.thread_id}}
        result = await self.compiled_graph.ainvoke(state, config=config)
        
        return {
            'response': self._extract_agent_response(result),
            'thread_id': result.thread_id,
            'escalated': result.requires_human,
            'escalation_info': result.escalation_info.dict() if result.escalation_info else None,
            'processing_time': result.total_processing_time,
            'metadata': result.metadata
        }
    
    def process_message_sync(self, user_id: str, message: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous wrapper for process_message"""
        return asyncio.run(self.process_message(user_id, message, thread_id))
    
    def _extract_agent_response(self, state: AgentState) -> str:
        """Extract the agent's response from the state"""
        if not state.messages:
            return "I'm sorry, I couldn't process your request at this time."
        
        # Get the last agent message
        for msg in reversed(state.messages):
            if msg.get('role') == StateConstants.ROLE_AGENT:
                return msg.get('content', '')
        
        return "I'm processing your request. Please wait a moment."
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.compiled_graph.get_state(config)
            if snapshot and snapshot.values:
                return snapshot.values.get('messages', [])
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
        
        return []
    
    def handle_human_response(self, thread_id: str, human_response: str, agent_id: str) -> Dict[str, Any]:
        """Handle response from human agent (HITL)"""
        try:
            # Get current state
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = self.compiled_graph.get_state(config)
            
            if not snapshot or not snapshot.values:
                return {"error": "Thread not found"}
            
            state = snapshot.values
            
            # Add human response to conversation
            human_msg = {
                'role': StateConstants.ROLE_HUMAN,
                'content': human_response,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'agent_id': agent_id,
                    'human_response': True
                }
            }
            
            state.messages.append(human_msg)
            state.human_agent_id = agent_id
            
            # Update escalation info
            if state.escalation_info:
                state.escalation_info.assigned_agent = agent_id
            
            # Update state in graph
            self.compiled_graph.update_state(config, state)
            
            return {
                'success': True,
                'message': 'Human response added successfully',
                'thread_id': thread_id
            }
            
        except Exception as e:
            return {'error': f'Error handling human response: {e}'}
    
    def get_escalated_queries(self) -> List[Dict[str, Any]]:
        """Get all escalated queries (for admin interface)"""
        # In a real system, this would query a database or queue
        # For now, we'll return empty list
        return []
    
    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        # In a real system, this would query analytics database
        return {
            'total_conversations': 0,
            'average_response_time': 0.0,
            'escalation_rate': 0.0,
            'user_satisfaction': 0.0,
            'uptime': '99.9%'
        }
