"""
Enhanced User Proxy Agent for ARIA
Streamlit-integrated user proxy with human-in-the-loop control
"""

import os
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

try:
    from autogen_agentchat.agents import UserProxyAgent
    from autogen_agentchat.base import ChatAgent
    import autogen_agentchat
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

# Simplified implementation that doesn't require complex autogen setup
class BaseUserProxy:
    """Base user proxy class for simplified implementation"""
    def __init__(self, name: str = "user_proxy", **kwargs):
        self.name = name
        self.kwargs = kwargs


class StreamlitUserProxy(UserProxyAgent if AUTOGEN_AVAILABLE else BaseUserProxy):
    """
    Enhanced User Proxy Agent integrated with Streamlit for human-in-the-loop control
    """
    
    def __init__(self,
                 name: str = "user_proxy",
                 streamlit_interface = None,
                 human_input_mode: str = "ALWAYS",
                 max_consecutive_auto_reply: int = 0,
                 code_execution_config: Dict[str, Any] = None,
                 conversation_log: List[Dict] = None,
                 **kwargs):
        """
        Initialize the Streamlit-integrated User Proxy
        
        Args:
            name: Agent name
            streamlit_interface: Streamlit interface object for UI integration
            human_input_mode: Mode for human input (ALWAYS, NEVER, TERMINATE)
            max_consecutive_auto_reply: Maximum consecutive auto-replies
            code_execution_config: Configuration for code execution
            conversation_log: Existing conversation log
            **kwargs: Additional arguments for UserProxyAgent
        """
        self.streamlit_interface = streamlit_interface
        self.conversation_log = conversation_log or []
        self.session_metadata = {
            "session_start": datetime.now().isoformat(),
            "total_interactions": 0,
            "user_inputs": 0,
            "agent_responses": 0
        }
        
        # Default code execution config
        if code_execution_config is None:
            code_execution_config = {
                "work_dir": "research_workspace",
                "use_docker": False,
                "timeout": 60,
                "last_n_messages": 3
            }
        
        # Initialize parent class (UserProxyAgent or BaseUserProxy)
        if AUTOGEN_AVAILABLE:
            # Note: New autogen might have different parameter names
            super().__init__(name=name, **kwargs)
        else:
            super().__init__(name=name, **kwargs)
        
        # Store configuration for our implementation
        self.human_input_mode = human_input_mode
        self.max_consecutive_auto_reply = max_consecutive_auto_reply
        self.code_execution_config = code_execution_config
    
    def _is_termination_message(self, msg: Dict[str, Any]) -> bool:
        """
        Determine if a message should terminate the conversation
        
        Args:
            msg: Message dictionary
            
        Returns:
            True if conversation should terminate
        """
        content = msg.get("content", "").upper()
        termination_keywords = ["TERMINATE", "END CONVERSATION", "STOP RESEARCH", "EXIT"]
        return any(keyword in content for keyword in termination_keywords)
    
    def get_human_input(self, prompt: str) -> str:
        """
        Get human input through Streamlit interface
        
        Args:
            prompt: Prompt to display to user
            
        Returns:
            User input string
        """
        if self.streamlit_interface is None:
            # Fallback to default behavior
            return input(prompt)
        
        # Use Streamlit interface for input
        try:
            # This would integrate with Streamlit session state
            user_input = self._get_streamlit_input(prompt)
            self.session_metadata["user_inputs"] += 1
            self._log_interaction("user_input", user_input, prompt)
            return user_input
        except Exception as e:
            # Fallback if Streamlit integration fails
            return f"Error getting user input: {str(e)}"
    
    def _get_streamlit_input(self, prompt: str) -> str:
        """
        Get input through Streamlit interface
        
        Args:
            prompt: Prompt for user
            
        Returns:
            User input
        """
        # This is a placeholder - actual implementation would integrate with
        # Streamlit session state and UI components
        if hasattr(self.streamlit_interface, 'session_state'):
            # Check for pending user input in session state
            if hasattr(self.streamlit_interface.session_state, 'pending_user_input'):
                user_input = self.streamlit_interface.session_state.pending_user_input
                delattr(self.streamlit_interface.session_state, 'pending_user_input')
                return user_input
        
        # Return a default response if no input available
        return "Please continue with the research."
    
    def process_agent_response(self, response: str, agent_name: str = "assistant"):
        """
        Process and log agent responses
        
        Args:
            response: Agent response
            agent_name: Name of the responding agent
        """
        self.session_metadata["agent_responses"] += 1
        self._log_interaction("agent_response", response, agent_name)
        
        # Update Streamlit interface if available
        if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
            # Add to conversation messages
            if not hasattr(self.streamlit_interface.session_state, 'aria_conversation_messages'):
                self.streamlit_interface.session_state.aria_conversation_messages = []
            
            self.streamlit_interface.session_state.aria_conversation_messages.append({
                'sender': agent_name,
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
    
    def _log_interaction(self, interaction_type: str, content: str, metadata: Any = None):
        """
        Log interactions for session tracking
        
        Args:
            interaction_type: Type of interaction (user_input, agent_response, etc.)
            content: Interaction content
            metadata: Additional metadata
        """
        self.conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "content": content,
            "metadata": metadata,
            "session_metadata": self.session_metadata.copy()
        })
        
        self.session_metadata["total_interactions"] += 1
    
    def set_research_preferences(self, preferences: Dict[str, Any]):
        """
        Set user research preferences
        
        Args:
            preferences: Dictionary containing user preferences
        """
        self.research_preferences = {
            "preferred_depth": preferences.get("depth", "intermediate"),
            "preferred_audience": preferences.get("audience", "general"),
            "max_response_length": preferences.get("max_length", 2000),
            "include_sources": preferences.get("sources", True),
            "preferred_format": preferences.get("format", "structured"),
            "auto_follow_up": preferences.get("auto_follow_up", False)
        }
        
        self._log_interaction("preferences_update", json.dumps(self.research_preferences))
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation
        
        Returns:
            Dictionary containing conversation summary
        """
        return {
            "session_metadata": self.session_metadata,
            "total_messages": len(self.conversation_log),
            "conversation_duration": self._calculate_duration(),
            "user_engagement": self._calculate_engagement(),
            "last_interaction": self.conversation_log[-1] if self.conversation_log else None
        }
    
    def _calculate_duration(self) -> str:
        """Calculate conversation duration"""
        if not self.conversation_log:
            return "0 minutes"
        
        start_time = datetime.fromisoformat(self.session_metadata["session_start"])
        current_time = datetime.now()
        duration = current_time - start_time
        
        minutes = int(duration.total_seconds() / 60)
        return f"{minutes} minutes"
    
    def _calculate_engagement(self) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        total_interactions = self.session_metadata["total_interactions"]
        user_inputs = self.session_metadata["user_inputs"]
        agent_responses = self.session_metadata["agent_responses"]
        
        return {
            "user_interaction_ratio": user_inputs / max(total_interactions, 1),
            "response_ratio": agent_responses / max(total_interactions, 1),
            "average_response_time": "N/A"  # Would need response time tracking
        }
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Export the complete conversation log
        
        Returns:
            Dictionary containing complete conversation data
        """
        return {
            "session_id": f"user_proxy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "session_metadata": self.session_metadata,
            "conversation_log": self.conversation_log,
            "conversation_summary": self.get_conversation_summary(),
            "preferences": getattr(self, 'research_preferences', {}),
            "export_timestamp": datetime.now().isoformat()
        }
    
    def handle_streamlit_callback(self, callback_type: str, data: Any):
        """
        Handle callbacks from Streamlit interface
        
        Args:
            callback_type: Type of callback
            data: Callback data
        """
        if callback_type == "user_input":
            self.process_user_input(data)
        elif callback_type == "session_reset":
            self.reset_session()
        elif callback_type == "export_request":
            return self.export_conversation()
        elif callback_type == "preferences_update":
            self.set_research_preferences(data)
    
    def process_user_input(self, user_input: str):
        """
        Process user input and update session state
        
        Args:
            user_input: User input string
        """
        self._log_interaction("user_input", user_input)
        
        # Update Streamlit session state if available
        if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
            if not hasattr(self.streamlit_interface.session_state, 'aria_conversation_messages'):
                self.streamlit_interface.session_state.aria_conversation_messages = []
            
            self.streamlit_interface.session_state.aria_conversation_messages.append({
                'sender': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
    
    def reset_session(self):
        """Reset the conversation session"""
        self.conversation_log = []
        self.session_metadata = {
            "session_start": datetime.now().isoformat(),
            "total_interactions": 0,
            "user_inputs": 0,
            "agent_responses": 0
        }
        
        # Clear Streamlit session state if available
        if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
            if hasattr(self.streamlit_interface.session_state, 'aria_conversation_messages'):
                self.streamlit_interface.session_state.aria_conversation_messages = []
    
    def initiate_chat(self, recipient, message: str, **kwargs) -> Dict[str, Any]:
        """
        Initiate a chat conversation with another agent
        
        Args:
            recipient: The agent to chat with
            message: Initial message to send
            **kwargs: Additional parameters (max_turns, silent, etc.)
            
        Returns:
            Dictionary containing chat results
        """
        try:
            # Log the initiation
            self._log_interaction("chat_initiation", message, {"recipient": getattr(recipient, 'name', 'unknown')})
            
            # Add initial message to conversation
            print(f"🔍 Debug: initiate_chat called with message: {message}")
            print(f"🔍 Debug: streamlit_interface = {self.streamlit_interface}")
            
            if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
                print(f"🔍 Debug: Streamlit interface available, adding user message")
                if not hasattr(self.streamlit_interface.session_state, 'aria_conversation_messages'):
                    self.streamlit_interface.session_state.aria_conversation_messages = []
                    print(f"🔍 Debug: Created new aria_conversation_messages list")
                
                # Add user message
                self.streamlit_interface.session_state.aria_conversation_messages.append({
                    'sender': 'user',
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"🔍 Debug: Added user message, total messages: {len(self.streamlit_interface.session_state.aria_conversation_messages)}")
            else:
                print(f"🔍 Debug: Streamlit interface not available")
            
            # Generate response from recipient
            if hasattr(recipient, 'generate_research_response'):
                response = recipient.generate_research_response(message)
            elif hasattr(recipient, 'generate_reply'):
                response = recipient.generate_reply(message)
            else:
                response = f"I understand you want to research: {message}. Let me help you with that."
            
            # Add agent response to conversation
            print(f"🔍 Debug: Generated response: {response[:100]}...")
            
            if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
                self.streamlit_interface.session_state.aria_conversation_messages.append({
                    'sender': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"🔍 Debug: Added assistant response, total messages: {len(self.streamlit_interface.session_state.aria_conversation_messages)}")
            else:
                print(f"🔍 Debug: Streamlit interface not available for assistant response")
            
            # Log the response
            self._log_interaction("agent_response", response, {"recipient": getattr(recipient, 'name', 'assistant')})
            
            # Return autogen-compatible result structure
            chat_history = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            
            return {
                "chat_history": chat_history,
                "summary": f"Research conversation initiated successfully",
                "cost": {"usage_including_cached_inference": {"total_cost": 0}},
                "human_input": []
            }
            
        except Exception as e:
            error_msg = f"Error in initiate_chat: {str(e)}"
            self._log_interaction("chat_error", error_msg)
            raise Exception(error_msg)


def create_enhanced_user_proxy(streamlit_interface = None, 
                             config: Dict[str, Any] = None) -> StreamlitUserProxy:
    """
    Factory function to create a configured Streamlit user proxy
    
    Args:
        streamlit_interface: Streamlit interface object
        config: Configuration dictionary
        
    Returns:
        Configured StreamlitUserProxy instance
    """
    if config is None:
        config = {}
    
    return StreamlitUserProxy(
        name=config.get("name", "user_proxy"),
        streamlit_interface=streamlit_interface,
        human_input_mode=config.get("human_input_mode", "ALWAYS"),
        max_consecutive_auto_reply=config.get("max_consecutive_auto_reply", 0),
        code_execution_config=config.get("code_execution_config")
    )


def get_user_proxy_capabilities() -> Dict[str, Any]:
    """
    Get information about user proxy capabilities
    
    Returns:
        Dictionary describing available capabilities
    """
    return {
        "core_features": [
            "Streamlit integration",
            "Human-in-the-loop control",
            "Conversation logging",
            "Session management",
            "User preference tracking",
            "Real-time interaction processing"
        ],
        "input_modes": ["ALWAYS", "NEVER", "TERMINATE"],
        "callback_types": ["user_input", "session_reset", "export_request", "preferences_update"],
        "export_formats": ["json", "conversation_log"],
        "streamlit_integration": True,
        "autogen_integration": AUTOGEN_AVAILABLE
    }