"""
Conversation Manager for ARIA
Orchestrates multi-agent conversations and manages research workflows
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

try:
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.base import ChatAgent
    import autogen_agentchat
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Mock classes for development
    class RoundRobinGroupChat:
        def __init__(self, *args, **kwargs):
            pass
    
    class ChatAgent:
        def __init__(self, *args, **kwargs):
            pass


class AutogenConversationManager:
    """
    Advanced conversation manager for orchestrating research workflows
    """
    
    def __init__(self, 
                 research_assistant,
                 user_proxy,
                 streamlit_interface = None,
                 max_rounds: int = 10,
                 conversation_config: Dict[str, Any] = None):
        """
        Initialize the conversation manager
        
        Args:
            research_assistant: Research assistant agent
            user_proxy: User proxy agent
            streamlit_interface: Streamlit interface for UI updates
            max_rounds: Maximum conversation rounds
            conversation_config: Additional configuration
        """
        self.research_assistant = research_assistant
        self.user_proxy = user_proxy
        self.streamlit_interface = streamlit_interface
        self.max_rounds = max_rounds
        self.conversation_config = conversation_config or {}
        
        # Conversation state
        self.conversation_state = {
            "active": False,
            "current_round": 0,
            "total_messages": 0,
            "research_topic": "",
            "conversation_id": None,
            "start_time": None,
            "last_activity": None
        }
        
        # Message history
        self.message_history = []
        self.research_outputs = []
        
        # Initialize group chat if Autogen is available
        if AUTOGEN_AVAILABLE:
            self._initialize_group_chat()
    
    def _initialize_group_chat(self):
        """Initialize Autogen group chat"""
        try:
            # Use RoundRobinGroupChat for the new autogen structure
            participants = [self.user_proxy, self.research_assistant]
            self.group_chat = RoundRobinGroupChat(participants=participants)
            self.group_chat_manager = self.group_chat
        except Exception as e:
            print(f"Warning: Could not initialize group chat: {e}")
            self.group_chat = None
            self.group_chat_manager = None
    
    def initiate_research(self, research_prompt: str) -> Dict[str, Any]:
        """
        Initiate a research conversation
        
        Args:
            research_prompt: Initial research prompt
            
        Returns:
            Dictionary containing conversation initiation results
        """
        # Update conversation state
        self.conversation_state.update({
            "active": True,
            "current_round": 0,
            "total_messages": 0,
            "conversation_id": f"aria_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "start_time": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        })
        
        # Extract topic from prompt
        self.conversation_state["research_topic"] = self._extract_topic(research_prompt)
        
        try:
            if AUTOGEN_AVAILABLE and self.group_chat_manager:
                # Use Autogen group chat
                result = self._initiate_autogen_conversation(research_prompt)
            else:
                # Use custom conversation flow
                result = self._initiate_custom_conversation(research_prompt)
            
            # Update Streamlit state after conversation initialization
            self._update_streamlit_state()
            
            return {
                "success": True,
                "conversation_id": self.conversation_state["conversation_id"],
                "message": "Research conversation initiated successfully",
                "result": result
            }
            
        except Exception as e:
            self.conversation_state["active"] = False
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in initiate_research: {str(e)}")
            print(f"Full traceback: {error_details}")
            return {
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "message": f"Failed to initiate research conversation: {str(e)}"
            }
    
    def _initiate_autogen_conversation(self, prompt: str) -> Dict[str, Any]:
        """Initiate conversation using Autogen framework"""
        try:
            
            # Clear any existing conversation history before starting
            self.message_history = []
            
            # Start the conversation
            chat_result = self.user_proxy.initiate_chat(
                recipient=self.research_assistant,
                message=prompt,
                max_turns=self.max_rounds,
                silent=False
            )
            
            
            # Extract chat history and populate our message history
            # Try both dict access and attribute access for chat_history
            chat_history = []
            if isinstance(chat_result, dict) and 'chat_history' in chat_result:
                chat_history = chat_result['chat_history']
            else:
                chat_history = getattr(chat_result, 'chat_history', [])
            
            
            # Convert Autogen chat history to our message format
            for i, msg in enumerate(chat_history):
                if isinstance(msg, dict):
                    sender = "assistant" if msg.get('role') == 'assistant' else "user"
                    content = msg.get('content', '')
                    self._log_message(sender, content)
            
            
            return {
                "type": "autogen",
                "chat_result": chat_result,
                "messages": chat_history
            }
            
        except Exception as e:
            print(f"Autogen conversation error: {e}")
            return self._initiate_custom_conversation(prompt)
    
    def _initiate_custom_conversation(self, prompt: str) -> Dict[str, Any]:
        """Initiate conversation using custom flow"""
        
        # Clear any existing conversation history before starting
        self.message_history = []
        
        # Add initial user message
        self._log_message("user", prompt)
        
        # Generate initial response using research assistant
        if hasattr(self.research_assistant, 'generate_research_response'):
            research_response = self.research_assistant.generate_research_response(prompt)
        else:
            research_response = f"I'll help you research: {self.conversation_state['research_topic']}"
        
        # Add assistant response
        self._log_message("assistant", research_response)
        
        
        return {
            "type": "custom",
            "initial_response": research_response,
            "messages": self.message_history[-2:]  # Last 2 messages
        }
    
    def continue_conversation(self, user_input: str) -> Dict[str, Any]:
        """
        Continue an active conversation
        
        Args:
            user_input: User's follow-up input
            
        Returns:
            Dictionary containing conversation continuation results
        """
        if not self.conversation_state["active"]:
            return {
                "success": False,
                "error": "No active conversation",
                "message": "Please start a research conversation first"
            }
        
        try:
            # Log user input
            self._log_message("user", user_input)
            
            # Process with research assistant
            if AUTOGEN_AVAILABLE and hasattr(self.research_assistant, 'generate_reply'):
                # Use Autogen reply generation
                response = self._generate_autogen_reply(user_input)
            else:
                # Use custom response generation
                response = self._generate_custom_reply(user_input)
            
            # Log assistant response
            self._log_message("assistant", response)
            
            # Update conversation state
            self.conversation_state["current_round"] += 1
            self.conversation_state["last_activity"] = datetime.now().isoformat()
            
            self._update_streamlit_state()
            
            return {
                "success": True,
                "response": response,
                "conversation_state": self.conversation_state.copy()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to continue conversation"
            }
    
    def _generate_autogen_reply(self, user_input: str) -> str:
        """Generate reply using Autogen framework"""
        try:
            # This would use the actual Autogen reply mechanism
            # For now, return a placeholder
            return f"Research Assistant: I understand you're asking about '{user_input}'. Let me provide a comprehensive analysis..."
        except Exception as e:
            return self._generate_custom_reply(user_input)
    
    def _generate_custom_reply(self, user_input: str) -> str:
        """Generate reply using custom logic"""
        # Analyze user input for research intent
        if "subtopics" in user_input.lower():
            return self._generate_subtopics_response()
        elif "summarize" in user_input.lower():
            return self._generate_summary_response()
        elif "more detail" in user_input.lower() or "elaborate" in user_input.lower():
            return self._generate_detailed_response(user_input)
        else:
            # Use research assistant for general responses
            if hasattr(self.research_assistant, 'generate_research_response'):
                return self.research_assistant.generate_research_response(user_input)
            else:
                return f"I understand your question about '{user_input}'. Let me provide relevant research insights..."
    
    def _generate_subtopics_response(self) -> str:
        """Generate response for subtopic requests"""
        topic = self.conversation_state.get("research_topic", "the current topic")
        
        if hasattr(self.research_assistant, 'generate_subtopics'):
            subtopics = self.research_assistant.generate_subtopics(topic)
            subtopics_text = "\n".join([f"• {subtopic}" for subtopic in subtopics])
            return f"Here are the key subtopics for {topic}:\n\n{subtopics_text}\n\nWould you like me to elaborate on any of these areas?"
        else:
            return f"I can break down {topic} into several key areas for deeper investigation. Which specific aspect interests you most?"
    
    def _generate_summary_response(self) -> str:
        """Generate response for summary requests"""
        if len(self.message_history) > 2:
            return "Based on our research conversation so far, here are the key insights:\n\n• Main findings and conclusions\n• Important patterns identified\n• Areas requiring further investigation\n\nWould you like me to focus on any particular aspect?"
        else:
            return "We're just getting started with the research. Let me provide an initial overview of the topic first."
    
    def _generate_detailed_response(self, user_input: str) -> str:
        """Generate detailed response based on user input"""
        return f"You've asked for more detail about '{user_input}'. Let me provide a comprehensive analysis with multiple perspectives, current developments, and practical implications..."
    
    def end_conversation(self) -> Dict[str, Any]:
        """
        End the current conversation
        
        Returns:
            Dictionary containing conversation summary
        """
        if not self.conversation_state["active"]:
            return {"success": False, "message": "No active conversation to end"}
        
        # Update conversation state
        self.conversation_state["active"] = False
        end_time = datetime.now().isoformat()
        
        # Generate conversation summary
        summary = self._generate_conversation_summary(end_time)
        
        # Log conversation end
        self._log_message("system", "Conversation ended", {"summary": summary})
        
        self._update_streamlit_state()
        
        return {
            "success": True,
            "message": "Conversation ended successfully",
            "summary": summary
        }
    
    def _generate_conversation_summary(self, end_time: str) -> Dict[str, Any]:
        """Generate summary of the conversation"""
        start_time = datetime.fromisoformat(self.conversation_state["start_time"])
        end_time_dt = datetime.fromisoformat(end_time)
        duration = end_time_dt - start_time
        
        return {
            "conversation_id": self.conversation_state["conversation_id"],
            "research_topic": self.conversation_state["research_topic"],
            "duration_minutes": int(duration.total_seconds() / 60),
            "total_rounds": self.conversation_state["current_round"],
            "total_messages": len(self.message_history),
            "start_time": self.conversation_state["start_time"],
            "end_time": end_time,
            "message_types": self._analyze_message_types()
        }
    
    def _analyze_message_types(self) -> Dict[str, int]:
        """Analyze types of messages in the conversation"""
        types = {"user": 0, "assistant": 0, "system": 0}
        for msg in self.message_history:
            msg_type = msg.get("sender", "unknown")
            if msg_type in types:
                types[msg_type] += 1
        return types
    
    def _extract_topic(self, prompt: str) -> str:
        """Extract research topic from prompt"""
        # Enhanced topic extraction
        prompt_lower = prompt.lower()
        
        # Look for "Topic: " pattern first (most specific)
        if "topic:" in prompt_lower:
            topic_start = prompt_lower.find("topic:") + len("topic:")
            topic_line = prompt[topic_start:].split('\n')[0].strip()
            if topic_line:
                return topic_line
        
        # Look for other patterns
        if "research on" in prompt_lower:
            topic = prompt_lower.split("research on")[1].split("\n")[0].split(".")[0].strip()
            return topic if topic else "research topic"
        elif "about" in prompt_lower:
            topic = prompt_lower.split("about")[1].split("\n")[0].split(".")[0].strip() 
            return topic if topic else "research topic"
        
        # Fallback: look for the first substantial line that might be a topic
        lines = prompt.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200 and not line.startswith('Please'):
                return line
        
        # Last resort
        return prompt[:100] + "..." if len(prompt) > 100 else prompt
    
    def _log_message(self, sender: str, content: str, metadata: Dict[str, Any] = None):
        """Log a message to the conversation history"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender,
            "content": content,
            "metadata": metadata or {},
            "round": self.conversation_state["current_round"]
        }
        
        self.message_history.append(message)
        self.conversation_state["total_messages"] += 1
        
        # Also log to research outputs if it's substantial content
        if sender == "assistant" and len(content) > 100:
            self.research_outputs.append({
                "timestamp": message["timestamp"],
                "content": content,
                "type": "research_response"
            })
    
    def _update_streamlit_state(self):
        """Update Streamlit session state with conversation data"""
        
        if self.streamlit_interface and hasattr(self.streamlit_interface, 'session_state'):
            
            # Update conversation messages
            if not hasattr(self.streamlit_interface.session_state, 'aria_conversation_messages'):
                self.streamlit_interface.session_state.aria_conversation_messages = []
            
            # Sync message history with Streamlit
            streamlit_messages = []
            for msg in self.message_history:
                if msg["sender"] != "system":  # Skip system messages in UI
                    streamlit_message = {
                        'sender': msg["sender"],
                        'content': msg["content"],
                        'timestamp': msg["timestamp"]
                    }
                    streamlit_messages.append(streamlit_message)
            
            self.streamlit_interface.session_state.aria_conversation_messages = streamlit_messages
            
            
            # Update research state
            if hasattr(self.streamlit_interface.session_state, 'aria_research_state'):
                self.streamlit_interface.session_state.aria_research_state.update({
                    'conversation_active': self.conversation_state["active"],
                    'current_topic': self.conversation_state["research_topic"],
                    'session_id': self.conversation_state["conversation_id"]
                })
    
    def get_conversation_data(self) -> Dict[str, Any]:
        """
        Get complete conversation data for export
        
        Returns:
            Dictionary containing all conversation data
        """
        return {
            "conversation_state": self.conversation_state,
            "message_history": self.message_history,
            "research_outputs": self.research_outputs,
            "configuration": self.conversation_config,
            "export_timestamp": datetime.now().isoformat()
        }
    
    def load_conversation_data(self, data: Dict[str, Any]):
        """
        Load conversation data from export
        
        Args:
            data: Previously exported conversation data
        """
        self.conversation_state = data.get("conversation_state", {})
        self.message_history = data.get("message_history", [])
        self.research_outputs = data.get("research_outputs", [])
        self.conversation_config = data.get("configuration", {})
        
        self._update_streamlit_state()


def create_conversation_manager(research_assistant, 
                              user_proxy, 
                              streamlit_interface = None,
                              config: Dict[str, Any] = None) -> AutogenConversationManager:
    """
    Factory function to create a conversation manager
    
    Args:
        research_assistant: Research assistant agent
        user_proxy: User proxy agent
        streamlit_interface: Streamlit interface
        config: Configuration dictionary
        
    Returns:
        Configured AutogenConversationManager instance
    """
    if config is None:
        config = {}
    
    return AutogenConversationManager(
        research_assistant=research_assistant,
        user_proxy=user_proxy,
        streamlit_interface=streamlit_interface,
        max_rounds=config.get("max_rounds", 10),
        conversation_config=config
    )


def get_conversation_manager_capabilities() -> Dict[str, Any]:
    """
    Get information about conversation manager capabilities
    
    Returns:
        Dictionary describing available capabilities
    """
    return {
        "core_features": [
            "Multi-agent conversation orchestration",
            "Research workflow management",
            "Conversation state tracking",
            "Message history logging",
            "Streamlit integration",
            "Export/import functionality"
        ],
        "conversation_modes": ["autogen_groupchat", "custom_flow"],
        "supported_agents": ["research_assistant", "user_proxy"],
        "export_formats": ["json", "conversation_log"],
        "autogen_integration": AUTOGEN_AVAILABLE
    }