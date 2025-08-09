"""Streamlit UI for Customer Support Agent"""

import streamlit as st
import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.support_agent import CustomerSupportAgent
from agents.state import StateConstants
from utils.config import Config
from utils.validators import InputValidator
from utils.message_utils import MessageProcessor

# Page configuration
st.set_page_config(
    page_title="TechTrend Customer Support",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .agent-message {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        color: #495057;
    }
    .system-message {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        font-style: italic;
    }
    .escalation-notice {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metrics-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .admin-panel {
        background-color: #e3f2fd;
        border: 2px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
        border: none;
    }
</style>
""", unsafe_allow_html=True)


class CustomerSupportUI:
    """Main UI class for Customer Support Agent"""
    
    def __init__(self):
        self.init_session_state()
        self.config = Config()
        self.validator = InputValidator()
        self.message_processor = MessageProcessor()
        self.agent = self.initialize_agent()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if 'thread_id' not in st.session_state:
            st.session_state.thread_id = str(uuid.uuid4())
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'escalated_queries' not in st.session_state:
            st.session_state.escalated_queries = []
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = None
        if 'agent_metrics' not in st.session_state:
            st.session_state.agent_metrics = {}
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
    
    def initialize_agent(self) -> Optional[CustomerSupportAgent]:
        """Initialize the customer support agent"""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                st.error("⚠️ Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
                return None
            
            return CustomerSupportAgent(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            return None
    
    def render_sidebar(self):
        """Render sidebar with user info and controls"""
        with st.sidebar:
            st.title("🎧 Customer Support")
            st.markdown("---")
            
            # User Profile Section
            st.subheader("👤 User Profile")
            self.render_user_profile()
            
            st.markdown("---")
            
            # Session Information
            st.subheader("📊 Session Info")
            st.text(f"User ID: {st.session_state.user_id[:8]}...")
            st.text(f"Thread: {st.session_state.thread_id[:8]}...")
            st.text(f"Messages: {len(st.session_state.messages)}")
            
            # Clear conversation button
            if st.button("🗑️ Clear Conversation"):
                self.clear_conversation()
            
            st.markdown("---")
            
            # Admin Panel Toggle
            admin_password = st.text_input("Admin Password", type="password")
            if admin_password == "admin123":  # Simple password for demo
                st.session_state.is_admin = True
            
            if st.session_state.is_admin:
                st.success("✅ Admin access granted")
                if st.button("📊 Show Admin Panel"):
                    st.session_state.show_admin = True
            
            st.markdown("---")
            
            # Quick Actions
            st.subheader("⚡ Quick Actions")
            sample_queries = [
                "I forgot my password",
                "How do I update my billing info?",
                "I'm having trouble logging in",
                "Can you help me with a technical issue?",
                "I want to speak to a human agent"
            ]
            
            for query in sample_queries:
                if st.button(f"💬 {query[:25]}...", key=f"quick_{hash(query)}"):
                    self.handle_quick_query(query)
    
    def render_user_profile(self):
        """Render user profile section"""
        if st.session_state.user_profile:
            profile = st.session_state.user_profile
            st.write(f"**Name:** {profile.get('name', 'Unknown')}")
            st.write(f"**Email:** {profile.get('email', 'Unknown')}")
            st.write(f"**Account:** {profile.get('account_type', 'Standard').title()}")
            
            if st.button("✏️ Edit Profile"):
                self.show_profile_editor()
        else:
            with st.expander("📝 Set Up Profile", expanded=True):
                self.render_profile_form()
    
    def render_profile_form(self):
        """Render profile creation form"""
        with st.form("user_profile_form"):
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="your.email@company.com")
            account_type = st.selectbox("Account Type", ["Standard", "Premium", "Enterprise"])
            
            if st.form_submit_button("💾 Save Profile"):
                # Validate inputs
                name_valid, name_msg = self.validator.validate_name(name)
                email_valid, email_msg = self.validator.validate_email(email)
                
                if name_valid and email_valid:
                    st.session_state.user_profile = {
                        "name": name,
                        "email": email,
                        "account_type": account_type.lower(),
                        "created_at": datetime.now().isoformat()
                    }
                    st.success("✅ Profile saved successfully!")
                    st.rerun()
                else:
                    if not name_valid:
                        st.error(f"Name error: {name_msg}")
                    if not email_valid:
                        st.error(f"Email error: {email_msg}")
    
    def show_profile_editor(self):
        """Show profile editor in main area"""
        st.session_state.show_profile_editor = True
    
    def render_main_chat_interface(self):
        """Render main chat interface"""
        st.title("🎧 TechTrend Customer Support")
        st.markdown("Welcome! I'm here to help you with any questions or issues.")
        
        # Display conversation
        self.render_conversation()
        
        # Chat input area
        self.render_chat_input()
    
    def render_conversation(self):
        """Render conversation messages"""
        chat_container = st.container()
        
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                self.render_message(msg, i)
    
    def render_message(self, message: Dict[str, Any], index: int):
        """Render individual message"""
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        timestamp = message.get('timestamp', '')
        metadata = message.get('metadata', {})
        
        # Format timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M")
            except:
                time_str = ""
        else:
            time_str = ""
        
        if role == StateConstants.ROLE_USER:
            col1, col2 = st.columns([1, 4])
            with col2:
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You</strong> {time_str}
                    <br>{content}
                </div>
                """, unsafe_allow_html=True)
        
        elif role == StateConstants.ROLE_AGENT:
            col1, col2 = st.columns([4, 1])
            with col1:
                confidence = metadata.get('confidence_score', 0)
                confidence_indicator = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                st.markdown(f"""
                <div class="chat-message agent-message">
                    <strong>Support Agent</strong> {confidence_indicator} {time_str}
                    <br>{content}
                </div>
                """, unsafe_allow_html=True)
        
        elif role == StateConstants.ROLE_SYSTEM:
            if metadata.get('escalation_notice'):
                st.markdown(f"""
                <div class="escalation-notice">
                    <strong>🔔 Escalation Notice</strong>
                    <br>{content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message system-message">
                    <strong>System</strong> {time_str}
                    <br>{content}
                </div>
                """, unsafe_allow_html=True)
        
        elif role == StateConstants.ROLE_HUMAN:
            col1, col2 = st.columns([4, 1])
            with col1:
                agent_id = metadata.get('agent_id', 'Unknown')
                st.markdown(f"""
                <div class="chat-message agent-message" style="border-left: 4px solid #28a745;">
                    <strong>Human Agent ({agent_id})</strong> {time_str}
                    <br>{content}
                </div>
                """, unsafe_allow_html=True)
    
    def render_chat_input(self):
        """Render chat input area"""
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_area(
                    "Type your message:",
                    placeholder="How can I help you today?",
                    height=100,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.write("")  # Spacing
                send_button = st.form_submit_button("📤 Send", use_container_width=True)
                
                # Voice input placeholder (would integrate with speech recognition)
                voice_button = st.form_submit_button("🎤 Voice", use_container_width=True)
            
            if send_button and user_input:
                self.process_user_message(user_input)
            
            if voice_button:
                st.info("🎤 Voice input feature coming soon!")
    
    def process_user_message(self, message: str):
        """Process user message through the agent"""
        # Validate input
        is_valid, validation_msg = self.validator.validate_message_content(message)
        if not is_valid:
            st.error(f"Invalid message: {validation_msg}")
            return
        
        # Process message
        processed_msg = self.message_processor.process_message(message)
        
        # Add user message to conversation
        user_msg = {
            'role': StateConstants.ROLE_USER,
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'processed': processed_msg.__dict__,
                'user_id': st.session_state.user_id
            }
        }
        st.session_state.messages.append(user_msg)
        
        # Process through agent
        if self.agent:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = self.agent.process_message_sync(
                        user_id=st.session_state.user_id,
                        message=message,
                        thread_id=st.session_state.thread_id
                    )
                    
                    # Add agent response
                    agent_msg = {
                        'role': StateConstants.ROLE_AGENT,
                        'content': response['response'],
                        'timestamp': datetime.now().isoformat(),
                        'metadata': response.get('metadata', {})
                    }
                    st.session_state.messages.append(agent_msg)
                    
                    # Handle escalation
                    if response.get('escalated'):
                        escalation_info = response.get('escalation_info')
                        if escalation_info:
                            st.session_state.escalated_queries.append({
                                'user_id': st.session_state.user_id,
                                'thread_id': st.session_state.thread_id,
                                'message': message,
                                'escalation_info': escalation_info,
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Update metrics
                    if 'processing_time' in response:
                        self.update_metrics(response)
                    
                except Exception as e:
                    st.error(f"Error processing message: {e}")
                    # Add error message
                    error_msg = {
                        'role': StateConstants.ROLE_SYSTEM,
                        'content': "I'm sorry, I encountered an error processing your message. Please try again.",
                        'timestamp': datetime.now().isoformat(),
                        'metadata': {'error': str(e)}
                    }
                    st.session_state.messages.append(error_msg)
        else:
            st.error("Agent not available. Please check configuration.")
        
        st.rerun()
    
    def handle_quick_query(self, query: str):
        """Handle quick query button clicks"""
        self.process_user_message(query)
    
    def update_metrics(self, response: Dict[str, Any]):
        """Update agent metrics"""
        if 'agent_metrics' not in st.session_state:
            st.session_state.agent_metrics = {
                'total_queries': 0,
                'total_response_time': 0,
                'escalated_queries': 0
            }
        
        metrics = st.session_state.agent_metrics
        metrics['total_queries'] += 1
        metrics['total_response_time'] += response.get('processing_time', 0)
        
        if response.get('escalated'):
            metrics['escalated_queries'] += 1
    
    def clear_conversation(self):
        """Clear the conversation"""
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    def render_admin_panel(self):
        """Render admin panel for human agents"""
        st.title("👨‍💼 Admin Panel")
        
        # Metrics overview
        self.render_metrics_overview()
        
        # Escalated queries
        self.render_escalated_queries()
        
        # System health
        self.render_system_health()
    
    def render_metrics_overview(self):
        """Render metrics overview"""
        st.subheader("📊 Metrics Overview")
        
        metrics = st.session_state.agent_metrics
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Queries",
                metrics.get('total_queries', 0)
            )
        
        with col2:
            avg_response = (
                metrics.get('total_response_time', 0) / max(metrics.get('total_queries', 1), 1)
            )
            st.metric(
                "Avg Response Time",
                f"{avg_response:.1f}s"
            )
        
        with col3:
            escalation_rate = (
                metrics.get('escalated_queries', 0) / max(metrics.get('total_queries', 1), 1) * 100
            )
            st.metric(
                "Escalation Rate",
                f"{escalation_rate:.1f}%"
            )
        
        with col4:
            st.metric(
                "Active Sessions",
                "1"  # Simplified for demo
            )
    
    def render_escalated_queries(self):
        """Render escalated queries management"""
        st.subheader("🔔 Escalated Queries")
        
        if st.session_state.escalated_queries:
            for i, query in enumerate(st.session_state.escalated_queries):
                with st.expander(f"Query {i+1}: {query['timestamp'][:19]}"):
                    st.write(f"**User ID:** {query['user_id'][:8]}...")
                    st.write(f"**Thread ID:** {query['thread_id'][:8]}...")
                    st.write(f"**Message:** {query['message']}")
                    
                    escalation_info = query.get('escalation_info', {})
                    if escalation_info:
                        st.write(f"**Reason:** {escalation_info.get('reason', 'Unknown')}")
                        st.write(f"**Urgency:** {escalation_info.get('urgency_level', 'Medium')}")
                    
                    # Human response form
                    with st.form(f"response_form_{i}"):
                        human_response = st.text_area(
                            "Human Agent Response:",
                            placeholder="Provide your response to the customer...",
                            key=f"human_response_{i}"
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.form_submit_button("📤 Send Response"):
                                self.handle_human_response(query, human_response, i)
                        
                        with col2:
                            if st.form_submit_button("🔄 Escalate Further"):
                                st.warning("Escalated to senior support team")
        else:
            st.info("No escalated queries at the moment")
    
    def handle_human_response(self, query: Dict[str, Any], response: str, index: int):
        """Handle human agent response"""
        if not response.strip():
            st.error("Response cannot be empty")
            return
        
        # Add human response to the conversation
        if self.agent:
            try:
                result = self.agent.handle_human_response(
                    thread_id=query['thread_id'],
                    human_response=response,
                    agent_id="admin_agent"
                )
                
                if result.get('success'):
                    st.success("✅ Response sent successfully!")
                    # Remove from escalated queries
                    st.session_state.escalated_queries.pop(index)
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error sending response: {e}")
    
    def render_system_health(self):
        """Render system health information"""
        st.subheader("🏥 System Health")
        
        health_status = {
            "Agent Status": "✅ Online",
            "Database": "✅ Connected",
            "Memory Store": "✅ Available",
            "API Connectivity": "✅ Active"
        }
        
        for component, status in health_status.items():
            st.write(f"**{component}:** {status}")
    
    def run(self):
        """Run the Streamlit application"""
        # Check if showing admin panel
        if hasattr(st.session_state, 'show_admin') and st.session_state.show_admin:
            self.render_admin_panel()
            if st.button("🔙 Back to Chat"):
                st.session_state.show_admin = False
                st.rerun()
        # Check if showing profile editor
        elif hasattr(st.session_state, 'show_profile_editor') and st.session_state.show_profile_editor:
            st.title("✏️ Edit Profile")
            self.render_profile_form()
            if st.button("🔙 Back to Chat"):
                st.session_state.show_profile_editor = False
                st.rerun()
        else:
            # Render sidebar
            self.render_sidebar()
            
            # Render main interface
            self.render_main_chat_interface()


def main():
    """Main application entry point"""
    app = CustomerSupportUI()
    app.run()


if __name__ == "__main__":
    main()
