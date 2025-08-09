"""Minimal Streamlit UI for Customer Support Agent"""

import streamlit as st
import uuid
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Customer Support",
    page_icon="🎧",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Main app
st.title("🎧 Customer Support Agent")
st.markdown("Welcome! I'm here to help you with any questions or issues.")

# Sidebar
with st.sidebar:
    st.title("Info")
    st.write(f"User ID: {st.session_state.user_id[:8]}...")
    st.write(f"Messages: {len(st.session_state.messages)}")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display messages
for msg in st.session_state.messages:
    role = msg.get('role', 'unknown')
    content = msg.get('content', '')
    
    if role == 'user':
        col1, col2 = st.columns([1, 4])
        with col2:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {content}
            </div>
            """, unsafe_allow_html=True)
    elif role == 'agent':
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div class="chat-message agent-message">
                <strong>Agent:</strong> {content}
            </div>
            """, unsafe_allow_html=True)

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message:", placeholder="How can I help you?")
    submit = st.form_submit_button("Send")
    
    if submit and user_input:
        # Add user message
        st.session_state.messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        response = "Thank you for your message. I'm here to help you with any questions or issues you may have. How can I assist you today?"
        
        # Add agent response
        st.session_state.messages.append({
            'role': 'agent',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        st.rerun()
