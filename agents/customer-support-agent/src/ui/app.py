"""Minimal Streamlit UI for Customer Support Agent"""

import streamlit as st
import uuid
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
env_path = root_dir / '.env'

# Debug: Check if .env file exists
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: try loading from current working directory
    load_dotenv()

# Get Google API key (using working Google API key)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Initialize LLM if API key is available
llm = None
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
    except ImportError:
        st.error("google-generativeai package not installed. Please install it with: pip install google-generativeai")
    except Exception as e:
        st.error(f"Error initializing Google Gemini: {e}")

def generate_response(user_input: str, conversation_history: list) -> str:
    """Generate AI response using Google Gemini"""
    if not llm:
        return "I'm sorry, but I'm currently unable to process your request due to API configuration issues. Please contact our technical support team."
    
    try:
        # Build conversation context
        context = "You are a helpful customer support agent. Be friendly, professional, and try to resolve the customer's issues. If you cannot help with something, politely explain and suggest they contact a human agent.\n\n"
        
        # Add recent conversation history for context
        recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        for msg in recent_messages:
            role = "Customer" if msg['role'] == 'user' else "Agent"
            context += f"{role}: {msg['content']}\n"
        
        context += f"\nCustomer: {user_input}\nAgent:"
        
        # Generate response
        response = llm.generate_content(context)
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "API key expired" in error_msg or "API_KEY_INVALID" in error_msg or "400" in error_msg:
            return "🔑 I apologize, but our API key has an issue. Please contact our technical team."
        else:
            return f"I apologize, but I'm experiencing technical difficulties. Please try again or contact human support. Error: {error_msg}"

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
    
    # API Status
    if GOOGLE_API_KEY:
        if llm:
            st.success("✅ Google API Connected")
        else:
            st.error("❌ API Connection Failed")
            st.warning("API key may be invalid")
        masked_key = GOOGLE_API_KEY[:8] + "*" * (len(GOOGLE_API_KEY) - 12) + GOOGLE_API_KEY[-4:]
        st.info(f"Using Google key: {masked_key}")
    else:
        st.error("❌ Google API Key Missing")
        st.warning("Add GOOGLE_API_KEY to .env file")
        st.markdown("[Get API Key →](https://aistudio.google.com/app/apikey)")
    
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
        
        # Generate AI response
        with st.spinner("Generating response..."):
            response = generate_response(user_input, st.session_state.messages)
        
        # Add agent response
        st.session_state.messages.append({
            'role': 'agent',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        st.rerun()
