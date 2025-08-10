import streamlit as st
import asyncio
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, Optional, Dict
import re
from dotenv import load_dotenv
import os
import requests
import logging
from pathlib import Path

# === CONFIG ===
# Load from shared .env at repository root
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
env_path = root_dir / '.env'

load_dotenv(env_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONSTANTS ===
INTENT_DETECTION_NODE = "Intent Detection"

# === STATE ===
class FinanceState(TypedDict):
    user_input: str
    intent: Optional[str]
    data: Optional[dict]
    user_profile: Optional[Dict[str, str]]
    short_term_memory: Optional[Dict[str, str]]
    long_term_memory: Optional[Dict[str, str]]
    hitl_flag: Optional[bool]

# === LLM ===
if GROQ_API_KEY:
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
else:
    llm = None

# === STREAMLIT UI ===
st.set_page_config(page_title="💸 FinAdvise", page_icon="💬", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "long_term_memory" not in st.session_state:
    st.session_state.long_term_memory = {}
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "age": "30",
        "income": "50000",
        "goals": "Save for retirement and emergency fund",
        "risk tolerance": "moderate"
    }

# Sidebar for user profile
with st.sidebar:
    st.header("👤 User Profile")
    st.caption("Help us personalize your experience")
    
    with st.form("profile_form"):
        age = st.number_input("Age", min_value=18, max_value=100, 
                             value=int(st.session_state.user_profile.get("age", 30)))
        income = st.number_input("Annual Income ($)", min_value=0, 
                                value=int(st.session_state.user_profile.get("income", 50000)), 
                                step=5000)
        goals = st.text_area("Financial Goals", 
                            value=st.session_state.user_profile.get("goals", "Save for retirement and emergency fund"),
                            placeholder="e.g., Save for a car, buy a house, retirement")
        risk_tolerance = st.selectbox("Risk Tolerance", 
                                     ["Conservative", "Moderate", "Aggressive"],
                                     index=["conservative", "moderate", "aggressive"].index(
                                         st.session_state.user_profile.get("risk tolerance", "moderate")))
        
        if st.form_submit_button("Update Profile"):
            st.session_state.user_profile = {
                "age": str(age),
                "income": str(income),
                "goals": goals,
                "risk tolerance": risk_tolerance.lower()
            }
            st.success("✅ Profile updated!")
    
    st.divider()
    
    # Display current profile summary
    st.subheader("📋 Current Profile")
    profile = st.session_state.user_profile
    if profile:
        st.info(f"""
        **Age:** {profile.get('age', 'Not set')}  
        **Income:** ${profile.get('income', 'Not set')}/year  
        **Goals:** {profile.get('goals', 'Not set')[:50]}...  
        **Risk:** {profile.get('risk tolerance', 'Not set').title()}
        """)
    
    st.divider()
    st.header("💡 Quick Actions")
    
    # Quick action buttons
    quick_queries = [
        "What's the price of AAPL stock?",
        "Show my budget summary",
        "Add $50 for groceries",
        "Give me investment advice"
    ]
    
    for query in quick_queries:
        if st.button(query, key=f"quick_{query}"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main content area
st.title("💸 FinAdvise")
st.caption("Your personal finance assistant for stocks, expenses, budgets, and tailored advice.")

# API key check and welcome message
if not st.session_state.messages:
    if not GROQ_API_KEY:
        st.error("⚠️ **GROQ_API_KEY not configured!**")
        st.info("Please add your GROQ API key to the `.env` file to enable financial advice features.")
    
    if not ALPHA_VANTAGE_API_KEY:
        st.error("⚠️ **ALPHA_VANTAGE_API_KEY not configured!**")
        st.info("Please add your Alpha Vantage API key to the `.env` file to enable stock price features.")
    
    welcome_msg = """
    Welcome to FinAdvise! I can help you with:
    - 📈 Real-time stock prices (Try: "What's the price of AAPL stock?")
    - 💰 Expense tracking
    - 📊 Budget summaries
    - 💡 Personalized financial advice
    
    **Note:** Your profile has been pre-filled with default values. Update them in the sidebar for personalized advice!
    """
    st.info(welcome_msg)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Simple chat input (without full LangGraph for now)
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Simple responses for now
            if "stock" in user_input.lower() and "aapl" in user_input.lower():
                if ALPHA_VANTAGE_API_KEY:
                    response = "📈 **AAPL Stock Price**\n\nI'm working on fetching real-time stock data. For now, please check financial websites for current prices."
                else:
                    response = "📈 I'd love to help with stock prices, but I need the ALPHA_VANTAGE_API_KEY to be configured first."
            elif "budget" in user_input.lower():
                profile = st.session_state.user_profile
                response = f"""📊 **Budget Summary**\n\nBased on your profile (Age: {profile['age']}, Income: ${profile['income']}/year):\n\n- **Monthly Income**: ${int(profile['income'])//12:,}\n- **Recommended Savings**: 20% (${int(profile['income'])//12*0.2:,.0f})\n- **Emergency Fund Goal**: 3-6 months expenses\n\nThis is a basic calculation. I'm being enhanced to provide more detailed budget analysis!"""
            elif "advice" in user_input.lower():
                profile = st.session_state.user_profile
                response = f"""💡 **Personalized Financial Advice**\n\nFor someone your age ({profile['age']}) with {profile['risk tolerance']} risk tolerance:\n\n- Focus on building an emergency fund first\n- Consider diversified index funds for long-term growth\n- Take advantage of employer 401(k) matching\n- Review and optimize your expenses regularly\n\n*This advice is being enhanced with AI-powered personalization.*"""
            else:
                response = "🤔 I'm still learning! Right now I can help with basic stock questions (try 'AAPL stock price'), budget summaries, and general financial advice. More features coming soon!"
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})