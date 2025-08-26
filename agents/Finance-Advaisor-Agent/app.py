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
# Use OpenAI instead of expired Groq API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        llm = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        llm = None
        logger.error("OpenAI package not installed")
else:
    llm = None

# === STOCK DATA FUNCTIONS ===
# Company name to symbol mapping
company_to_symbol = {
    'apple': 'AAPL', 'aapl': 'AAPL',
    'google': 'GOOGL', 'alphabet': 'GOOGL', 'googl': 'GOOGL',
    'microsoft': 'MSFT', 'msft': 'MSFT',
    'amazon': 'AMZN', 'amzn': 'AMZN',
    'tesla': 'TSLA', 'tsla': 'TSLA',
    'meta': 'META', 'facebook': 'META', 'meta': 'META',
    'nvidia': 'NVDA', 'nvda': 'NVDA',
    'netflix': 'NFLX', 'nflx': 'NFLX',
    'walmart': 'WMT', 'wmt': 'WMT',
    'coca-cola': 'KO', 'cocacola': 'KO', 'ko': 'KO'
}

def extract_stock_symbol(text: str) -> str:
    """Extract stock symbol from user input"""
    text_lower = text.lower()
    
    # Check for direct symbol matches
    for company, symbol in company_to_symbol.items():
        if company in text_lower:
            return symbol
    
    # Look for potential ticker symbols (3-5 uppercase letters)
    import re
    ticker_pattern = r'\b[A-Z]{3,5}\b'
    matches = re.findall(ticker_pattern, text.upper())
    if matches:
        return matches[0]
    
    return None

def get_stock_price(symbol: str) -> str:
    """Fetch current stock price from Alpha Vantage API"""
    if not ALPHA_VANTAGE_API_KEY:
        return "❌ Alpha Vantage API key not configured."
    
    try:
        url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        logger.info(f"Fetching stock data for {symbol}")
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            return f"❌ Error: {data['Error Message']}"
        
        if "Note" in data:
            return f"⚠️ API Limit: {data['Note']}"
        
        if "Global Quote" not in data or not data["Global Quote"]:
            return f"❌ No data found for symbol '{symbol}'. Please verify the ticker symbol."
        
        quote = data["Global Quote"]
        current_price = float(quote["05. price"])
        change = float(quote["09. change"])
        change_percent = quote["10. change percent"].replace('%', '')
        
        # Format the response
        change_emoji = "📈" if change >= 0 else "📉"
        change_color = "🟢" if change >= 0 else "🔴"
        
        response = f"""📊 **{symbol} Stock Price**

💰 **Current Price**: ${current_price:.2f}
{change_emoji} **Change**: {change_color} ${change:+.2f} ({change_percent}%)

*Data provided by Alpha Vantage*"""
        
        return response
        
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return f"❌ Network error: Unable to fetch stock data."
    except KeyError as e:
        logger.error(f"Data parsing error: {e}")
        return f"❌ Error parsing stock data. API response format may have changed."
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"❌ Unexpected error: {str(e)}"

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
            # Handle stock price requests with real API calls
            if "stock" in user_input.lower() or "price" in user_input.lower():
                if ALPHA_VANTAGE_API_KEY:
                    # Extract stock symbol from query
                    symbol = extract_stock_symbol(user_input)
                    if symbol:
                        response = get_stock_price(symbol)
                    else:
                        response = "📈 I couldn't identify a stock symbol in your query. Please specify a company name or ticker (e.g., AAPL, Tesla, Microsoft)."
                else:
                    response = "📈 I'd love to help with stock prices, but I need the ALPHA_VANTAGE_API_KEY to be configured first."
            elif "budget" in user_input.lower():
                profile = st.session_state.user_profile
                response = f"""📊 **Budget Summary**\n\nBased on your profile (Age: {profile['age']}, Income: ${profile['income']}/year):\n\n- **Monthly Income**: ${int(profile['income'])//12:,}\n- **Recommended Savings**: 20% (${int(profile['income'])//12*0.2:,.0f})\n- **Emergency Fund Goal**: 3-6 months expenses\n\nThis is a basic calculation. I'm being enhanced to provide more detailed budget analysis!"""
            elif "advice" in user_input.lower():
                profile = st.session_state.user_profile
                response = f"""💡 **Personalized Financial Advice**\n\nFor someone your age ({profile['age']}) with {profile['risk tolerance']} risk tolerance:\n\n- Focus on building an emergency fund first\n- Consider diversified index funds for long-term growth\n- Take advantage of employer 401(k) matching\n- Review and optimize your expenses regularly\n\n*This advice is being enhanced with AI-powered personalization.*"""
            else:
                # Use OpenAI LLM for general financial questions if available
                if llm and OPENAI_API_KEY:
                    try:
                        profile = st.session_state.user_profile
                        context = f"User profile: Age {profile['age']}, Annual income ${profile['income']}, Goals: {profile['goals']}, Risk tolerance: {profile['risk tolerance']}"
                        
                        messages = [
                            {"role": "system", "content": f"You are FinAdvise, a helpful personal finance assistant. Based on the user's profile, provide helpful financial advice. {context}"},
                            {"role": "user", "content": user_input}
                        ]
                        
                        llm_response = llm.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            max_tokens=300,
                            temperature=0.7
                        )
                        response = f"💡 {llm_response.choices[0].message.content}"
                    except Exception as e:
                        logger.error(f"LLM error: {e}")
                        response = "🤔 I'm still learning! Right now I can help with stock price questions (try 'AAPL stock price'), budget summaries, and general financial advice. More features coming soon!"
                else:
                    response = "🤔 I'm still learning! Right now I can help with stock price questions (try 'AAPL stock price'), budget summaries, and general financial advice. More features coming soon!"
            
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})