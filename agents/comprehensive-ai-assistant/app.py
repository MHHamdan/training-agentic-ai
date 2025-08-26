"""
Comprehensive AI Assistant Agent - Main Application
Your All-in-One Information Hub with Visual Workflow Observability
Author: Mohammed Hamdan
"""

import streamlit as st
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import logging

# Import our components
from workflows.ai_assistant_graph import ComprehensiveAIWorkflow
from services.multi_api_service import MultiAPIService
from config.settings import API_CONFIGS, SERVICE_CATEGORIES, DEFAULT_USER_PREFERENCES
from utils.response_formatter import ResponseFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Comprehensive AI Assistant - Your All-in-One Information Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .workflow-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border: 1px solid #e1e5e9;
    }
    
    .service-status {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .service-status.warning {
        border-left-color: #ffc107;
        background: #fff8e1;
    }
    
    .service-status.error {
        border-left-color: #dc3545;
        background: #ffebee;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-around;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .workflow-step {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .workflow-step.completed {
        border-left-color: #28a745;
        background: #f8fff8;
    }
    
    .workflow-step.running {
        border-left-color: #ffc107;
        background: #fff8f0;
        animation: pulse 2s infinite;
    }
    
    .workflow-step.failed {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .response-container {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
    }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .confidence-high {
        background: #d4edda;
        color: #155724;
    }
    
    .confidence-medium {
        background: #fff3cd;
        color: #856404;
    }
    
    .confidence-low {
        background: #f8d7da;
        color: #721c24;
    }
    
    .chat-input {
        position: sticky;
        bottom: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e1e5e9;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "workflow_manager" not in st.session_state:
        st.session_state.workflow_manager = ComprehensiveAIWorkflow()
    
    if "multi_api_service" not in st.session_state:
        st.session_state.multi_api_service = MultiAPIService()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_workflow_result" not in st.session_state:
        st.session_state.current_workflow_result = None
    
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = DEFAULT_USER_PREFERENCES.copy()
    
    if "processing_in_progress" not in st.session_state:
        st.session_state.processing_in_progress = False

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Comprehensive AI Assistant</h1>
        <p>Your All-in-One Information Hub - News, Weather, Finance, Entertainment & More</p>
    </div>
    """, unsafe_allow_html=True)

def display_service_status():
    """Display service status in sidebar"""
    st.sidebar.markdown("## 🔧 Service Status")
    
    # Get service status
    service_status = st.session_state.multi_api_service.get_service_status()
    
    # Count configured services
    configured_count = sum(1 for status in service_status.values() if status["configured"])
    total_services = len(service_status)
    
    # Display overall status
    if configured_count > 0:
        st.sidebar.success(f"✅ {configured_count}/{total_services} services configured")
    else:
        st.sidebar.warning(f"⚠️ {configured_count}/{total_services} services configured")
    
    # Service categories
    with st.sidebar.expander("📊 Service Categories", expanded=True):
        for category, services in SERVICE_CATEGORIES.items():
            configured_in_category = sum(
                1 for service in services 
                if service in service_status and service_status[service]["configured"]
            )
            
            if configured_in_category > 0:
                st.write(f"✅ **{category.title()}**: {configured_in_category}/{len(services)}")
            else:
                st.write(f"⚠️ **{category.title()}**: {configured_in_category}/{len(services)}")
    
    # API Configuration Guide
    with st.sidebar.expander("🔑 API Setup Guide", expanded=False):
        st.markdown("""
        **Quick Setup:**
        1. Get free API keys from providers
        2. Add keys to `.env` file
        3. Restart the application
        
        **Free APIs Available:**
        - DuckDuckGo (No key required)
        - CoinGecko (No key required)  
        - NewsAPI (1000 calls/day)
        - OpenWeatherMap (1000 calls/day)
        - TMDB (Free tier available)
        """)
        
        if st.button("📋 Copy .env Template"):
            st.code("""
# Add these to your .env file:
NEWSAPI_KEY=your-newsapi-key
OPENWEATHER_API_KEY=your-weather-key
TMDB_API_KEY=your-tmdb-key
YELP_API_KEY=your-yelp-key
# More keys in .env.example
            """)

def display_workflow_visualization(workflow_result: Dict[str, Any]):
    """Display workflow visualization"""
    if not workflow_result or "workflow_steps" not in workflow_result:
        return
    
    st.markdown("### 🔄 Workflow Execution")
    
    workflow_steps = workflow_result["workflow_steps"]
    
    # Create workflow progress visualization
    if workflow_steps:
        # Workflow timeline chart
        fig = go.Figure()
        
        step_names = []
        start_times = []
        durations = []
        statuses = []
        
        base_time = 0
        for step in workflow_steps:
            step_names.append(step["name"])
            start_times.append(base_time)
            duration = step.get("duration", 0.1)
            durations.append(duration)
            statuses.append(step["status"])
            base_time += duration
        
        # Color mapping for statuses
        color_map = {
            "completed": "#28a745",
            "running": "#ffc107", 
            "failed": "#dc3545",
            "pending": "#6c757d"
        }
        
        colors = [color_map.get(status, "#6c757d") for status in statuses]
        
        fig.add_trace(go.Bar(
            y=step_names,
            x=durations,
            orientation='h',
            marker_color=colors,
            text=[f"{d:.2f}s" for d in durations],
            textposition="inside"
        ))
        
        fig.update_layout(
            title="Workflow Step Execution Timeline",
            xaxis_title="Duration (seconds)",
            yaxis_title="Workflow Steps",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Workflow steps details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Workflow Steps")
        for step in workflow_steps:
            status = step["status"]
            status_emoji = {
                "completed": "✅",
                "running": "🔄",
                "failed": "❌",
                "pending": "⏳"
            }.get(status, "⚫")
            
            status_class = {
                "completed": "completed",
                "running": "running", 
                "failed": "failed",
                "pending": "pending"
            }.get(status, "")
            
            st.markdown(f"""
            <div class="workflow-step {status_class}">
                <strong>{status_emoji} {step['name']}</strong><br>
                <small>Duration: {step.get('duration', 0):.2f}s</small>
                {f"<br><small style='color: red;'>Error: {step['error']}</small>" if step.get('error') else ""}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Metrics")
        
        # Calculate metrics
        total_steps = len(workflow_steps)
        completed_steps = sum(1 for step in workflow_steps if step["status"] == "completed")
        failed_steps = sum(1 for step in workflow_steps if step["status"] == "failed")
        total_time = sum(step.get("duration", 0) for step in workflow_steps)
        
        st.metric("Total Steps", total_steps)
        st.metric("Completed", completed_steps)
        st.metric("Failed", failed_steps)
        st.metric("Total Time", f"{total_time:.2f}s")
        
        # Success rate
        success_rate = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")

def display_api_performance_metrics(workflow_result: Dict[str, Any]):
    """Display API performance metrics"""
    if not workflow_result or "metadata" not in workflow_result:
        return
    
    metadata = workflow_result["metadata"]
    
    st.markdown("### 📊 API Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "APIs Called", 
            metadata.get("total_apis", 0),
            help="Total number of API services called"
        )
    
    with col2:
        st.metric(
            "Successful", 
            metadata.get("successful_apis", 0),
            help="Number of APIs that returned data successfully"
        )
    
    with col3:
        success_rate = metadata.get("success_rate", 0) * 100
        st.metric(
            "Success Rate", 
            f"{success_rate:.1f}%",
            help="Percentage of APIs that succeeded"
        )
    
    with col4:
        processing_time = workflow_result.get("processing_time", 0)
        st.metric(
            "Processing Time", 
            f"{processing_time:.2f}s",
            help="Total time to process the request"
        )

def display_confidence_indicator(confidence: float):
    """Display confidence indicator"""
    if confidence >= 0.8:
        badge_class = "confidence-high"
        emoji = "🎯"
    elif confidence >= 0.5:
        badge_class = "confidence-medium"
        emoji = "⚠️"
    else:
        badge_class = "confidence-low"
        emoji = "❓"
    
    st.markdown(f"""
    <span class="confidence-badge {badge_class}">
        {emoji} Confidence: {confidence:.1%}
    </span>
    """, unsafe_allow_html=True)

def process_user_query():
    """Process user query through the AI workflow"""
    user_query = st.session_state.get("user_input", "").strip()
    
    if not user_query:
        st.warning("Please enter a question or request")
        return
    
    # Clear the input
    st.session_state.user_input = ""
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now()
    })
    
    # Process the query
    async def run_workflow():
        st.session_state.processing_in_progress = True
        
        try:
            result = await st.session_state.workflow_manager.process_request(
                user_query=user_query,
                user_location=st.session_state.user_preferences.get("location"),
                user_preferences=st.session_state.user_preferences
            )
            
            st.session_state.current_workflow_result = result
            
            # Add AI response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.get("response", "I apologize, but I couldn't process your request."),
                "confidence": result.get("confidence", 0.0),
                "metadata": result.get("metadata", {}),
                "workflow_result": result,
                "timestamp": datetime.now()
            })
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {e}")
        
        finally:
            st.session_state.processing_in_progress = False
    
    # Run the async workflow
    asyncio.run(run_workflow())

def display_chat_interface():
    """Display chat interface"""
    st.markdown("## 💬 Chat with AI Assistant")
    
    # Chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history[-10:]:  # Show last 10 messages
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                    <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #2196f3;">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="response-container">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <strong>🤖 AI Assistant:</strong>
                        </div>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show confidence and metadata for AI responses
                    if message.get("confidence") is not None:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            display_confidence_indicator(message["confidence"])
                        with col2:
                            timestamp = message["timestamp"].strftime("%I:%M %p")
                            st.caption(f"Response generated at {timestamp}")
                    
                    # Show workflow visualization for the latest response
                    if message.get("workflow_result") and message == st.session_state.chat_history[-1]:
                        with st.expander("🔍 View Workflow Details", expanded=False):
                            display_workflow_visualization(message["workflow_result"])
                            display_api_performance_metrics(message["workflow_result"])
    
    else:
        st.info("👋 Welcome! Ask me anything about news, weather, finance, entertainment, and more!")
        
        # Show example queries
        st.markdown("**Try asking:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌤️ What's the weather like?"):
                st.session_state.user_input = "What's the weather like?"
                process_user_query()
                st.rerun()
            
            if st.button("📰 Show me latest tech news"):
                st.session_state.user_input = "Show me latest tech news"
                process_user_query()
                st.rerun()
        
        with col2:
            if st.button("💰 What's the AAPL stock price?"):
                st.session_state.user_input = "What's the AAPL stock price?"
                process_user_query()
                st.rerun()
            
            if st.button("🍽️ Find restaurants near me"):
                st.session_state.user_input = "Find restaurants near me"
                process_user_query()
                st.rerun()

def display_user_preferences():
    """Display user preferences in sidebar"""
    st.sidebar.markdown("## ⚙️ Preferences")
    
    with st.sidebar.expander("🌍 Location & Interests", expanded=False):
        # Location
        location = st.text_input(
            "📍 Your Location",
            value=st.session_state.user_preferences.get("location", ""),
            help="City, State or City, Country"
        )
        if location != st.session_state.user_preferences.get("location", ""):
            st.session_state.user_preferences["location"] = location
        
        # Interests
        interests = st.multiselect(
            "📚 Interests",
            options=["technology", "health", "finance", "sports", "entertainment", "politics", "science"],
            default=st.session_state.user_preferences.get("interests", [])
        )
        st.session_state.user_preferences["interests"] = interests
        
        # Weather units
        units = st.selectbox(
            "🌡️ Temperature Units",
            options=["imperial", "metric"],
            index=0 if st.session_state.user_preferences.get("weather_units") == "imperial" else 1
        )
        st.session_state.user_preferences["weather_units"] = units

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    display_service_status()
    display_user_preferences()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        display_chat_interface()
        
        # Chat input (sticky at bottom)
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        
        # Input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Ask me anything...",
                placeholder="What's the weather? Show me news. Find restaurants. Check stocks...",
                key="user_input_form",
                disabled=st.session_state.processing_in_progress
            )
            
            col_submit1, col_submit2 = st.columns([1, 4])
            with col_submit1:
                submit_button = st.form_submit_button(
                    "🚀 Send",
                    disabled=st.session_state.processing_in_progress
                )
            
            if submit_button and user_input.strip():
                st.session_state.user_input = user_input
                process_user_query()
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Real-time workflow status
        if st.session_state.processing_in_progress:
            st.markdown("### 🔄 Processing...")
            with st.spinner("Gathering information from multiple sources..."):
                time.sleep(0.1)  # Small delay for UI responsiveness
        
        # Latest workflow result
        if st.session_state.current_workflow_result:
            st.markdown("### 📈 Latest Request Metrics")
            display_api_performance_metrics(st.session_state.current_workflow_result)
        
        # Service performance over time
        if len(st.session_state.chat_history) > 1:
            st.markdown("### 📊 Performance History")
            
            # Extract performance data from chat history
            response_times = []
            confidences = []
            timestamps = []
            
            for msg in st.session_state.chat_history:
                if msg.get("role") == "assistant" and msg.get("workflow_result"):
                    result = msg["workflow_result"]
                    response_times.append(result.get("processing_time", 0))
                    confidences.append(result.get("confidence", 0))
                    timestamps.append(msg["timestamp"])
            
            if response_times:
                # Performance chart
                df = pd.DataFrame({
                    "Response Time": response_times,
                    "Confidence": [c * 100 for c in confidences],
                    "Request": range(1, len(response_times) + 1)
                })
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=df["Request"], y=df["Response Time"], name="Response Time (s)"),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Scatter(x=df["Request"], y=df["Confidence"], name="Confidence (%)"),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Request Number")
                fig.update_yaxes(title_text="Response Time (seconds)", secondary_y=False)
                fig.update_yaxes(title_text="Confidence (%)", secondary_y=True)
                
                fig.update_layout(height=300, showlegend=True)
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>Comprehensive AI Assistant</strong> - Your All-in-One Information Hub<br>
        Powered by LangGraph, Multi-API Integration & Real-time Observability<br>
        <em>Author: Mohammed Hamdan</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()