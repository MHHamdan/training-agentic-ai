"""Simplified Unified Landing Page for Training Agentic AI - Agent Orchestrator"""

import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Training Agentic AI - Agent Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
        transition: transform 0.2s;
    }
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Training Agentic AI Platform</h1>
        <p>Unified Orchestrator for AI Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Navigation")
        st.markdown("---")
        
        # System Info
        st.subheader("📊 System Status")
        st.metric("Total Agents", "2")
        st.metric("Platform Status", "🟢 Online")
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown("---")
        st.subheader("🔧 Development")
        st.info("All agents are running in Docker containers")
        st.success("Platform is ready for development!")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>🎧 Customer Support Agent</h3>
            <p><strong>Port:</strong> 8502</p>
            <p><strong>Status:</strong> 🟢 Running</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>Multi-turn conversations</li>
                <li>Context awareness</li>
                <li>Escalation handling</li>
                <li>User profile management</li>
            </ul>
            <p><strong>Tech Stack:</strong> LangGraph, Streamlit, Google Gemini</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Customer Support Agent", key="cs_btn"):
            st.success("Opening Customer Support Agent at http://localhost:8502")
            st.markdown("[Click here to open](http://localhost:8502)")
    
    with col2:
        st.markdown("""
        <div class="agent-card">
            <h3>⚖️ Legal Document Review</h3>
            <p><strong>Port:</strong> 8501</p>
            <p><strong>Status:</strong> 🟢 Running</p>
            <p><strong>Features:</strong></p>
            <ul>
                <li>PDF document processing</li>
                <li>Semantic search</li>
                <li>Question answering</li>
                <li>Document summarization</li>
            </ul>
            <p><strong>Tech Stack:</strong> LangChain, FAISS, Google Gemini</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Legal Document Review", key="ldr_btn"):
            st.success("Opening Legal Document Review at http://localhost:8501")
            st.markdown("[Click here to open](http://localhost:8501)")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Training Agentic AI Platform</strong> | Built with Streamlit & Docker</p>
        <p>All agents are containerized and ready for development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

