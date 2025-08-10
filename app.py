"""Dynamic Multi-Agent Platform Dashboard"""

import streamlit as st
from datetime import datetime
import os
import glob
from pathlib import Path

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

# Agent configuration
AGENTS_CONFIG = {
    "customer-support-agent": {
        "name": "Customer Support Agent",
        "icon": "🎧",
        "port": 8502,
        "description": "Handles customer questions and can escalate to humans when needed",
        "features": ["Multi-turn conversations", "Context awareness", "Escalation handling", "User profile management"],
        "tech_stack": "LangGraph, Streamlit, Google Gemini",
        "path": "agents/customer-support-agent/src/ui/app.py"
    },
    "legal-document-review": {
        "name": "Legal Document Review",
        "icon": "⚖️",
        "port": 8501,
        "description": "Reads legal documents and answers questions about them",
        "features": ["PDF document processing", "Semantic search", "Question answering", "Document summarization"],
        "tech_stack": "LangChain, FAISS, Google Gemini",
        "path": "agents/legal-document-review/app.py"
    },
    "Finance-Advaisor-Agent": {
        "name": "Finance Advisor Agent",
        "icon": "💰",
        "port": 8503,
        "description": "Provides stock prices, tracks spending, and gives financial advice",
        "features": ["Real-time stock prices", "Personalized advice", "Expense tracking", "Budget management"],
        "tech_stack": "LangGraph, Groq LLM, Alpha Vantage",
        "path": "agents/Finance-Advaisor-Agent/app.py"
    }
}

def discover_agents():
    """Dynamically discover available agents"""
    agents_dir = Path("agents")
    available_agents = {}
    
    if agents_dir.exists():
        for agent_folder in agents_dir.iterdir():
            if agent_folder.is_dir() and agent_folder.name in AGENTS_CONFIG:
                agent_config = AGENTS_CONFIG[agent_folder.name]
                # Check if the agent has a main app file
                app_path = Path(agent_config["path"])
                if app_path.exists():
                    available_agents[agent_folder.name] = agent_config
    
    return available_agents

def main():
    """Main application entry point"""
    
    # Discover available agents
    available_agents = discover_agents()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-Agent AI Platform</h1>
        <p>Your Personal AI Assistant Workspace</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Quick Navigation")
        st.markdown("---")
        
        # System Info
        st.subheader("📊 System Status")
        st.metric("Total Agents", len(available_agents))
        st.metric("Platform Status", "🟢 Online")
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown("---")
        st.subheader("🔧 Available Agents")
        for agent_id, config in available_agents.items():
            st.info(f"{config['icon']} {config['name']} - Port {config['port']}")
        
        if len(available_agents) == 0:
            st.warning("No agents detected. Please check the agents/ directory.")
        
        st.markdown("---")
        st.success(f"Platform ready with {len(available_agents)} agents!")
    
    # Main content - Dynamic agent cards
    if available_agents:
        # Create dynamic columns based on number of agents
        num_agents = len(available_agents)
        if num_agents == 1:
            cols = [st.container()]
        elif num_agents == 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)  # Max 3 columns for better layout
        
        agent_list = list(available_agents.items())
        
        for i, (agent_id, config) in enumerate(agent_list):
            col_index = i % len(cols) if num_agents > len(cols) else i
            
            with cols[col_index]:
                # Create feature list HTML
                features_html = ""
                for feature in config["features"]:
                    features_html += f"<li>{feature}</li>"
                
                st.markdown(f"""
                <div class="agent-card">
                    <h3>{config['icon']} {config['name']}</h3>
                    <p><strong>Port:</strong> {config['port']}</p>
                    <p><strong>Status:</strong> 🟢 Running</p>
                    <p><strong>What it does:</strong> {config['description']}</p>
                    <p><strong>Features:</strong></p>
                    <ul>
                        {features_html}
                    </ul>
                    <p><strong>Tech Stack:</strong> {config['tech_stack']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"🚀 Launch {config['name']}", key=f"{agent_id}_btn"):
                    st.success(f"Opening {config['name']} at http://localhost:{config['port']}")
                    st.markdown(f"[Click here to open](http://localhost:{config['port']})")
    else:
        st.warning("""
        🔍 **No agents detected!**
        
        Please make sure you have agent directories in the `agents/` folder with their respective app files.
        
        Expected structure:
        ```
        agents/
        ├── customer-support-agent/src/ui/app.py
        ├── legal-document-review/app.py
        └── Finance-Advaisor-Agent/app.py
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #666;">
        <p><strong>Multi-Agent AI Platform</strong> | {len(available_agents)} Assistants Available</p>
        <p>Built with Streamlit & Docker | Ready for Development</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

