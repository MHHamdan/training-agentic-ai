"""Unified Landing Page for Training Agentic AI - Agent Orchestrator"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional
import os

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
    .status-online {
        color: #10b981;
        font-weight: bold;
    }
    .status-offline {
        color: #ef4444;
        font-weight: bold;
    }
    .status-unknown {
        color: #f59e0b;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
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
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
</style>
""", unsafe_allow_html=True)

class AgentOrchestrator:
    """Main orchestrator class for managing all agents"""
    
    def __init__(self):
        self.agents = {
            "customer-support-agent": {
                "name": "Customer Support Agent",
                "description": "AI-powered customer support system with LangGraph workflow",
                "port": 8502,
                "icon": "🎧",
                "status": "unknown",
                "features": [
                    "Multi-turn conversations",
                    "Context awareness",
                    "Escalation handling",
                    "User profile management"
                ],
                "tech_stack": ["LangGraph", "Streamlit", "Google Gemini", "SQLite"],
                "url": "http://localhost:8502"
            },
            "legal-document-review": {
                "name": "Legal Document Review",
                "description": "AI-powered legal document analysis with RAG capabilities",
                "port": 8501,
                "icon": "⚖️",
                "status": "unknown",
                "features": [
                    "PDF document processing",
                    "Semantic search",
                    "Question answering",
                    "Document summarization"
                ],
                "tech_stack": ["LangChain", "FAISS", "Google Gemini", "PyPDF2"],
                "url": "http://localhost:8501"
            }
        }
        self.system_metrics = {
            "total_agents": len(self.agents),
            "online_agents": 0,
            "total_requests": 0,
            "uptime": datetime.now().isoformat()
        }
    
    def check_agent_status(self, agent_name: str) -> str:
        """Check if an agent is online"""
        agent = self.agents.get(agent_name)
        if not agent:
            return "unknown"
        
        try:
            response = requests.get(f"{agent['url']}/_stcore/health", timeout=3)
            if response.status_code == 200:
                return "online"
            else:
                return "offline"
        except:
            return "offline"
    
    def update_all_statuses(self):
        """Update status of all agents"""
        online_count = 0
        for agent_name in self.agents:
            status = self.check_agent_status(agent_name)
            self.agents[agent_name]["status"] = status
            if status == "online":
                online_count += 1
        
        self.system_metrics["online_agents"] = online_count
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Training Agentic AI - Agent Orchestrator</h1>
            <p>Unified platform for managing and orchestrating AI agents</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with system information"""
        with st.sidebar:
            st.title("🎛️ System Control")
            
            # System metrics
            st.subheader("📊 System Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Agents", self.system_metrics["total_agents"])
            with col2:
                st.metric("Online Agents", self.system_metrics["online_agents"])
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔄 Refresh Status"):
                self.update_all_statuses()
                st.rerun()
            
            if st.button("🚀 Start All Agents"):
                st.info("Use 'docker compose up -d' to start all agents")
            
            if st.button("🛑 Stop All Agents"):
                st.info("Use 'docker compose down' to stop all agents")
            
            st.markdown("---")
            
            # System info
            st.subheader("ℹ️ System Info")
            st.write(f"**Uptime:** {self.system_metrics['uptime'][:19]}")
            st.write(f"**Environment:** {os.getenv('ENVIRONMENT', 'development')}")
            
            # Docker commands
            st.markdown("---")
            st.subheader("🐳 Docker Commands")
            st.code("docker compose up -d", language="bash")
            st.code("docker compose down", language="bash")
            st.code("docker compose logs", language="bash")
    
    def render_agent_card(self, agent_name: str, agent_info: Dict):
        """Render individual agent card"""
        status = agent_info["status"]
        status_color = {
            "online": "status-online",
            "offline": "status-offline",
            "unknown": "status-unknown"
        }.get(status, "status-unknown")
        
        with st.container():
            st.markdown(f"""
            <div class="agent-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2>{agent_info['icon']} {agent_info['name']}</h2>
                        <p style="color: #666; margin-bottom: 1rem;">{agent_info['description']}</p>
                    </div>
                    <div>
                        <span class="{status_color}">● {status.upper()}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Agent details in expander
            with st.expander(f"📋 {agent_info['name']} Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚀 Features")
                    for feature in agent_info["features"]:
                        st.write(f"• {feature}")
                
                with col2:
                    st.subheader("🛠️ Tech Stack")
                    for tech in agent_info["tech_stack"]:
                        st.write(f"• {tech}")
                
                # Action buttons
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"🌐 Open {agent_info['name']}", key=f"open_{agent_name}"):
                        st.markdown(f"[Open {agent_info['name']}]({agent_info['url']})")
                
                with col2:
                    if st.button(f"📊 Monitor {agent_info['name']}", key=f"monitor_{agent_name}"):
                        st.info(f"Monitoring {agent_info['name']} at {agent_info['url']}")
                
                with col3:
                    if st.button(f"🔄 Restart {agent_info['name']}", key=f"restart_{agent_name}"):
                        st.info(f"Use 'docker compose restart {agent_name}' to restart")
    
    def render_agents_grid(self):
        """Render all agents in a grid"""
        st.subheader("🤖 Available Agents")
        
        # Update statuses
        self.update_all_statuses()
        
        # Render each agent
        for agent_name, agent_info in self.agents.items():
            self.render_agent_card(agent_name, agent_info)
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Health",
                f"{self.system_metrics['online_agents']}/{self.system_metrics['total_agents']}",
                f"{'🟢' if self.system_metrics['online_agents'] == self.system_metrics['total_agents'] else '🟡' if self.system_metrics['online_agents'] > 0 else '🔴'}"
            )
        
        with col2:
            st.metric("Total Requests", self.system_metrics["total_requests"])
        
        with col3:
            uptime_duration = datetime.now() - datetime.fromisoformat(self.system_metrics["uptime"])
            st.metric("Uptime", f"{uptime_duration.seconds // 3600}h {(uptime_duration.seconds % 3600) // 60}m")
        
        with col4:
            st.metric("Environment", os.getenv("ENVIRONMENT", "development").title())
    
    def render_development_info(self):
        """Render development information"""
        st.subheader("🛠️ Development Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📁 Project Structure:**
            ```
            training-agentic-ai/
            ├── agents/
            │   ├── customer-support-agent/
            │   └── legal-document-review/
            ├── tests/
            ├── requirements.txt
            ├── docker-compose.yml
            └── app.py (this orchestrator)
            ```
            """)
        
        with col2:
            st.markdown("""
            **🚀 Getting Started:**
            1. Set up environment variables
            2. Install dependencies: `pip install -r requirements.txt`
            3. Start all agents: `docker compose up -d`
            4. Access this orchestrator at `http://localhost:8500`
            """)
    
    def run(self):
        """Run the orchestrator application"""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🤖 Agents", "📊 Analytics", "🛠️ Development"])
        
        with tab1:
            self.render_agents_grid()
        
        with tab2:
            self.render_analytics()
        
        with tab3:
            self.render_development_info()


def main():
    """Main application entry point"""
    orchestrator = AgentOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
