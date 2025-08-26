"""Dynamic Multi-Agent Platform Dashboard"""

import streamlit as st
from datetime import datetime
import os
import glob
from pathlib import Path
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Page configuration
st.set_page_config(
    page_title="Training Agentic AI - Agent Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI with enhanced visuals
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.3; }
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 1.2rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .agent-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .agent-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    .agent-card h3 {
        color: #2d3436;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    .agent-card .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.5rem 0;
    }
    
    .status-online {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #d63031 0%, #e17055 100%);
        color: white;
    }
    
    .status-checking {
        background: linear-gradient(135deg, #fdcb6e 0%, #f39c12 100%);
        color: white;
        animation: pulse-badge 1.5s infinite;
    }
    
    @keyframes pulse-badge {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .feature-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: 500;
    }
    
    .tech-stack {
        background: linear-gradient(135deg, #f5f6fa 0%, #e8ecf3 100%);
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #636e72;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6b4190 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stats-container {
        background: linear-gradient(135deg, #f5f6fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #636e72;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .connectivity-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: blink 2s infinite;
    }
    
    .indicator-online {
        background: #00b894;
        box-shadow: 0 0 10px rgba(0, 184, 148, 0.5);
    }
    
    .indicator-offline {
        background: #d63031;
        box-shadow: 0 0 10px rgba(214, 48, 49, 0.5);
    }
    
    .indicator-checking {
        background: #fdcb6e;
        box-shadow: 0 0 10px rgba(253, 203, 110, 0.5);
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
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
    },
    "handwriting-document-agent": {
        "name": "Handwriting Document Agent",
        "icon": "📜",
        "port": 8516,
        "description": "Professional AI-powered OCR for handwritten documents and historical texts with multi-language support",
        "features": ["API-based OCR (OpenAI/Claude/Gemini)", "Multi-language support", "Historical document analysis", "Interactive chat", "Professional accuracy (95%+)"],
        "tech_stack": "LangGraph, Multi-provider APIs, ChromaDB, Streamlit",
        "path": "agents/handwriting-document-agent/app.py"
    },
    "competitive-intel-agent": {
        "name": "Competitive Intel Agent",
        "icon": "🔍",
        "port": 8504,
        "description": "AI-powered competitive analysis with advanced reasoning and insights",
        "features": ["Competitor analysis", "Market insights", "Strategic recommendations", "ReAct reasoning"],
        "tech_stack": "Cohere, LlamaIndex, Agentic RAG",
        "path": "agents/competitive-intel-agent/app.py"
    },
    "comprehensive-ai-assistant": {
        "name": "Comprehensive AI Assistant",
        "icon": "🤖",
        "port": 8517,
        "description": "Your All-in-One Information Hub with real-time news, weather, finance, entertainment, and more with visual workflow observability",
        "features": ["Multi-API Integration", "LangGraph Workflows", "Real-time Data", "Visual Observability", "12+ Service Categories", "Intelligent Aggregation"],
        "tech_stack": "LangGraph, LangSmith, Multi-API, Plotly, Streamlit",
        "path": "agents/comprehensive-ai-assistant/app.py"
    },
    "insights-explorer-agent": {
        "name": "Insights Explorer Agent",
        "icon": "📊",
        "port": 8505,
        "description": "AI-powered data analysis with semantic memory and intelligent insights",
        "features": ["Dataset analysis", "Statistical summaries", "Semantic search", "Data embeddings"],
        "tech_stack": "Phidata, Google Gemini, Pinecone, SentenceTransformers",
        "path": "agents/insights-explorer-agent/app.py"
    },
    "Customer-Support-Triage": {
        "name": "Customer Support Triage Agent",
        "icon": "🎫",
        "port": 8506,
        "description": "AI-powered customer support triage for e-commerce with sentiment analysis and automated routing",
        "features": ["Sentiment analysis", "Intent classification", "Response generation", "Ticket routing", "Management insights"],
        "tech_stack": "Agno/Phidata, Google Gemini, Pinecone, OpenAI Embeddings",
        "path": "agents/Customer-Support-Triage/app.py"
    },
    "stock-analysis-extended": {
        "name": "Multi-Agent Stock Analysis System",
        "icon": "🚀",
        "port": 8507,
        "description": "Advanced Multi-Agent Stock Analysis Platform with AI-powered agents for comprehensive risk assessment, sentiment analysis, and technical indicators",
        "features": ["Technical Analysis", "Risk Assessment", "Sentiment Analysis", "Multi-Agent Orchestration", "Real-time Market Data"],
        "tech_stack": "CrewAI, Yahoo Finance, Alpha Vantage, Finnhub, NewsAPI",
        "path": "agents/stock-analysis-extended/app.py"
    },
    "multi-agent-financial-analysis": {
        "name": "Multi-Agent Financial Analysis System",
        "icon": "💹",
        "port": 8508,
        "description": "Advanced LangGraph-powered multi-agent financial analysis platform with sophisticated workflow orchestration, human-in-the-loop approvals, and real-time market alerts",
        "features": ["LangGraph Workflows", "7 Specialized Agents", "Human-in-Loop", "Real-time Alerts", "Portfolio Optimization", "Compliance Monitoring", "Advanced Routing"],
        "tech_stack": "LangGraph, LangChain, Grok (xAI), Yahoo Finance, Alpha Vantage",
        "path": "agents/multi-agent-financial-analysis/app.py"
    },
    "ai-content-creation-system": {
        "name": "AI Content Creation System",
        "icon": "✍️",
        "port": 8509,
        "description": "Advanced LangGraph-powered multi-agent content creation platform with 7 specialized content agents, SEO optimization, quality assurance, and comprehensive workflow orchestration",
        "features": ["7 Specialized Agents", "SEO Optimization", "Quality Assurance", "Brand Compliance", "Multi-format Content", "LangGraph Workflows", "Human-in-Loop"],
        "tech_stack": "LangGraph, LangChain, Grok (xAI), OpenAI, Google Gemini, Anthropic",
        "path": "agents/ai-content-creation-system/app.py"
    },
    "autogen-research-intelligence-agent": {
        "name": "ARIA - Autogen Research Intelligence Agent",
        "icon": "🔬",
        "port": 8510,
        "description": "Advanced AI-powered research assistant built with Microsoft Autogen framework. Features multi-agent conversations, comprehensive research workflows, and human-in-the-loop control.",
        "features": ["Microsoft Autogen", "Multi-Agent Research", "Human-in-Loop", "Export Tools", "Conversation Manager", "Research Tools", "Subtopic Generation"],
        "tech_stack": "Microsoft Autogen, Google Gemini, OpenAI, Anthropic, Streamlit",
        "path": "agents/autogen-research-intelligence-agent/app.py"
    },
    "medical-research-intelligence-agent": {
        "name": "MARIA - Medical Research Intelligence Agent",
        "icon": "🏥",
        "port": 8511,
        "description": "Comprehensive AI-powered healthcare research assistant built with AutoGen framework. Specialized in medical literature analysis, treatment comparison, and clinical research with human-in-the-loop medical approval system.",
        "features": ["Medical Literature Analysis", "Treatment Efficacy Comparison", "Clinical Trial Research", "Drug Interaction Checking", "HITL Medical Approval", "PRISMA-Style Reports", "Healthcare AutoGen Agents"],
        "tech_stack": "Microsoft Autogen, Google Gemini, PubMed API, ClinicalTrials.gov, Medical NLP",
        "path": "agents/medical-research-intelligence-agent/app.py"
    },
    "resume-screening": {
        "name": "Resume Screening Agent",
        "icon": "📄",
        "port": 8512,
        "description": "Production-ready AI resume screening agent with full observability, multi-model support, and comprehensive analysis capabilities. Leverages 15+ Hugging Face models for unbiased, thorough resume evaluation.",
        "features": ["15+ AI Models", "Multi-Model Comparison", "5-Dimensional Scoring", "LangSmith Observability", "Vector Storage", "OCR Support", "Batch Processing", "Export Options"],
        "tech_stack": "Hugging Face, LangSmith, ChromaDB, PyTesseract, Streamlit, PDF2Image",
        "path": "agents/resume-screening/run.py"
    },
    "stock-analysis": {
        "name": "Stock Analysis Agent",
        "icon": "📈",
        "port": 8513,
        "description": "Enterprise-grade AI-powered stock analysis with multi-agent orchestration, comprehensive risk assessment, and regulatory compliance. Features 5 specialized agents for fundamental, technical, sentiment, risk, and report generation.",
        "features": ["Multi-Agent CrewAI", "AgentOps Observability", "SEC/FINRA Compliance", "Risk Assessment", "Technical Analysis", "Sentiment Analysis", "Professional Reports"],
        "tech_stack": "CrewAI, AgentOps, Hugging Face, yfinance, Alpha Vantage, Streamlit",
        "path": "agents/stock-analysis/app.py"
    },
    "research-agent": {
        "name": "Research Agent V2",
        "icon": "🔬",
        "port": 8514,
        "description": "Production-ready AI research agent with full Langfuse observability, LangGraph orchestration, and multi-model support. Enterprise-grade research with academic compliance, fact-checking, and citation management.",
        "features": ["Langfuse Observability", "LangGraph Multi-Agent", "Hugging Face Models", "Academic Citations", "Fact Checking", "Real-time Monitoring", "Model Comparison", "Research Quality Evaluation"],
        "tech_stack": "LangGraph, Langfuse, Hugging Face, OpenAI, Anthropic, DuckDuckGo, ArXiv, Streamlit",
        "path": "agents/research-agent/app.py"
    }
}

def check_agent_status(agent_name, port, timeout=2):
    """Check if an agent is running on its designated port"""
    try:
        response = requests.get(f"http://localhost:{port}", timeout=timeout)
        return "online" if response.status_code == 200 else "offline"
    except requests.exceptions.ConnectionError:
        return "offline"
    except requests.exceptions.Timeout:
        return "timeout"
    except Exception:
        return "error"

def check_all_agents_parallel(agents_config):
    """Check the status of all agents in parallel"""
    agent_statuses = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_agent = {
            executor.submit(check_agent_status, name, config["port"]): (name, config)
            for name, config in agents_config.items()
        }
        
        for future in as_completed(future_to_agent):
            agent_name, config = future_to_agent[future]
            try:
                status = future.result()
                agent_statuses[agent_name] = status
            except Exception:
                agent_statuses[agent_name] = "error"
    
    return agent_statuses

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
    
    # Initialize session state for connectivity checks
    if 'last_check' not in st.session_state:
        st.session_state.last_check = time.time()
    if 'agent_statuses' not in st.session_state:
        st.session_state.agent_statuses = {}
    
    # Discover available agents
    available_agents = discover_agents()
    
    # Check agent connectivity (refresh every 30 seconds or on button click)
    current_time = time.time()
    if current_time - st.session_state.last_check > 30:
        st.session_state.agent_statuses = check_all_agents_parallel(available_agents)
        st.session_state.last_check = current_time
    
    # Header with animated gradient
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Multi-Agent AI Platform</h1>
        <p>Orchestrating Intelligent Agents for Your Workflow</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Agents</div>
        </div>
        """.format(len(available_agents)), unsafe_allow_html=True)
    
    with col2:
        online_count = sum(1 for status in st.session_state.agent_statuses.values() if status == "online")
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Agents Online</div>
        </div>
        """.format(online_count), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">✅</div>
            <div class="metric-label">Platform Status</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Current Time</div>
        </div>
        """.format(datetime.now().strftime("%H:%M")), unsafe_allow_html=True)
    
    # Refresh connectivity button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Refresh Agent Connectivity", use_container_width=True):
            with st.spinner("Checking agent connectivity..."):
                st.session_state.agent_statuses = check_all_agents_parallel(available_agents)
                st.session_state.last_check = time.time()
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sidebar with live status
    with st.sidebar:
        st.markdown("### 🎯 Agent Control Center")
        st.markdown("---")
        
        # Quick Actions
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🚀 Start All Agents", use_container_width=True):
            st.info("Starting all agents...")
            # This would trigger the start_all_agents.sh script
        
        if st.button("🛑 Stop All Agents", use_container_width=True):
            st.warning("Stopping all agents...")
            # This would trigger the stop_all_agents.sh script
        
        st.markdown("---")
        
        # Live Agent Status
        st.markdown("#### 📡 Live Agent Status")
        for agent_id, config in available_agents.items():
            status = st.session_state.agent_statuses.get(agent_id, "checking")
            
            if status == "online":
                indicator_html = '<span class="connectivity-indicator indicator-online"></span>'
                status_text = "Online"
            elif status == "offline":
                indicator_html = '<span class="connectivity-indicator indicator-offline"></span>'
                status_text = "Offline"
            else:
                indicator_html = '<span class="connectivity-indicator indicator-checking"></span>'
                status_text = "Checking..."
            
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.5rem 0;">
                {indicator_html}
                <span style="font-weight: 500;">{config['icon']} {config['name']}</span>
                <br>
                <span style="font-size: 0.8rem; color: #666;">Port {config['port']} - {status_text}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption(f"Last checked: {datetime.fromtimestamp(st.session_state.last_check).strftime('%H:%M:%S')}")
    
    # Main content - Enhanced agent cards
    if available_agents:
        st.markdown("## 🎨 Available AI Agents")
        
        # Create dynamic columns based on number of agents
        num_agents = len(available_agents)
        if num_agents <= 2:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        agent_list = list(available_agents.items())
        
        for i, (agent_id, config) in enumerate(agent_list):
            col_index = i % len(cols)
            
            with cols[col_index]:
                # Get agent status
                status = st.session_state.agent_statuses.get(agent_id, "checking")
                
                # Determine status badge
                if status == "online":
                    status_badge = '<span class="status-badge status-online">🟢 Online</span>'
                elif status == "offline":
                    status_badge = '<span class="status-badge status-offline">🔴 Offline</span>'
                else:
                    status_badge = '<span class="status-badge status-checking">🟡 Checking...</span>'
                
                # Create feature tags
                features_html = ""
                for feature in config["features"][:3]:  # Show first 3 features as tags
                    features_html += f'<span class="feature-tag">{feature}</span> '
                
                st.markdown(f"""
                <div class="agent-card">
                    <h3>{config['icon']} {config['name']}</h3>
                    {status_badge}
                    <p style="margin-top: 1rem; color: #636e72; font-size: 0.95rem;">
                        {config['description']}
                    </p>
                    <div style="margin: 1rem 0;">
                        {features_html}
                    </div>
                    <div class="tech-stack">
                        <strong>🛠️ Tech:</strong> {config['tech_stack']}
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <strong>🔌 Port:</strong> {config['port']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Launch button with conditional styling
                if status == "online":
                    button_text = f"✨ Open {config['name']}"
                    help_text = "Agent is running and ready"
                else:
                    button_text = f"🚀 Launch {config['name']}"
                    help_text = "Click to start the agent"
                
                if st.button(button_text, key=f"{agent_id}_btn", use_container_width=True, help=help_text):
                    if status == "online":
                        st.success(f"✅ {config['name']} is already running!")
                        st.markdown(f"🔗 [Open in browser](http://localhost:{config['port']})")
                    else:
                        st.info(f"Starting {config['name']}...")
                        st.markdown(f"The agent will be available at http://localhost:{config['port']}")
    else:
        # No agents found - enhanced error message
        st.error("### 🔍 No Agents Detected")
        st.markdown("""
        <div class="agent-card" style="background: #fff3cd; border-left-color: #ffc107;">
            <h4>⚠️ Setup Required</h4>
            <p>No agent directories found in the repository. Please ensure you have:</p>
            <ol>
                <li>Cloned the repository with all submodules</li>
                <li>Agent directories present in the <code>agents/</code> folder</li>
                <li>Each agent has its respective app.py file</li>
            </ol>
            <p><strong>Expected structure:</strong></p>
            <pre style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
agents/
├── customer-support-agent/
│   └── src/ui/app.py
├── legal-document-review/
│   └── app.py
├── Finance-Advaisor-Agent/
│   └── app.py
└── ...</pre>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with gradient
    st.markdown("---")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f5f6fa 0%, #ffffff 100%); 
                padding: 2rem; border-radius: 15px; text-align: center;">
        <h3 style="color: #2d3436; margin-bottom: 1rem;">🌟 Multi-Agent AI Platform</h3>
        <p style="color: #636e72;">
            <strong>{len(available_agents)}</strong> Intelligent Assistants | 
            <strong>{online_count}</strong> Currently Online | 
            Powered by LangChain, LangGraph & More
        </p>
        <p style="color: #b2bec3; font-size: 0.9rem; margin-top: 1rem;">
            Built with ❤️ using Streamlit | Docker-Ready | Production-Grade Architecture
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

