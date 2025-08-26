"""
Research Agent V2 - Streamlit Application
Enterprise-grade research agent with Langfuse observability
"""

import streamlit as st
import asyncio
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import logging

# Import agent components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from graph.workflow_manager import workflow_manager
from graph.state import ResearchPhase, get_state_summary

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Research Agent V2 - AI-Powered Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .status-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .phase-indicator {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: bold;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "current_research" not in st.session_state:
        st.session_state.current_research = None
    if "research_in_progress" not in st.session_state:
        st.session_state.research_in_progress = False
    if "selected_models" not in st.session_state:
        st.session_state.selected_models = []
    if "langfuse_enabled" not in st.session_state:
        st.session_state.langfuse_enabled = config.langfuse.enabled
    if "last_research_result" not in st.session_state:
        st.session_state.last_research_result = None
    if "show_success_message" not in st.session_state:
        st.session_state.show_success_message = False

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">🔬 Research Agent V2</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>Enterprise-grade AI research with Langfuse observability, "
        "multi-model support, and academic compliance</p>",
        unsafe_allow_html=True
    )

def display_configuration_status():
    """Display configuration and system status"""
    with st.expander("⚙️ System Configuration", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Langfuse Status",
                "✅ Enabled" if config.langfuse.enabled else "❌ Disabled",
                delta="Observability Active" if config.langfuse.enabled else None
            )
        
        with col2:
            validation = config.validate_configuration()
            valid_count = sum(validation.values())
            st.metric(
                "Configuration",
                f"{valid_count}/{len(validation)} Valid",
                delta="Ready" if all(validation.values()) else "Check Settings"
            )
        
        with col3:
            st.metric(
                "Model Provider",
                config.get_model_by_task("general_research").split("/")[0],
                delta="Active"
            )
        
        with col4:
            st.metric(
                "Agent Version",
                config.agent_version,
                delta=f"Port {config.port}"
            )
        
        # Detailed configuration
        if st.checkbox("Show detailed configuration"):
            st.json(validation)

def display_research_interface():
    """Display main research interface"""
    st.markdown("### 🔍 New Research Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your research question",
            placeholder="Example: What are the latest developments in quantum computing and their potential applications in cryptography?",
            height=100,
            key="research_query"
        )
    
    with col2:
        st.markdown("#### Research Settings")
        
        depth = st.select_slider(
            "Analysis Depth",
            options=["quick", "standard", "comprehensive", "exhaustive"],
            value="comprehensive"
        )
        
        citation_format = st.selectbox(
            "Citation Format",
            options=["APA", "MLA", "Chicago", "IEEE", "Harvard"],
            index=0
        )
        
        enable_fact_check = st.checkbox("Enable Fact Checking", value=True)
        enable_comparison = st.checkbox("Enable Model Comparison", value=config.models.enable_model_comparison)
    
    # Model Configuration
    with st.expander("🤖 Model Configuration"):
        st.markdown("### Choose Your Research Models")
        
        model_option = st.radio(
            "Model Provider",
            options=[
                "🆓 Use Free Hugging Face Models (Recommended)", 
                "🔑 Use Your Own API Keys",
                "⚙️ Use System Default Keys"
            ],
            index=0
        )
        
        if model_option == "🔑 Use Your Own API Keys":
            st.markdown("#### Enter Your API Keys")
            col1, col2 = st.columns(2)
            
            with col1:
                user_openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    placeholder="sk-...",
                    help="Enter your OpenAI API key for GPT models"
                )
                
                user_anthropic_key = st.text_input(
                    "Anthropic API Key", 
                    type="password",
                    placeholder="sk-ant-...",
                    help="Enter your Anthropic API key for Claude models"
                )
            
            with col2:
                user_google_key = st.text_input(
                    "Google AI API Key",
                    type="password", 
                    placeholder="AIza...",
                    help="Enter your Google AI API key for Gemini models"
                )
                
                user_hf_key = st.text_input(
                    "Hugging Face API Key",
                    type="password",
                    placeholder="hf_...", 
                    help="Enter your Hugging Face API key for better rate limits"
                )
            
            # Store user keys in session state
            st.session_state.user_api_keys = {
                "openai": user_openai_key,
                "anthropic": user_anthropic_key, 
                "google": user_google_key,
                "huggingface": user_hf_key
            }
            
        elif model_option == "🆓 Use Free Hugging Face Models (Recommended)":
            st.success("✅ Using free Hugging Face models - no API keys required!")
            st.info("📝 Free models include: microsoft/phi-3-mini-4k-instruct, google/gemma-2b-it, and more")
            st.session_state.use_free_models = True
            
        else:  # System default
            st.warning("⚠️ Using system default API keys (may have quota limits)")
            st.session_state.use_free_models = False
    
    # Advanced options
    with st.expander("🎯 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_sources = st.slider(
                "Maximum Sources",
                min_value=5,
                max_value=50,
                value=config.research.max_search_results,
                step=5
            )
            
            quality_threshold = st.slider(
                "Quality Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.research.quality_score_minimum,
                step=0.1
            )
        
        with col2:
            selected_sources = st.multiselect(
                "Search Sources",
                options=["DuckDuckGo", "ArXiv", "Wikipedia", "News", "PubMed"],
                default=["DuckDuckGo", "ArXiv", "Wikipedia"]
            )
            
            output_format = st.selectbox(
                "Output Format",
                options=["Detailed Report", "Executive Summary", "Academic Paper", "Presentation"],
                index=0
            )
        
        with col3:
            if enable_comparison:
                st.markdown("**Model Selection**")
                selected_models = st.multiselect(
                    "Compare Models",
                    options=config.models.general_research_models + config.models.text_generation_models,
                    default=[config.models.default_model],
                    max_selections=3
                )
                st.session_state.selected_models = selected_models
    
    # Start research button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_button = st.button(
            "🚀 Start Research",
            type="primary",
            disabled=st.session_state.research_in_progress or not query,
            use_container_width=True
        )
    
    with col2:
        if st.button("🛑 Stop Research", disabled=not st.session_state.research_in_progress):
            st.session_state.research_in_progress = False
            st.rerun()
    
    return {
        "query": query,
        "depth": depth,
        "citation_format": citation_format,
        "enable_fact_check": enable_fact_check,
        "enable_comparison": enable_comparison,
        "max_sources": max_sources,
        "quality_threshold": quality_threshold,
        "selected_sources": selected_sources,
        "output_format": output_format,
        "model_option": model_option,
        "user_api_keys": st.session_state.get("user_api_keys", {}),
        "use_free_models": st.session_state.get("use_free_models", True),
        "start": start_button
    }

def display_research_progress(phase: ResearchPhase, progress: float):
    """Display research progress indicator"""
    phase_colors = {
        ResearchPhase.INITIALIZATION: "#gray",
        ResearchPhase.SEARCH: "#blue",
        ResearchPhase.ANALYSIS: "#orange",
        ResearchPhase.SYNTHESIS: "#green",
        ResearchPhase.EVALUATION: "#purple",
        ResearchPhase.COMPLETE: "#teal",
        ResearchPhase.ERROR: "#red"
    }
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        color = phase_colors.get(phase, "#gray")
        st.markdown(
            f'<div class="phase-indicator" style="background: {color};">{phase.value.upper()}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.progress(progress, text=f"Research Progress: {int(progress * 100)}%")

def display_research_results(state: Dict[str, Any]):
    """Display comprehensive research results"""
    st.markdown("## 📊 Research Results")
    
    # Quality metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        quality_score = state.get("quality_score", 0) or 0
        color = "#4CAF50" if quality_score > 0.7 else "#FFC107" if quality_score > 0.5 else "#F44336"
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #666; margin: 0;">Overall Quality</p>
            <p class="quality-score" style="color: {color};">{quality_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Sources Analyzed",
            len(state.get("search_results", [])),
            delta=f"{len(state.get('verified_claims', []))} verified"
        )
    
    with col3:
        st.metric(
            "Key Insights",
            len(state.get("key_insights", [])),
            delta=f"{len(state.get('recommendations', []))} recommendations"
        )
    
    with col4:
        accuracy = state.get("accuracy_score", 0) or 0
        st.metric(
            "Accuracy Score",
            f"{accuracy:.1%}",
            delta="High" if accuracy > 0.8 else "Moderate"
        )
    
    with col5:
        processing_time = sum(state.get("processing_time", {}).values())
        st.metric(
            "Processing Time",
            f"{processing_time:.1f}s",
            delta="Fast" if processing_time < 60 else "Standard"
        )
    
    # Main content tabs
    tabs = st.tabs([
        "📝 Executive Summary",
        "🔍 Detailed Findings",
        "💡 Key Insights",
        "📚 Citations",
        "✅ Fact Check",
        "📊 Analytics",
        "⚠️ Issues"
    ])
    
    with tabs[0]:  # Executive Summary
        st.markdown("### Executive Summary")
        summary = state.get("executive_summary", "No summary available")
        st.markdown(summary)
        
        if state.get("recommendations"):
            st.markdown("### 🎯 Recommendations")
            for i, rec in enumerate(state["recommendations"], 1):
                st.markdown(f"{i}. {rec}")
    
    with tabs[1]:  # Detailed Findings
        st.markdown("### Detailed Research Findings")
        synthesis = state.get("synthesis", "No synthesis available")
        st.markdown(synthesis)
        
        if state.get("detailed_findings"):
            for section, content in state["detailed_findings"].items():
                with st.expander(f"📌 {section}"):
                    st.markdown(content)
    
    with tabs[2]:  # Key Insights
        st.markdown("### 💡 Key Insights Extracted")
        insights = state.get("key_insights", [])
        if insights:
            for i, insight in enumerate(insights, 1):
                st.info(f"**Insight {i}:** {insight}")
        else:
            st.warning("No key insights extracted")
    
    with tabs[3]:  # Citations
        st.markdown("### 📚 References and Citations")
        citations = state.get("citations", [])
        if citations:
            for citation in citations:
                st.markdown(f"- {citation}")
        
        # Bibliography
        if state.get("bibliography"):
            st.markdown("### Bibliography")
            df = pd.DataFrame(state["bibliography"])
            st.dataframe(df, use_container_width=True)
    
    with tabs[4]:  # Fact Check
        st.markdown("### ✅ Fact Checking Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Verified Claims")
            verified = state.get("verified_claims", [])
            if verified:
                for claim in verified:
                    st.success(f"✓ {claim}")
            else:
                st.info("No claims verified")
        
        with col2:
            st.markdown("#### Disputed Claims")
            disputed = state.get("disputed_claims", [])
            if disputed:
                for claim in disputed:
                    st.error(f"✗ {claim}")
            else:
                st.info("No disputed claims found")
    
    with tabs[5]:  # Analytics
        display_research_analytics(state)
    
    with tabs[6]:  # Issues
        errors = state.get("errors", [])
        warnings = state.get("warnings", [])
        
        if errors:
            st.markdown("### 🚨 Errors")
            for error in errors:
                st.error(error)
        
        if warnings:
            st.markdown("### ⚠️ Warnings")
            for warning in warnings:
                st.warning(warning)
        
        if not errors and not warnings:
            st.success("✅ No issues detected during research")

def display_research_analytics(state: Dict[str, Any]):
    """Display research analytics and visualizations"""
    st.markdown("### 📊 Research Analytics")
    
    # Processing time breakdown
    if state.get("processing_time"):
        fig = px.pie(
            values=list(state["processing_time"].values()),
            names=list(state["processing_time"].keys()),
            title="Processing Time Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Source quality distribution
    if state.get("source_quality_scores"):
        scores = list(state["source_quality_scores"].values())
        fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=20)])
        fig.update_layout(
            title="Source Quality Distribution",
            xaxis_title="Quality Score",
            yaxis_title="Number of Sources"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model performance comparison
    if state.get("model_comparisons"):
        st.markdown("#### Model Performance Comparison")
        comparison_df = pd.DataFrame(state["model_comparisons"])
        st.dataframe(comparison_df, use_container_width=True)

def display_langfuse_dashboard():
    """Display Langfuse observability dashboard"""
    st.markdown("### 📈 Langfuse Observability Dashboard")
    
    if config.langfuse.enabled:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Project:** {config.langfuse.project}")
            st.info(f"**Organization:** {config.langfuse.organization}")
            st.info(f"**Environment:** {config.langfuse.environment}")
        
        with col2:
            if st.button("🔗 Open Langfuse Dashboard"):
                st.markdown(f"[Open Langfuse Dashboard]({config.langfuse.host})")
            
            st.markdown("**Active Traces:**")
            if st.session_state.current_research:
                trace_id = st.session_state.current_research.get("trace_id")
                if trace_id:
                    st.code(trace_id)
    else:
        st.warning("Langfuse observability is disabled. Enable it in configuration to track research workflows.")

def display_research_history():
    """Display research history"""
    st.markdown("### 📜 Research History")
    
    if st.session_state.research_history:
        for i, research in enumerate(reversed(st.session_state.research_history[-10:])):
            with st.expander(f"🔍 {research['query'][:100]}... ({research['timestamp']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Quality Score", f"{research.get('quality_score', 0):.1%}")
                
                with col2:
                    st.metric("Sources", research.get("source_count", 0))
                
                with col3:
                    st.metric("Processing Time", f"{research.get('processing_time', 0):.1f}s")
                
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.current_research = research
                    st.rerun()
    else:
        st.info("No research history available yet. Start your first research above!")

async def run_research(params: Dict[str, Any]):
    """Execute research workflow"""
    try:
        # Update session state
        st.session_state.research_in_progress = True
        
        # Prepare user preferences including model configuration
        user_preferences = {
            "depth": params["depth"],
            "citation_format": params["citation_format"],
            "enable_fact_check": params["enable_fact_check"],
            "max_sources": params["max_sources"],
            "quality_threshold": params["quality_threshold"],
            "output_format": params["output_format"],
            "model_option": params.get("model_option", "🆓 Use Free Hugging Face Models (Recommended)"),
            "user_api_keys": params.get("user_api_keys", {}),
            "use_free_models": params.get("use_free_models", True)
        }
        
        # Run research workflow
        result = await workflow_manager.run_research(
            query=params["query"],
            user_preferences=user_preferences
        )
        
        # Store result
        st.session_state.current_research = result
        st.session_state.research_history.append({
            **get_state_summary(result),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Update session state
        st.session_state.research_in_progress = False
        
        return result
        
    except Exception as e:
        logger.error(f"Research execution error: {e}")
        st.error(f"Research failed: {str(e)}")
        st.session_state.research_in_progress = False
        return None

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Display configuration status
    display_configuration_status()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔬 Research Agent V2")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 New Research", "📊 Current Results", "📜 History", "📈 Observability"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Researches", len(st.session_state.research_history))
        
        if st.session_state.research_history:
            quality_scores = [r.get("quality_score", 0) or 0 for r in st.session_state.research_history]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            st.metric("Avg Quality Score", f"{avg_quality:.1%}")
        
        st.markdown("---")
        
        # Configuration
        if st.checkbox("🔧 Show Configuration"):
            st.json(config.validate_configuration())
    
    # Main content area
    if page == "🔍 New Research":
        params = display_research_interface()
        
        if params["start"]:
            # Run research asynchronously
            with st.spinner("🔬 Conducting research..."):
                try:
                    result = asyncio.run(run_research(params))
                    if result:
                        # Store success message in session state (persists after rerun)
                        quality_score = result.get('quality_score', 0)
                        insights_count = len(result.get('key_insights', []))
                        
                        st.session_state.last_research_result = {
                            'quality_score': quality_score,
                            'insights_count': insights_count,
                            'query': result.get('query', ''),
                            'executive_summary': result.get('executive_summary', '')
                        }
                        st.session_state.show_success_message = True
                        
                    else:
                        st.error("❌ Research failed. Please try again.")
                        st.session_state.show_success_message = False
                        
                except Exception as e:
                    st.error(f"❌ Research execution error: {str(e)}")
                    logger.error(f"Research execution error: {e}")
                    st.session_state.show_success_message = False
            
            # Force UI refresh
            st.rerun()
        
        # Show success message if available (persists after rerun)
        if st.session_state.show_success_message and st.session_state.last_research_result:
            result_info = st.session_state.last_research_result
            
            st.success("🎉 Research completed successfully!")
            st.info(f"📊 Quality Score: {result_info['quality_score']:.1%} | 💡 Insights: {result_info['insights_count']} | 🔍 Click 'Current Results' to view full findings")
            
            # Show a sample of results immediately
            if result_info.get('executive_summary'):
                with st.expander("📝 Quick Preview"):
                    summary = result_info['executive_summary']
                    st.write(summary[:200] + "..." if len(summary) > 200 else summary)
            
            # Clear the message after showing it
            if st.button("✅ Got it! Clear this message"):
                st.session_state.show_success_message = False
                st.rerun()
        
        # Show progress if research is in progress
        if st.session_state.research_in_progress:
            display_research_progress(ResearchPhase.SEARCH, 0.25)
    
    elif page == "📊 Current Results":
        if st.session_state.current_research:
            # Show results are available with key metrics
            quality_score = st.session_state.current_research.get('quality_score', 0)
            query = st.session_state.current_research.get('query', 'Unknown Query')
            st.success(f"✅ Research Results: '{query}' | Quality Score: {quality_score:.1%}")
            display_research_results(st.session_state.current_research)
        else:
            st.info("No current research results. Start a new research or select from history.")
            # Show available options
            col1, col2 = st.columns(2)
            with col1:
                st.info("👆 Use the navigation on the left to start a new research")
            with col2:
                if hasattr(st.session_state, 'research_history') and st.session_state.research_history:
                    st.info(f"📜 {len(st.session_state.research_history)} research(es) available in History tab")
    
    elif page == "📜 History":
        display_research_history()
    
    elif page == "📈 Observability":
        display_langfuse_dashboard()

if __name__ == "__main__":
    main()