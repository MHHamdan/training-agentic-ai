"""
Streamlit UI for Code Review Agent with Multi-Provider Support
Enterprise-grade interface with provider selection and real-time analysis
Author: Mohammed Hamdan
"""

import streamlit as st
import asyncio
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Import agent components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from core.api_key_manager import APIKeyManager
from models.hf_models import HuggingFaceCodeModels
from analyzers.security_analyzer import SecurityAnalyzer
from analyzers.performance_analyzer import PerformanceAnalyzer  
from analyzers.style_analyzer import StyleAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Code Review Agent V1 - Enterprise Code Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .provider-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .analysis-result {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .security-high { border-left: 4px solid #dc3545; }
    .security-medium { border-left: 4px solid #ffc107; }
    .security-low { border-left: 4px solid #28a745; }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "analysis_in_progress" not in st.session_state:
        st.session_state.analysis_in_progress = False
    if "user_api_keys" not in st.session_state:
        st.session_state.user_api_keys = {}
    if "selected_providers" not in st.session_state:
        st.session_state.selected_providers = ["huggingface"]
    if "api_key_manager" not in st.session_state:
        st.session_state.api_key_manager = APIKeyManager()

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">🔍 Code Review Agent V1</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>Enterprise-grade AI code analysis with multi-provider support, "
        "security scanning, and performance optimization</p>",
        unsafe_allow_html=True
    )

def display_provider_configuration():
    """Display provider configuration sidebar"""
    st.sidebar.title("🔧 Provider Configuration")
    
    # Show system status
    with st.sidebar.expander("📊 System Status", expanded=True):
        provider_status = st.session_state.api_key_manager.get_provider_status()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Providers", provider_status["total_providers"])
            st.metric("Free Providers", provider_status["free_providers"])
        with col2:
            st.metric("Paid Providers", provider_status["paid_providers"])
            st.metric("User Providers", provider_status["user_providers"])
        
        recommended = provider_status["recommended_provider"]
        st.info(f"🎯 Recommended: **{recommended.title()}**")
    
    # HuggingFace (Always Available)
    with st.sidebar.expander("🤗 HuggingFace (Free Models)", expanded=True):
        st.success("✅ Using system HuggingFace API key")
        st.info("🆓 **Primary Option** - No costs, always available")
        
        # Show available models
        hf_models = HuggingFaceCodeModels()
        model_count = len(hf_models.get_available_models())
        st.metric("Available Models", f"{model_count} specialized code models")
        
        if st.checkbox("🔧 Show HF Model Details", key="show_hf_details"):
            models = hf_models.get_available_models()
            for task, model_list in models.items():
                st.write(f"**{task.replace('_', ' ').title()}**: {len(model_list)} models")
    
    # User API Keys (Optional)
    st.sidebar.subheader("🔑 Your API Keys (Optional)")
    st.sidebar.markdown("*Add your own API keys for enhanced analysis*")
    
    user_keys = {}
    
    # OpenAI
    with st.sidebar.expander("🤖 OpenAI"):
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            key="openai_key",
            placeholder="sk-...",
            help="Enhanced code analysis with GPT models"
        )
        if openai_key:
            user_keys["openai"] = openai_key
            st.success("✅ OpenAI key provided")
    
    # Anthropic
    with st.sidebar.expander("🧠 Anthropic Claude"):
        anthropic_key = st.text_input(
            "Anthropic API Key",
            type="password", 
            key="anthropic_key",
            placeholder="sk-ant-...",
            help="Advanced reasoning with Claude models"
        )
        if anthropic_key:
            user_keys["anthropic"] = anthropic_key
            st.success("✅ Anthropic key provided")
    
    # Google
    with st.sidebar.expander("🔍 Google Gemini"):
        google_key = st.text_input(
            "Google AI API Key",
            type="password",
            key="google_key", 
            placeholder="AIza...",
            help="Code analysis with Gemini models"
        )
        if google_key:
            user_keys["google"] = google_key
            st.success("✅ Google key provided")
    
    # Groq
    with st.sidebar.expander("⚡ Groq"):
        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            key="groq_key",
            placeholder="gsk_...",
            help="Fast inference with Groq"
        )
        if groq_key:
            user_keys["groq"] = groq_key
            st.success("✅ Groq key provided")
    
    # Mistral
    with st.sidebar.expander("🌟 Mistral AI"):
        mistral_key = st.text_input(
            "Mistral API Key",
            type="password",
            key="mistral_key",
            placeholder="...",
            help="European AI with Mistral models"
        )
        if mistral_key:
            user_keys["mistral"] = mistral_key
            st.success("✅ Mistral key provided")
    
    # Update API key manager
    if user_keys:
        st.session_state.api_key_manager.set_user_api_keys(user_keys)
        st.session_state.user_api_keys = user_keys
    
    # Provider Selection
    st.sidebar.subheader("🎯 Analysis Configuration")
    
    available_providers = st.session_state.api_key_manager.get_available_providers()
    provider_options = []
    
    for provider_id, provider_info in available_providers.items():
        if provider_info["status"] == "available":
            provider_options.append({
                "id": provider_id,
                "label": provider_info["name"],
                "type": provider_info["type"]
            })
    
    # Sort by priority (free first, then user, then system)
    provider_options.sort(key=lambda x: (
        0 if x["type"] == "free" else 1 if "Your API Key" in x["label"] else 2
    ))
    
    # Determine default providers
    available_provider_ids = [p["id"] for p in provider_options]
    default_providers = []
    if "huggingface" in available_provider_ids:
        default_providers = ["huggingface"]
    elif available_provider_ids:
        default_providers = [available_provider_ids[0]]
    
    selected_provider_ids = st.sidebar.multiselect(
        "Select Providers for Analysis",
        options=available_provider_ids,
        default=default_providers,
        format_func=lambda x: next(p["label"] for p in provider_options if p["id"] == x),
        help="Choose which providers to use for code analysis"
    )
    
    st.session_state.selected_providers = selected_provider_ids
    
    # Analysis Settings
    with st.sidebar.expander("⚙️ Analysis Settings"):
        enable_security = st.checkbox("🔒 Security Analysis", value=True)
        enable_performance = st.checkbox("⚡ Performance Analysis", value=True)
        enable_style = st.checkbox("🎨 Style Analysis", value=True)
        include_ai_analysis = st.checkbox("🤖 AI-Enhanced Analysis", value=True)
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            options=["quick", "standard", "comprehensive"],
            index=1
        )
    
    return {
        "providers": selected_provider_ids,
        "enable_security": enable_security,
        "enable_performance": enable_performance,
        "enable_style": enable_style,
        "include_ai_analysis": include_ai_analysis,
        "analysis_depth": analysis_depth
    }

def display_code_input():
    """Display code input interface"""
    st.markdown("### 📝 Code Analysis")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        options=["Direct Input", "File Upload"],
        horizontal=True
    )
    
    code = ""
    context = ""
    
    if input_method == "Direct Input":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            code = st.text_area(
                "Enter your Python code:",
                height=400,
                placeholder="""def example_function(data):
    # Your code here
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result""",
                key="code_input"
            )
        
        with col2:
            context = st.text_area(
                "Context (optional):",
                height=180,
                placeholder="Describe the purpose of this code, any specific concerns, or requirements...",
                key="context_input"
            )
            
            # Code statistics
            if code:
                lines = len([line for line in code.split('\n') if line.strip()])
                chars = len(code)
                st.metric("Lines of Code", lines)
                st.metric("Characters", chars)
    
    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload Python file:",
            type=['py'],
            help="Upload a .py file for analysis"
        )
        
        if uploaded_file:
            code = uploaded_file.read().decode('utf-8')
            st.success(f"✅ Loaded {uploaded_file.name}")
            
            # Show preview
            if st.checkbox("Show file preview"):
                st.code(code[:500] + "..." if len(code) > 500 else code, language="python")
            
            context = st.text_input(
                "Additional context:",
                placeholder="Any specific concerns or requirements for this file..."
            )
    
    return code, context

async def run_code_analysis(code: str, context: str, settings: Dict[str, Any]):
    """Run comprehensive code analysis"""
    try:
        st.session_state.analysis_in_progress = True
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "code_length": len(code),
            "context": context,
            "settings": settings,
            "providers_used": settings["providers"]
        }
        
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Initialize analyzers
        analyzers = {}
        if settings["enable_security"]:
            analyzers["security"] = SecurityAnalyzer()
        if settings["enable_performance"]:
            analyzers["performance"] = PerformanceAnalyzer()
        if settings["enable_style"]:
            analyzers["style"] = StyleAnalyzer()
        
        total_analyses = len(analyzers)
        completed = 0
        
        # Run analyses
        for analyzer_name, analyzer in analyzers.items():
            status_text.text(f"🔄 Running {analyzer_name} analysis...")
            
            if analyzer_name == "security":
                result = await analyzer.analyze_security(
                    code, context, settings["include_ai_analysis"]
                )
            elif analyzer_name == "performance":
                result = await analyzer.analyze_performance(
                    code, context, settings["include_ai_analysis"]
                )
            elif analyzer_name == "style":
                result = await analyzer.analyze_style(
                    code, context, settings["include_ai_analysis"]
                )
            
            results[analyzer_name] = result
            completed += 1
            progress_bar.progress(completed / total_analyses)
        
        # Calculate overall score
        scores = []
        if "security" in results:
            scores.append(results["security"].security_score)
        if "performance" in results:
            scores.append(results["performance"].performance_score)
        if "style" in results:
            scores.append(results["style"].style_score)
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        results["overall_score"] = overall_score
        
        # Calculate total issues
        total_issues = 0
        if "security" in results:
            total_issues += results["security"].total_issues
        if "performance" in results:
            total_issues += results["performance"].total_issues
        if "style" in results:
            total_issues += results["style"].total_issues
        
        results["total_issues"] = total_issues
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        st.error(f"Analysis failed: {str(e)}")
        return None
    
    finally:
        st.session_state.analysis_in_progress = False

def display_analysis_results(results: Dict[str, Any]):
    """Display comprehensive analysis results"""
    if not results:
        return
    
    st.markdown("## 📊 Analysis Results")
    
    # Overall metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        score = results.get("overall_score", 0)
        color = "#4CAF50" if score > 7 else "#FFC107" if score > 4 else "#F44336"
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #666; margin: 0;">Overall Score</p>
            <p style="color: {color}; font-size: 2rem; font-weight: bold; margin: 0;">{score:.1f}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Total Issues", results.get("total_issues", 0))
    
    with col3:
        st.metric("Code Length", f"{results.get('code_length', 0)} chars")
    
    with col4:
        providers = results.get("providers_used", [])
        st.metric("Providers Used", len(providers))
    
    with col5:
        analysis_time = 0
        if "security" in results:
            analysis_time += results["security"].analysis_time
        if "performance" in results:
            analysis_time += results["performance"].analysis_time
        if "style" in results:
            analysis_time += results["style"].analysis_time
        st.metric("Analysis Time", f"{analysis_time:.1f}s")
    
    # Analysis tabs
    tabs = []
    if "security" in results:
        tabs.append("🔒 Security")
    if "performance" in results:
        tabs.append("⚡ Performance")
    if "style" in results:
        tabs.append("🎨 Style")
    tabs.extend(["📊 Summary", "📈 Charts"])
    
    tab_objects = st.tabs(tabs)
    tab_index = 0
    
    # Security Results
    if "security" in results:
        with tab_objects[tab_index]:
            display_security_results(results["security"])
        tab_index += 1
    
    # Performance Results
    if "performance" in results:
        with tab_objects[tab_index]:
            display_performance_results(results["performance"])
        tab_index += 1
    
    # Style Results
    if "style" in results:
        with tab_objects[tab_index]:
            display_style_results(results["style"])
        tab_index += 1
    
    # Summary
    with tab_objects[tab_index]:
        display_summary_results(results)
    tab_index += 1
    
    # Charts
    with tab_objects[tab_index]:
        display_analysis_charts(results)

def display_security_results(security_result):
    """Display security analysis results"""
    st.markdown("### 🔒 Security Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Security Score", f"{security_result.security_score:.1f}/10")
    with col2:
        st.metric("Critical Issues", security_result.critical_issues)
    with col3:
        st.metric("High Issues", security_result.high_issues)
    with col4:
        st.metric("Total Issues", security_result.total_issues)
    
    if security_result.vulnerabilities:
        st.markdown("#### 🚨 Vulnerabilities Found")
        
        for vuln in security_result.vulnerabilities:
            severity_class = f"security-{vuln.severity.value}"
            
            with st.expander(f"🔸 {vuln.title} - {vuln.severity.value.upper()}", expanded=False):
                st.markdown(f"**Description:** {vuln.description}")
                if vuln.line_number:
                    st.markdown(f"**Line:** {vuln.line_number}")
                if vuln.code_snippet:
                    st.code(vuln.code_snippet, language="python")
                if vuln.recommendation:
                    st.markdown(f"**Recommendation:** {vuln.recommendation}")
                if vuln.cwe_id:
                    st.markdown(f"**CWE ID:** {vuln.cwe_id}")
                st.progress(vuln.confidence)
    
    else:
        st.success("✅ No security vulnerabilities detected!")
    
    if security_result.recommendations:
        st.markdown("#### 💡 Security Recommendations")
        for rec in security_result.recommendations:
            st.markdown(f"• {rec}")

def display_performance_results(performance_result):
    """Display performance analysis results"""
    st.markdown("### ⚡ Performance Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Performance Score", f"{performance_result.performance_score:.1f}/10")
    with col2:
        st.metric("Critical Issues", performance_result.critical_issues)
    with col3:
        st.metric("High Issues", performance_result.high_issues)
    with col4:
        st.metric("Total Issues", performance_result.total_issues)
    
    if performance_result.issues:
        st.markdown("#### 🔧 Performance Issues")
        
        for issue in performance_result.issues:
            with st.expander(f"⚡ {issue.title} - {issue.impact.value.upper()}", expanded=False):
                st.markdown(f"**Description:** {issue.description}")
                if issue.line_number:
                    st.markdown(f"**Line:** {issue.line_number}")
                if issue.code_snippet:
                    st.code(issue.code_snippet, language="python")
                if issue.recommendation:
                    st.markdown(f"**Recommendation:** {issue.recommendation}")
                if issue.complexity_before and issue.complexity_after:
                    st.markdown(f"**Complexity:** {issue.complexity_before} → {issue.complexity_after}")
                st.progress(issue.confidence)
    
    else:
        st.success("✅ No performance issues detected!")
    
    if performance_result.optimizations:
        st.markdown("#### 🚀 Optimization Suggestions")
        for opt in performance_result.optimizations:
            st.markdown(f"• {opt}")
    
    if performance_result.complexity_metrics:
        st.markdown("#### 📊 Complexity Metrics")
        metrics_df = pd.DataFrame([performance_result.complexity_metrics])
        st.dataframe(metrics_df, use_container_width=True)

def display_style_results(style_result):
    """Display style analysis results"""
    st.markdown("### 🎨 Style Analysis Results")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Style Score", f"{style_result.style_score:.1f}/10")
    with col2:
        st.metric("PEP8 Compliance", f"{style_result.pep8_compliance:.1%}")
    with col3:
        st.metric("Errors", style_result.error_count)
    with col4:
        st.metric("Total Issues", style_result.total_issues)
    
    if style_result.issues:
        st.markdown("#### 🎯 Style Issues")
        
        for issue in style_result.issues:
            with st.expander(f"🎨 {issue.title} - {issue.severity.value.upper()}", expanded=False):
                st.markdown(f"**Description:** {issue.description}")
                if issue.line_number:
                    st.markdown(f"**Line:** {issue.line_number}")
                if issue.code_snippet:
                    st.code(issue.code_snippet, language="python")
                if issue.fix_suggestion:
                    st.markdown(f"**Fix:** {issue.fix_suggestion}")
                if issue.rule_code:
                    st.markdown(f"**Rule:** {issue.rule_code}")
                st.progress(issue.confidence)
    
    else:
        st.success("✅ No style issues detected!")
    
    if style_result.improvements:
        st.markdown("#### ✨ Style Improvements")
        for imp in style_result.improvements:
            st.markdown(f"• {imp}")

def display_summary_results(results: Dict[str, Any]):
    """Display analysis summary"""
    st.markdown("### 📋 Analysis Summary")
    
    # Create summary table
    summary_data = []
    
    if "security" in results:
        sec = results["security"]
        summary_data.append({
            "Analysis Type": "🔒 Security",
            "Score": f"{sec.security_score:.1f}/10",
            "Issues": sec.total_issues,
            "Critical": sec.critical_issues,
            "Status": "✅ Pass" if sec.security_score > 7 else "⚠️ Review" if sec.security_score > 4 else "❌ Fail"
        })
    
    if "performance" in results:
        perf = results["performance"]
        summary_data.append({
            "Analysis Type": "⚡ Performance",
            "Score": f"{perf.performance_score:.1f}/10",
            "Issues": perf.total_issues,
            "Critical": perf.critical_issues,
            "Status": "✅ Pass" if perf.performance_score > 7 else "⚠️ Review" if perf.performance_score > 4 else "❌ Fail"
        })
    
    if "style" in results:
        style = results["style"]
        summary_data.append({
            "Analysis Type": "🎨 Style",
            "Score": f"{style.style_score:.1f}/10",
            "Issues": style.total_issues,
            "Critical": style.error_count,
            "Status": "✅ Pass" if style.style_score > 7 else "⚠️ Review" if style.style_score > 4 else "❌ Fail"
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    
    # Overall recommendation
    overall_score = results.get("overall_score", 0)
    if overall_score > 8:
        st.success("🎉 **Excellent Code Quality** - Your code meets high standards!")
    elif overall_score > 6:
        st.warning("📈 **Good Code Quality** - Minor improvements recommended")
    elif overall_score > 4:
        st.warning("🔧 **Moderate Issues** - Several areas need attention")
    else:
        st.error("🚨 **Significant Issues** - Major improvements required")

def display_analysis_charts(results: Dict[str, Any]):
    """Display analysis charts and visualizations"""
    st.markdown("### 📈 Analysis Visualizations")
    
    # Score comparison chart
    scores = {}
    if "security" in results:
        scores["Security"] = results["security"].security_score
    if "performance" in results:
        scores["Performance"] = results["performance"].performance_score
    if "style" in results:
        scores["Style"] = results["style"].style_score
    
    if scores:
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart for scores
            fig = go.Figure()
            
            categories = list(scores.keys())
            values = list(scores.values())
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Code Quality'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Code Quality Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart for issues
            issue_counts = {}
            if "security" in results:
                issue_counts["Security"] = results["security"].total_issues
            if "performance" in results:
                issue_counts["Performance"] = results["performance"].total_issues
            if "style" in results:
                issue_counts["Style"] = results["style"].total_issues
            
            if any(issue_counts.values()):
                fig = px.bar(
                    x=list(issue_counts.keys()),
                    y=list(issue_counts.values()),
                    title="Issues by Category",
                    labels={"x": "Analysis Type", "y": "Number of Issues"}
                )
                st.plotly_chart(fig, use_container_width=True)

def display_analysis_history():
    """Display analysis history"""
    st.markdown("### 📜 Analysis History")
    
    if st.session_state.analysis_history:
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-10:])):
            with st.expander(f"Analysis {len(st.session_state.analysis_history) - i} - {analysis['timestamp'][:19]}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Overall Score", f"{analysis.get('overall_score', 0):.1f}/10")
                with col2:
                    st.metric("Total Issues", analysis.get('total_issues', 0))
                with col3:
                    st.metric("Code Length", f"{analysis.get('code_length', 0)} chars")
                
                if st.button(f"View Details", key=f"view_{i}"):
                    st.session_state.current_analysis = analysis
                    st.rerun()
    else:
        st.info("No analysis history available. Run your first code analysis above!")

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Provider configuration (sidebar)
    settings = display_provider_configuration()
    
    # Main content
    st.markdown("---")
    
    # Code input
    code, context = display_code_input()
    
    # Analysis controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_button = st.button(
            "🚀 Analyze Code",
            type="primary",
            disabled=st.session_state.analysis_in_progress or not code.strip(),
            use_container_width=True
        )
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.current_analysis = None
            st.rerun()
    
    # Run analysis
    if analyze_button:
        if not settings["providers"]:
            st.error("❌ Please select at least one provider for analysis")
        else:
            with st.spinner("🔄 Running comprehensive code analysis..."):
                try:
                    analysis_result = asyncio.run(run_code_analysis(code, context, settings))
                    if analysis_result:
                        st.session_state.current_analysis = analysis_result
                        st.session_state.analysis_history.append(analysis_result)
                        st.success("✅ Analysis completed successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results
    if st.session_state.current_analysis:
        st.markdown("---")
        display_analysis_results(st.session_state.current_analysis)
    
    # Analysis history (in sidebar)
    with st.sidebar:
        st.markdown("---")
        if st.session_state.analysis_history:
            st.markdown("### 📊 Quick Stats")
            total_analyses = len(st.session_state.analysis_history)
            avg_score = sum(a.get('overall_score', 0) for a in st.session_state.analysis_history) / total_analyses
            st.metric("Total Analyses", total_analyses)
            st.metric("Average Score", f"{avg_score:.1f}/10")

if __name__ == "__main__":
    main()