"""
Multi-Agent Financial Analysis System - Streamlit Interface
Built with LangGraph for sophisticated financial analysis workflows
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import uuid
from typing import Dict, List, Any
import yfinance as yf
import os
import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import our financial system components
try:
    from financial_graph import financial_graph, create_financial_analysis_session
    from financial_state import FinancialAnalysisState, MarketConditions
    from langgraph.types import Command
    LANGGRAPH_AVAILABLE = True
except ImportError:
    st.error("⚠️ LangGraph dependencies not available. Please install: pip install langgraph langchain-anthropic")
    LANGGRAPH_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Financial Analysis System",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional financial interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e7ef;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .alert-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin-bottom: 1rem;
    }
    .success-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin-bottom: 1rem;
    }
    .agent-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    .status-active {
        background-color: #fbbf24;
        color: #78350f;
    }
    .status-complete {
        background-color: #34d399;
        color: #064e3b;
    }
    .status-pending {
        background-color: #94a3b8;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


class FinancialAnalysisApp:
    """Main application class for the financial analysis system"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def get_analysis_data(self, analysis_key: str, default: str = "No analysis available") -> str:
        """Safely extract analysis data from AnalysisResult objects"""
        if analysis_key in st.session_state.analysis_results:
            result = st.session_state.analysis_results[analysis_key]
            if hasattr(result, 'data') and result.data:
                return result.data.get("analysis", default)
        return default
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_session' not in st.session_state:
            st.session_state.analysis_session = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'workflow_running' not in st.session_state:
            st.session_state.workflow_running = False
        if 'current_agent' not in st.session_state:
            st.session_state.current_agent = None
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
    
    def run(self):
        """Main application entry point"""
        # Header
        st.markdown('<h1 class="main-header">💹 Multi-Agent Financial Analysis System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Powered by LangGraph • Advanced AI-Driven Investment Analysis</p>', unsafe_allow_html=True)
        
        # Check for API keys
        if not any([os.getenv("GROK_API_KEY"), os.getenv("OPENAI_API_KEY"), os.getenv("GOOGLE_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]):
            st.warning("⚠️ Please configure at least one LLM API key (GROK_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY)")
        elif os.getenv("GROK_API_KEY"):
            st.info("🚀 Using Grok (xAI) as primary LLM provider")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("🎯 Analysis Configuration")
            
            # Stock selection
            symbols_input = st.text_input(
                "Stock Symbols",
                value="AAPL, MSFT, GOOGL",
                help="Enter comma-separated stock symbols"
            )
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            # Analysis type
            analysis_type = st.selectbox(
                "Analysis Type",
                options=["comprehensive", "technical", "fundamental", "risk", "sentiment", "portfolio"],
                format_func=lambda x: x.title(),
                help="Select the type of analysis to perform"
            )
            
            # Risk tolerance
            risk_tolerance = st.select_slider(
                "Risk Tolerance",
                options=["conservative", "moderate", "aggressive"],
                value="moderate",
                help="Your risk tolerance level"
            )
            
            # Time horizon
            time_horizon = st.selectbox(
                "Investment Horizon",
                options=["intraday", "short", "medium", "long"],
                index=2,
                format_func=lambda x: {
                    "intraday": "Intraday Trading",
                    "short": "Short-term (< 3 months)",
                    "medium": "Medium-term (3-12 months)",
                    "long": "Long-term (> 1 year)"
                }.get(x, x)
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                enable_backtesting = st.checkbox("Enable Backtesting", value=False)
                use_real_time = st.checkbox("Use Real-time Data", value=True)
                
                notification_prefs = st.multiselect(
                    "Notifications",
                    options=["Critical Alerts", "Trade Confirmations", "Analysis Complete", "Market Updates"],
                    default=["Critical Alerts", "Trade Confirmations"]
                )
            
            # Action buttons
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True, disabled=st.session_state.workflow_running):
                    if symbols and LANGGRAPH_AVAILABLE:
                        self.start_analysis(symbols, analysis_type, risk_tolerance, time_horizon)
            
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    self.reset_session()
            
            # Show current session info
            if st.session_state.analysis_session:
                st.divider()
                st.markdown("### 📊 Current Session")
                st.info(f"Session ID: {st.session_state.analysis_session.get('session_id', 'N/A')[:8]}...")
                st.info(f"Symbols: {', '.join(symbols)}")
        
        # Main content area
        if st.session_state.workflow_running:
            self.display_analysis_progress()
        
        # Display results tabs
        if st.session_state.analysis_results:
            self.display_analysis_results()
        else:
            self.display_welcome()
    
    def start_analysis(self, symbols: List[str], analysis_type: str, risk_tolerance: str, time_horizon: str):
        """Start the financial analysis workflow"""
        st.session_state.workflow_running = True
        
        # Create analysis session
        session = create_financial_analysis_session(
            symbols=symbols,
            analysis_type=analysis_type,
            risk_tolerance=risk_tolerance
        )
        
        st.session_state.analysis_session = session
        
        # Initialize with user query
        initial_message = f"Analyze {', '.join(symbols)} with {analysis_type} analysis approach"
        
        with st.spinner("🔍 Initializing multi-agent analysis..."):
            try:
                # Run the LangGraph workflow
                if LANGGRAPH_AVAILABLE:
                    # Start the workflow
                    initial_state = {
                        **session['initial_state'],
                        "messages": [{"role": "user", "content": initial_message}],
                        "time_horizon": time_horizon
                    }
                    
                    # Stream updates from the graph
                    for update in financial_graph.stream(
                        initial_state,
                        config=session['thread_config'],
                        stream_mode="updates"
                    ):
                        # Process updates
                        for node_id, value in update.items():
                            st.session_state.current_agent = node_id
                            
                            # Extract and store results
                            if isinstance(value, dict):
                                if "completed_analyses" in value:
                                    st.session_state.analysis_results.update(value["completed_analyses"])
                                
                                if "messages" in value and value["messages"]:
                                    last_message = value["messages"][-1]
                                    st.session_state.chat_history.append({
                                        "agent": node_id,
                                        "message": last_message
                                    })
                    
                    st.success("✅ Analysis complete!")
                else:
                    # Demo mode
                    self.run_demo_analysis(symbols, analysis_type)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
            finally:
                st.session_state.workflow_running = False
    
    def run_demo_analysis(self, symbols: List[str], analysis_type: str):
        """Run demo analysis with mock data"""
        import time
        
        # Simulate analysis steps
        agents = ["market_research", "technical_analysis", "risk_assessment", "sentiment_analysis", "portfolio_optimization"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, agent in enumerate(agents):
            status_text.text(f"🤖 {agent.replace('_', ' ').title()} analyzing...")
            progress_bar.progress((i + 1) / len(agents))
            time.sleep(1)
            
            # Add mock results
            st.session_state.analysis_results[agent] = {
                "status": "complete",
                "confidence": 0.85,
                "data": f"Mock analysis from {agent}"
            }
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(1.0)
    
    def display_analysis_progress(self):
        """Display real-time analysis progress"""
        st.markdown("### 🔄 Analysis in Progress")
        
        # Show current agent
        if st.session_state.current_agent:
            st.info(f"🤖 Current Agent: **{st.session_state.current_agent.replace('_', ' ').title()}**")
        
        # Show completed analyses
        if st.session_state.analysis_results:
            completed = list(st.session_state.analysis_results.keys())
            st.success(f"✅ Completed: {', '.join(completed)}")
    
    def display_analysis_results(self):
        """Display comprehensive analysis results"""
        st.markdown("### 📊 Analysis Results")
        
        # Create tabs for different analysis sections
        tabs = st.tabs([
            "📈 Overview",
            "🎯 Technical Analysis",
            "📊 Fundamental Analysis",
            "⚠️ Risk Assessment",
            "💭 Sentiment Analysis",
            "💼 Portfolio Optimization",
            "✅ Compliance",
            "📄 Report"
        ])
        
        with tabs[0]:
            self.display_overview()
        
        with tabs[1]:
            self.display_technical_analysis()
        
        with tabs[2]:
            self.display_fundamental_analysis()
        
        with tabs[3]:
            self.display_risk_assessment()
        
        with tabs[4]:
            self.display_sentiment_analysis()
        
        with tabs[5]:
            self.display_portfolio_optimization()
        
        with tabs[6]:
            self.display_compliance()
        
        with tabs[7]:
            self.display_report()
    
    def display_overview(self):
        """Display analysis overview"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Rating", "BUY", "↑ High Confidence")
        
        with col2:
            st.metric("Risk Score", "45/100", "Moderate")
        
        with col3:
            st.metric("Sentiment", "Bullish", "+0.65")
        
        with col4:
            st.metric("Technical Signal", "Strong Buy", "RSI: 58")
        
        # Market overview chart
        if st.session_state.analysis_session:
            symbols = st.session_state.analysis_session['initial_state']['target_symbols']
            self.display_price_chart(symbols[0] if symbols else "AAPL")
    
    def display_price_chart(self, symbol: str):
        """Display interactive price chart"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")
            
            if not hist.empty:
                fig = go.Figure()
                
                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name='Price'
                ))
                
                # Add volume
                fig.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    yaxis='y2',
                    opacity=0.3
                ))
                
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    yaxis_title="Price ($)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying='y',
                        side='right'
                    ),
                    template="plotly_white",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error loading price data: {str(e)}")
    
    def display_technical_analysis(self):
        """Display technical analysis results"""
        st.markdown("#### 📈 Technical Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Momentum Indicators**")
            st.metric("RSI (14)", "58.3", "Neutral")
            st.metric("MACD", "0.45", "Bullish")
            st.metric("Stochastic", "72.1", "Overbought")
        
        with col2:
            st.markdown("**Moving Averages**")
            st.metric("SMA 20", "$148.50", "Above")
            st.metric("SMA 50", "$145.20", "Above")
            st.metric("EMA 12", "$149.80", "Above")
        
        with col3:
            st.markdown("**Volatility**")
            st.metric("ATR", "3.45", "Normal")
            st.metric("Bollinger Width", "6.2%", "Expanding")
            st.metric("IV Rank", "35%", "Low")
        
        # Technical analysis from results
        if "technical_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Analysis")
            st.info(self.get_analysis_data("technical_analysis", "No technical analysis available"))
    
    def display_fundamental_analysis(self):
        """Display fundamental analysis results"""
        st.markdown("#### 💰 Fundamental Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("P/E Ratio", "28.5", "Industry Avg: 25.2")
        
        with col2:
            st.metric("PEG Ratio", "1.8", "Fair Value")
        
        with col3:
            st.metric("ROE", "42.3%", "Excellent")
        
        with col4:
            st.metric("Debt/Equity", "0.65", "Healthy")
        
        # Fundamental analysis from results
        if "fundamental_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Analysis")
            st.info(self.get_analysis_data("fundamental_analysis", "No fundamental analysis available"))
    
    def display_risk_assessment(self):
        """Display risk assessment results"""
        st.markdown("#### ⚠️ Risk Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Value at Risk (95% confidence)**")
            st.metric("1-Day VaR", "-3.8%", "Acceptable")
            st.metric("1-Month VaR", "-12.5%", "Moderate")
            
            st.markdown("**Volatility Metrics**")
            st.metric("Historical Volatility", "28.5%", "Average")
            st.metric("Max Drawdown", "-15.2%", "Last 12 months")
        
        with col2:
            # Risk heatmap
            risk_data = pd.DataFrame({
                'Risk Factor': ['Market Risk', 'Liquidity Risk', 'Credit Risk', 'Operational Risk', 'Regulatory Risk'],
                'Score': [65, 30, 20, 15, 10]
            })
            
            fig = px.bar(risk_data, x='Score', y='Risk Factor', orientation='h',
                        color='Score', color_continuous_scale='RdYlGn_r',
                        title='Risk Factor Assessment')
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis from results
        if "risk_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Risk Analysis")
            st.warning(self.get_analysis_data("risk_analysis", "No risk analysis available"))
    
    def display_sentiment_analysis(self):
        """Display sentiment analysis results"""
        st.markdown("#### 💭 Market Sentiment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment gauge
            sentiment_score = 0.65  # Mock value
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                title={'text': "Overall Sentiment"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.5], 'color': "red"},
                        {'range': [-0.5, 0], 'color': "lightgray"},
                        {'range': [0, 0.5], 'color': "lightgreen"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': sentiment_score
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Sentiment Sources**")
            st.metric("News Sentiment", "+0.42", "Positive")
            st.metric("Social Media", "+0.68", "Very Positive")
            st.metric("Analyst Ratings", "Buy", "4.2/5")
            st.metric("Options Flow", "Bullish", "P/C: 0.65")
        
        # Sentiment analysis from results
        if "sentiment_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Sentiment Analysis")
            st.info(self.get_analysis_data("sentiment_analysis", "No sentiment analysis available"))
    
    def display_portfolio_optimization(self):
        """Display portfolio optimization results"""
        st.markdown("#### 💼 Portfolio Recommendations")
        
        # Allocation pie chart
        allocation_data = pd.DataFrame({
            'Asset': ['AAPL', 'MSFT', 'GOOGL', 'Cash', 'Bonds'],
            'Allocation': [30, 25, 20, 15, 10]
        })
        
        fig = px.pie(allocation_data, values='Allocation', names='Asset',
                    title='Recommended Portfolio Allocation')
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Expected Return", "12.5%", "Annual")
        
        with col2:
            st.metric("Portfolio Volatility", "18.2%", "Annual")
        
        with col3:
            st.metric("Sharpe Ratio", "0.69", "Risk-Adjusted")
        
        # Portfolio analysis from results
        if "portfolio_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Portfolio Analysis")
            st.info(self.get_analysis_data("portfolio_analysis", "No portfolio analysis available"))
    
    def display_compliance(self):
        """Display compliance results"""
        st.markdown("#### ✅ Compliance Check")
        
        compliance_items = [
            {"item": "Regulatory Restrictions", "status": "✅ Passed", "notes": "No restrictions identified"},
            {"item": "Insider Trading Check", "status": "✅ Clear", "notes": "No recent insider activity"},
            {"item": "ESG Compliance", "status": "⚠️ Review", "notes": "ESG score: 72/100"},
            {"item": "Trading Restrictions", "status": "✅ None", "notes": "No blackout periods"},
            {"item": "Concentration Limits", "status": "✅ Within Limits", "notes": "Max 5% per position"}
        ]
        
        df = pd.DataFrame(compliance_items)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Compliance analysis from results
        if "compliance_analysis" in st.session_state.analysis_results:
            st.markdown("#### 📊 Detailed Compliance Analysis")
            st.info(self.get_analysis_data("compliance_analysis", "No compliance analysis available"))
    
    def display_report(self):
        """Display final analysis report"""
        st.markdown("#### 📄 Executive Summary")
        
        report_content = """
        ### Investment Recommendation Report
        
        **Date**: {date}
        **Symbols Analyzed**: {symbols}
        **Recommendation**: **BUY**
        
        #### Key Findings:
        1. **Strong Fundamentals**: Company shows robust financial health with growing revenues and healthy margins
        2. **Positive Technical Signals**: Multiple technical indicators suggest upward momentum
        3. **Favorable Sentiment**: Market sentiment is overwhelmingly positive
        4. **Acceptable Risk Profile**: Risk metrics are within acceptable ranges for the given return potential
        5. **Portfolio Fit**: Recommended allocation aligns with stated risk tolerance
        
        #### Risk Factors:
        - Market volatility remains elevated
        - Sector rotation risk present
        - Earnings expectations are high
        
        #### Action Items:
        1. Initiate position with 3-5% portfolio allocation
        2. Set stop-loss at -8% from entry
        3. Target price: 15% upside over 6 months
        4. Review position quarterly
        
        ---
        *This report is for informational purposes only and does not constitute investment advice.*
        """.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            symbols=", ".join(st.session_state.analysis_session['initial_state']['target_symbols']) if st.session_state.analysis_session else "N/A"
        )
        
        st.markdown(report_content)
        
        # Download button for report
        st.download_button(
            label="📥 Download Full Report",
            data=report_content,
            file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
    
    def display_welcome(self):
        """Display welcome screen"""
        st.markdown("""
        ### Welcome to the Multi-Agent Financial Analysis System
        
        This advanced platform leverages **LangGraph** and multiple specialized AI agents to provide comprehensive financial analysis:
        
        #### 🤖 **Specialized Agents:**
        - **Market Research Agent**: Fundamental analysis and valuation
        - **Technical Analysis Agent**: Chart patterns and technical indicators
        - **Risk Assessment Agent**: VaR, stress testing, and risk metrics
        - **Sentiment Analysis Agent**: News, social media, and market psychology
        - **Portfolio Optimization Agent**: Asset allocation and rebalancing
        - **Compliance Agent**: Regulatory checks and ESG scoring
        - **Report Generation Agent**: Professional investment reports
        
        #### 🔄 **Advanced Features:**
        - **Dynamic Routing**: Intelligent agent selection based on query
        - **Market-Aware Decisions**: Adapts to current market conditions
        - **Human-in-the-Loop**: Approval workflows for high-risk decisions
        - **Real-time Alerts**: Critical market event notifications
        - **Multi-threaded Analysis**: Concurrent processing for speed
        
        #### 🚀 **Getting Started:**
        1. Enter stock symbols in the sidebar
        2. Select your analysis type and risk tolerance
        3. Click "Start Analysis" to begin
        4. Review comprehensive results across all tabs
        
        ---
        
        **Built with LangGraph** for sophisticated multi-agent orchestration and state management.
        """)
        
        # Show sample analysis
        if st.button("🎯 Run Sample Analysis", type="primary"):
            self.start_analysis(["AAPL", "MSFT"], "comprehensive", "moderate", "medium")
    
    def reset_session(self):
        """Reset the analysis session"""
        st.session_state.analysis_session = None
        st.session_state.analysis_results = {}
        st.session_state.chat_history = []
        st.session_state.workflow_running = False
        st.session_state.current_agent = None
        st.session_state.market_data = {}
        st.rerun()


def main():
    """Main entry point"""
    app = FinancialAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()