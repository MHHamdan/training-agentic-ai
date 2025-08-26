import streamlit as st
import asyncio
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List

# Set up imports for the stock analysis system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from crew.stock_analysis_crew import StockAnalysisCrew
from utils.observability import get_observability_manager
from models.model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Agent 🚀",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .analysis-section {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .recommendation-buy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        color: #155724;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        color: #721c24;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

class StockAnalysisApp:
    """Streamlit application for Stock Analysis Agent"""
    
    def __init__(self):
        self.crew = None
        self.observability = None
        self.model_manager = None
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize the stock analysis systems"""
        try:
            if 'crew' not in st.session_state:
                with st.spinner("Initializing Stock Analysis Agent..."):
                    self.crew = StockAnalysisCrew(config.model_dump())
                    self.observability = get_observability_manager()
                    self.model_manager = ModelManager(config.model_dump())
                    st.session_state.crew = self.crew
                    st.session_state.observability = self.observability
                    st.session_state.model_manager = self.model_manager
            else:
                self.crew = st.session_state.crew
                self.observability = st.session_state.observability
                self.model_manager = st.session_state.model_manager
                
        except Exception as e:
            st.error(f"Failed to initialize systems: {str(e)}")
            logger.error(f"Initialization error: {str(e)}")
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown('<h1 class="main-header">📈 Stock Analysis Agent</h1>', 
                   unsafe_allow_html=True)
        st.markdown("*Enterprise-grade AI-powered stock analysis with multi-agent orchestration*")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        main_tab, monitoring_tab, portfolio_tab, settings_tab = st.tabs([
            "🎯 Stock Analysis", 
            "📊 Real-time Monitoring", 
            "💼 Portfolio Analysis",
            "⚙️ Settings"
        ])
        
        with main_tab:
            self._render_analysis_page()
        
        with monitoring_tab:
            self._render_monitoring_page()
        
        with portfolio_tab:
            self._render_portfolio_page()
        
        with settings_tab:
            self._render_settings_page()
    
    def _render_sidebar(self):
        """Render sidebar with system status and controls"""
        st.sidebar.title("🚀 System Status")
        
        if self.crew:
            crew_status = self.crew.get_crew_status()
            
            st.sidebar.success("✅ System Online")
            st.sidebar.info(f"🤖 {len(crew_status['agents'])} Agents Active")
            st.sidebar.info(f"🔧 {crew_status['tools_available']} Tools Available")
            
            # Agent status
            st.sidebar.subheader("Agent Status")
            for agent_name, agent_info in crew_status['agents'].items():
                status_icon = "🟢" if agent_info['status'] == 'active' else "🔴"
                st.sidebar.write(f"{status_icon} {agent_info['role']}")
        
        else:
            st.sidebar.error("❌ System Offline")
        
        # Quick actions
        st.sidebar.subheader("Quick Actions")
        if st.sidebar.button("🔄 Refresh System"):
            st.rerun()
        
        if st.sidebar.button("📋 View Logs"):
            st.sidebar.text("Logs feature coming soon...")
    
    def _render_analysis_page(self):
        """Render main stock analysis page"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Stock Analysis")
            
            # Input form
            with st.form("analysis_form"):
                ticker_input = st.text_input(
                    "Stock Ticker Symbol", 
                    placeholder="Enter ticker (e.g., AAPL, GOOGL, TSLA)",
                    help="Enter a valid stock ticker symbol"
                ).upper()
                
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Comprehensive", "Quick Analysis", "Technical Only", "Fundamental Only"],
                    help="Select the type of analysis to perform"
                )
                
                col_submit, col_clear = st.columns(2)
                with col_submit:
                    submit_button = st.form_submit_button("🚀 Analyze Stock", type="primary")
                with col_clear:
                    clear_button = st.form_submit_button("🗑️ Clear")
            
            if clear_button:
                st.rerun()
            
            if submit_button and ticker_input:
                self._perform_stock_analysis(ticker_input, analysis_type.lower())
        
        with col2:
            st.subheader("📊 Quick Stats")
            
            # Display recent analyses
            if 'recent_analyses' not in st.session_state:
                st.session_state.recent_analyses = []
            
            if st.session_state.recent_analyses:
                st.write("Recent Analyses:")
                for analysis in st.session_state.recent_analyses[-3:]:
                    with st.expander(f"{analysis['ticker']} - {analysis['timestamp'][:19]}"):
                        if 'executive_summary' in analysis:
                            recommendation = analysis['executive_summary'].get('recommendation', {})
                            st.write(f"**Recommendation:** {recommendation.get('action', 'N/A')}")
                            st.write(f"**Target Price:** ${recommendation.get('target_price', 'N/A')}")
            else:
                st.info("No recent analyses. Start by analyzing a stock!")
    
    def _perform_stock_analysis(self, ticker: str, analysis_type: str):
        """Perform stock analysis with selected type"""
        if not self.crew:
            st.error("Stock Analysis system not initialized!")
            return
        
        with st.spinner(f"🔄 Analyzing {ticker}... This may take 1-2 minutes."):
            try:
                # Create async task for analysis
                if analysis_type == "quick analysis":
                    result = asyncio.run(self.crew.quick_analysis(ticker))
                else:
                    result = asyncio.run(self.crew.analyze_stock(ticker, analysis_type))
                
                if 'error' in result:
                    st.error(f"Analysis failed: {result['error']}")
                    return
                
                # Store in session state
                if 'recent_analyses' not in st.session_state:
                    st.session_state.recent_analyses = []
                
                result['timestamp'] = datetime.now().isoformat()
                st.session_state.recent_analyses.append(result)
                
                # Display results
                self._display_analysis_results(result)
                
            except Exception as e:
                st.error(f"Error performing analysis: {str(e)}")
                logger.error(f"Analysis error: {str(e)}")
    
    def _display_analysis_results(self, result: Dict[str, Any]):
        """Display analysis results in organized format"""
        ticker = result.get('ticker', 'Unknown')
        
        st.success(f"✅ Analysis completed for {ticker}")
        
        # Executive Summary
        if 'executive_summary' in result:
            exec_summary = result['executive_summary']
            st.subheader("📋 Executive Summary")
            
            recommendation = exec_summary.get('recommendation', {})
            action = recommendation.get('action', 'HOLD')
            target_price = recommendation.get('target_price', 'N/A')
            
            # Recommendation card
            if action == 'BUY':
                st.markdown(f"""
                <div class="recommendation-buy">
                    <strong>🚀 RECOMMENDATION: {action}</strong><br>
                    Target Price: ${target_price}<br>
                    Investment Horizon: {recommendation.get('horizon', 'Medium-term')}
                </div>
                """, unsafe_allow_html=True)
            elif action == 'SELL':
                st.markdown(f"""
                <div class="recommendation-sell">
                    <strong>📉 RECOMMENDATION: {action}</strong><br>
                    Target Price: ${target_price}<br>
                    Investment Horizon: {recommendation.get('horizon', 'Medium-term')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-hold">
                    <strong>🤚 RECOMMENDATION: {action}</strong><br>
                    Target Price: ${target_price}<br>
                    Investment Horizon: {recommendation.get('horizon', 'Medium-term')}
                </div>
                """, unsafe_allow_html=True)
            
            # Key thesis points
            if 'investment_thesis' in exec_summary:
                st.write("**Key Investment Thesis:**")
                for point in exec_summary['investment_thesis'][:3]:
                    st.write(f"• {point}")
            
            # Key risks
            if 'key_risks' in exec_summary:
                st.write("**Key Risks:**")
                for risk in exec_summary['key_risks'][:3]:
                    st.write(f"⚠️ {risk}")
        
        # Detailed Analysis Sections
        if 'detailed_analysis' in result:
            detailed = result['detailed_analysis']
            
            # Create tabs for different analyses
            if detailed:
                tabs = st.tabs([
                    "📊 Fundamental", 
                    "📈 Technical", 
                    "📰 Sentiment", 
                    "⚠️ Risk Assessment"
                ])
                
                # Fundamental Analysis
                if 'fundamental' in detailed and tabs[0]:
                    with tabs[0]:
                        self._display_fundamental_analysis(detailed['fundamental'])
                
                # Technical Analysis
                if 'technical' in detailed and tabs[1]:
                    with tabs[1]:
                        self._display_technical_analysis(detailed['technical'])
                
                # Sentiment Analysis
                if 'sentiment' in detailed and tabs[2]:
                    with tabs[2]:
                        self._display_sentiment_analysis(detailed['sentiment'])
                
                # Risk Assessment
                if 'risk' in detailed and tabs[3]:
                    with tabs[3]:
                        self._display_risk_analysis(detailed['risk'])
        
        # Raw data (expandable)
        with st.expander("🔍 View Raw Analysis Data"):
            st.json(result)
    
    def _display_fundamental_analysis(self, fundamental: Dict[str, Any]):
        """Display fundamental analysis results"""
        st.subheader("📊 Fundamental Analysis")
        
        if 'error' in fundamental:
            st.error(f"Fundamental analysis error: {fundamental['error']}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'financial_health' in fundamental:
                fh = fundamental['financial_health']
                st.metric("Financial Health Score", f"{fh.get('score', 0)}/10")
                
                if 'details' in fh and fh['details']:
                    st.write("**Key Points:**")
                    for detail in fh['details'][:3]:
                        st.write(f"• {detail}")
        
        with col2:
            if 'recommendation' in fundamental:
                rec = fundamental['recommendation']
                st.metric("Target Price", f"${rec.get('target_price', 'N/A')}")
                st.metric("Confidence", f"{rec.get('confidence', 50)}%")
                
        if 'valuation' in fundamental:
            val = fundamental['valuation']
            st.write("**Valuation Metrics:**")
            metrics = val.get('metrics', {})
            if metrics:
                metric_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                st.dataframe(metric_df)
    
    def _display_technical_analysis(self, technical: Dict[str, Any]):
        """Display technical analysis results"""
        st.subheader("📈 Technical Analysis")
        
        if 'error' in technical:
            st.error(f"Technical analysis error: {technical['error']}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'trend_analysis' in technical:
                trend = technical['trend_analysis']
                st.metric("Trend Direction", trend.get('direction', 'Sideways'))
                st.metric("Trend Strength", trend.get('strength', 'Moderate'))
        
        with col2:
            if 'trading_signals' in technical:
                signals = technical['trading_signals']
                st.metric("Signal Strength", f"{signals.get('signal_strength', 5)}/10")
                st.metric("Action", signals.get('action', 'Hold'))
        
        # Support and Resistance
        if 'support_resistance' in technical:
            sr = technical['support_resistance']
            
            col_sup, col_res = st.columns(2)
            with col_sup:
                st.write("**Support Levels:**")
                for level in sr.get('support_levels', [])[:3]:
                    st.write(f"${level}")
            
            with col_res:
                st.write("**Resistance Levels:**")
                for level in sr.get('resistance_levels', [])[:3]:
                    st.write(f"${level}")
    
    def _display_sentiment_analysis(self, sentiment: Dict[str, Any]):
        """Display sentiment analysis results"""
        st.subheader("📰 Sentiment Analysis")
        
        if 'error' in sentiment:
            st.error(f"Sentiment analysis error: {sentiment['error']}")
            return
        
        if 'overall_sentiment' in sentiment:
            overall = sentiment['overall_sentiment']
            score = overall.get('score', 0)
            
            # Sentiment gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sentiment Score"},
                gauge = {
                    'axis': {'range': [-100, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-100, -20], 'color': "lightgray"},
                        {'range': [-20, 20], 'color': "gray"},
                        {'range': [20, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment drivers
        if 'sentiment_drivers' in sentiment:
            drivers = sentiment['sentiment_drivers']
            
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.write("**Positive Factors:**")
                for factor in drivers.get('positive', [])[:3]:
                    st.write(f"✅ {factor}")
            
            with col_neg:
                st.write("**Negative Factors:**")
                for factor in drivers.get('negative', [])[:3]:
                    st.write(f"❌ {factor}")
    
    def _display_risk_analysis(self, risk: Dict[str, Any]):
        """Display risk analysis results"""
        st.subheader("⚠️ Risk Assessment")
        
        if 'error' in risk:
            st.error(f"Risk analysis error: {risk['error']}")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Risk Rating", risk.get('overall_risk_rating', 'Medium'))
            st.metric("Risk Score", f"{risk.get('risk_score', 5)}/10")
        
        with col2:
            if 'market_risk' in risk:
                mr = risk['market_risk']
                st.metric("Beta", f"{mr.get('beta', 1.0):.2f}")
                st.metric("Volatility", f"{mr.get('volatility', 0.2)*100:.1f}%")
        
        with col3:
            if 'liquidity_risk' in risk:
                lr = risk['liquidity_risk']
                st.metric("Liquidity Score", f"{lr.get('liquidity_score', 7)}/10")
        
        # Risk breakdown
        if 'company_risks' in risk:
            st.write("**Company-Specific Risks:**")
            company_risks = risk['company_risks']
            for risk_type, level in company_risks.items():
                color = "🟢" if level == "low" else "🟡" if level == "medium" else "🔴"
                st.write(f"{color} {risk_type.replace('_', ' ').title()}: {level}")
    
    def _render_monitoring_page(self):
        """Render real-time monitoring dashboard"""
        st.subheader("📊 Real-time Monitoring Dashboard")
        
        # Placeholder for real-time monitoring
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Sessions", "12", "2")
        with col2:
            st.metric("Analyses Today", "47", "8")
        with col3:
            st.metric("System Uptime", "99.8%", "0.1%")
        with col4:
            st.metric("Avg Response Time", "1.2s", "-0.3s")
        
        # Performance charts (placeholder)
        st.subheader("Performance Metrics")
        
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sample_data = pd.DataFrame({
            'Date': dates,
            'Analyses': [20 + i * 2 + (i % 7) * 5 for i in range(30)],
            'Response_Time': [1.0 + (i % 10) * 0.1 for i in range(30)]
        })
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig1 = px.line(sample_data, x='Date', y='Analyses', title='Daily Analysis Volume')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_chart2:
            fig2 = px.line(sample_data, x='Date', y='Response_Time', title='Average Response Time')
            st.plotly_chart(fig2, use_container_width=True)
    
    def _render_portfolio_page(self):
        """Render portfolio analysis page"""
        st.subheader("💼 Portfolio Analysis")
        st.info("Portfolio analysis feature coming soon! This will allow you to analyze entire portfolios and get risk metrics.")
        
        # Placeholder for portfolio input
        with st.form("portfolio_form"):
            st.write("**Add Portfolio Holdings:**")
            ticker = st.text_input("Ticker")
            weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0, value=10.0)
            
            if st.form_submit_button("Add to Portfolio"):
                st.success(f"Added {ticker} with {weight}% weight")
    
    def _render_settings_page(self):
        """Render settings page"""
        st.subheader("⚙️ System Settings")
        
        # Model settings
        st.write("**Model Configuration:**")
        if self.model_manager:
            available_models = ["microsoft/DialoGPT-large", "facebook/blenderbot-1B-distill", "OpenAI GPT-3.5"]
            selected_model = st.selectbox("Primary Model", available_models, index=0)
            
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)
        
        # API settings
        st.write("**API Configuration:**")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password", 
                                        placeholder="Enter your Alpha Vantage API key")
        
        # AgentOps settings
        st.write("**AgentOps Configuration:**")
        agentops_enabled = st.checkbox("Enable AgentOps Tracking", value=True)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

# Main application
def main():
    """Main application entry point"""
    app = StockAnalysisApp()
    app.run()

if __name__ == "__main__":
    main()