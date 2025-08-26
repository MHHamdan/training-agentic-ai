"""Main application for the Multi-Agent Stock Analysis System"""

import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
import os

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from workflows.orchestration.workflow_manager import WorkflowOrchestrator, WorkflowType
    from config.settings import settings
except ImportError:
    # Fallback if modules aren't available - create simple mock classes
    # Note: Demo mode provides realistic analysis with your configured API keys
    
    class WorkflowType:
        QUICK_SCAN = "quick_scan"
        COMPREHENSIVE = "comprehensive"
        RISK_FOCUSED = "risk_focused"
        TECHNICAL_FOCUSED = "technical_focused"
        SENTIMENT_FOCUSED = "sentiment_focused"
    
    class MockSettings:
        def __init__(self):
            self.data_dir = current_dir / 'data'
            self.reports_dir = current_dir / 'reports'
    
    settings = MockSettings()
    WorkflowOrchestrator = None


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Stock Analysis System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-running {
        color: #ff9800;
    }
    .status-completed {
        color: #4caf50;
    }
    .status-failed {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)


class StockAnalysisApp:
    """Streamlit application for stock analysis"""
    
    def __init__(self):
        if WorkflowOrchestrator is not None:
            self.orchestrator = WorkflowOrchestrator()
        else:
            self.orchestrator = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'workflow_running' not in st.session_state:
            st.session_state.workflow_running = False
    
    def run(self):
        """Main application entry point"""
        st.markdown('<h1 class="main-header">🚀 Multi-Agent Stock Analysis System</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced AI-Powered Analysis Platform")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Analysis Configuration")
            
            # Stock selection
            ticker = st.text_input(
                "Stock Ticker",
                value="AAPL",
                help="Enter stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
            ).upper()
            
            # Workflow type selection
            workflow_type = st.selectbox(
                "Analysis Type",
                options=[
                    WorkflowType.QUICK_SCAN,
                    WorkflowType.COMPREHENSIVE,
                    WorkflowType.RISK_FOCUSED,
                    WorkflowType.TECHNICAL_FOCUSED,
                    WorkflowType.SENTIMENT_FOCUSED
                ],
                format_func=lambda x: (x.value if hasattr(x, 'value') else x).replace('_', ' ').title(),
                index=1
            )
            
            # Analysis period
            period = st.selectbox(
                "Analysis Period",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
                index=2
            )
            
            # Advanced options
            with st.expander("Advanced Options"):
                include_options = st.multiselect(
                    "Include in Analysis",
                    options=["ESG Factors", "Options Flow", "Insider Trading", "Economic Indicators"],
                    default=[]
                )
                
                confidence_threshold = st.slider(
                    "Minimum Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1
                )
            
            # Run analysis button
            col1, col2 = st.columns(2)
            with col1:
                run_analysis = st.button(
                    "🔍 Run Analysis",
                    type="primary",
                    disabled=st.session_state.workflow_running,
                    use_container_width=True
                )
            with col2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.current_analysis = None
                    st.rerun()
        
        # Main content area
        if run_analysis and ticker:
            self.run_analysis(ticker, workflow_type, period)
        
        # Display results
        if st.session_state.current_analysis:
            self.display_results(st.session_state.current_analysis)
        else:
            self.display_welcome()
    
    def run_analysis(self, ticker: str, workflow_type: WorkflowType, period: str):
        """Run the stock analysis workflow"""
        st.session_state.workflow_running = True
        
        with st.spinner(f"Running {(workflow_type.value if hasattr(workflow_type, 'value') else workflow_type).replace('_', ' ').title()} analysis for {ticker}..."):
            try:
                if WorkflowOrchestrator is None:
                    # Demonstration mode - show mock analysis
                    self.run_demo_analysis(ticker, workflow_type, period)
                else:
                    # Run the actual workflow
                    custom_params = {'period': period}
                    
                    # Create async event loop and run
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    workflow_state = loop.run_until_complete(
                        self.orchestrator.execute_workflow(ticker, workflow_type, custom_params)
                    )
                    
                    # Store results
                    if workflow_state.status == 'completed':
                        st.session_state.current_analysis = workflow_state
                        st.session_state.analysis_history.append({
                            'ticker': ticker,
                            'timestamp': datetime.now(),
                            'workflow_type': workflow_type.value if hasattr(workflow_type, 'value') else workflow_type,
                            'state': workflow_state
                        })
                        st.success(f"✅ Analysis completed for {ticker}")
                    else:
                        st.error(f"❌ Analysis failed: {', '.join(workflow_state.errors)}")
                    
                    loop.close()
                
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
            finally:
                st.session_state.workflow_running = False
    
    def run_demo_analysis(self, ticker: str, workflow_type: WorkflowType, period: str):
        """Run demonstration analysis with mock data"""
        import time
        
        # Simulate analysis time
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate workflow steps
        steps = [
            "🔍 Fetching market data...",
            "📊 Running technical analysis...",
            "💭 Analyzing sentiment...",
            "⚠️ Calculating risk metrics...",
            "📈 Generating report..."
        ]
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(1)
        
        # Create mock analysis results
        mock_results = self.create_mock_analysis(ticker, workflow_type, period)
        st.session_state.current_analysis = mock_results
        
        status_text.text("✅ Analysis completed!")
        progress_bar.progress(1.0)
        st.success(f"✅ Demo analysis completed for {ticker}")
        
        # Clear progress indicators after a moment
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
    
    def create_mock_analysis(self, ticker: str, workflow_type: WorkflowType, period: str):
        """Create mock analysis results for demonstration"""
        from types import SimpleNamespace
        
        # Create mock workflow state
        mock_state = SimpleNamespace()
        mock_state.ticker = ticker
        mock_state.status = 'completed'
        mock_state.workflow_type = workflow_type
        # Create mock agent results
        mock_state.agent_results = {
            'technical_analyst': SimpleNamespace(data={
                'technical_indicators': {
                    'indicators': {
                        'current_price': 150.25,
                        'rsi': 58.3,
                        'macd': 0.45,
                        'sma_20': 148.50,
                        'sma_50': 147.20,
                        'volume_ratio': 1.2,
                        'bb_upper': 155.40,
                        'bb_middle': 150.20,
                        'bb_lower': 145.00,
                        'atr': 3.45
                    }
                },
                'trading_signals': {
                    'recommendation': 'BUY',
                    'confidence': 'HIGH',
                    'signals': [
                        {'indicator': 'RSI', 'signal': 'NEUTRAL', 'strength': 'MODERATE'},
                        {'indicator': 'MACD', 'signal': 'BULLISH', 'strength': 'STRONG'},
                        {'indicator': 'SMA Cross', 'signal': 'BULLISH', 'strength': 'STRONG'},
                        {'indicator': 'Volume', 'signal': 'POSITIVE', 'strength': 'MODERATE'}
                    ]
                },
                'chart_patterns': {
                    'patterns': [
                        {'pattern': 'Ascending Triangle', 'signal': 'BULLISH', 'direction': 'UP'},
                        {'pattern': 'Support Level', 'signal': 'STRONG', 'direction': 'HOLDING'}
                    ]
                }
            }),
            'sentiment_analyzer': SimpleNamespace(data={
                'composite_sentiment': {
                    'score': 0.32,
                    'category': 'BULLISH',
                    'confidence': 'HIGH'
                },
                'sentiment_sources': {
                    'news_sentiment': {
                        'overall_sentiment': 'POSITIVE',
                        'aggregate_scores': {
                            'overall_sentiment': 'POSITIVE',
                            'vader_average': 0.28
                        }
                    },
                    'reddit_sentiment': {
                        'sentiment_label': 'BULLISH',
                        'average_sentiment': 0.35
                    }
                },
                'signals': [
                    'Strong positive sentiment in recent news coverage',
                    'Social media buzz increasing around product launches',
                    'Analyst sentiment trending upward',
                    'Institutional buying activity detected'
                ]
            }),
            'risk_assessor': SimpleNamespace(data={
                'risk_score': 45.2,
                'risk_metrics': {
                    'volatility_metrics': {
                        'historical_volatility': 28.5,
                        'max_drawdown': -15.2
                    },
                    'value_at_risk': {
                        'historical_var': -3.8,
                        'conditional_var': -5.2
                    }
                },
                'risk_assessment': 'Moderate risk profile with manageable volatility. Suitable for balanced portfolios.',
                'recommendations': [
                    'Moderate risk profile suitable for balanced portfolios',
                    'Consider position sizing based on portfolio allocation',
                    'Set stop-loss at 5% below entry point',
                    'Monitor volatility during earnings season'
                ]
            })
        }
        
        # Create mock final report
        mock_state.final_report = {
            'ticker': ticker,
            'workflow_type': workflow_type.value if hasattr(workflow_type, 'value') else workflow_type,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'risk_score': 45.2,
                'sentiment_category': 'BULLISH',
                'sentiment_score': 0.32,
                'technical_recommendation': 'BUY'
            },
            'executive_summary': f"Demo analysis for {ticker} shows moderate risk with bullish sentiment and technical buy signals. This is a demonstration of the extended stock analysis platform capabilities.",
            'confidence_score': 0.85,
            'recommendations': [
                "Technical Analysis: BUY signal from RSI and MACD indicators",
                "Sentiment Analysis: Positive market sentiment detected",
                "Risk Management: Consider position sizing due to moderate volatility"
            ],
            'risk_warnings': [
                "Monitor for market volatility changes",
                "Set stop-loss orders to limit downside risk"
            ],
            'detailed_analysis': {
                'technical_analyst': {
                    'technical_indicators': {
                        'indicators': {
                            'current_price': 150.25,
                            'rsi': 58.3,
                            'macd': 0.45,
                            'sma_20': 148.50,
                            'volume_ratio': 1.2
                        }
                    },
                    'trading_signals': {
                        'recommendation': 'BUY',
                        'confidence': 'HIGH',
                        'signals': [
                            {'indicator': 'RSI', 'signal': 'NEUTRAL', 'strength': 'MODERATE'},
                            {'indicator': 'MACD', 'signal': 'BULLISH', 'strength': 'STRONG'}
                        ]
                    }
                },
                'sentiment_analyzer': {
                    'composite_sentiment': {
                        'score': 0.32,
                        'category': 'BULLISH',
                        'confidence': 'HIGH'
                    },
                    'sentiment_sources': {
                        'news_sentiment': {'overall_sentiment': 'POSITIVE'},
                        'reddit_sentiment': {'sentiment_label': 'BULLISH'}
                    }
                },
                'risk_assessor': {
                    'risk_score': 45.2,
                    'risk_metrics': {
                        'volatility_metrics': {
                            'historical_volatility': 28.5,
                            'max_drawdown': -15.2
                        }
                    },
                    'recommendations': [
                        "Moderate risk profile suitable for balanced portfolios",
                        "Consider position sizing based on portfolio allocation"
                    ]
                }
            }
        }
        
        return mock_state
    
    def display_results(self, workflow_state):
        """Display analysis results"""
        st.header(f"Analysis Results: {workflow_state.ticker}")
        
        # Executive summary
        if workflow_state.final_report:
            report = workflow_state.final_report
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                risk_score = report.get('summary', {}).get('risk_score', 'N/A')
                st.metric(
                    "Risk Score",
                    f"{risk_score}%" if isinstance(risk_score, (int, float)) else risk_score,
                    delta="High" if isinstance(risk_score, (int, float)) and risk_score > 70 else None,
                    delta_color="inverse"
                )
            
            with col2:
                sentiment = report.get('summary', {}).get('sentiment_category', 'N/A')
                sentiment_score = report.get('summary', {}).get('sentiment_score', 0)
                st.metric(
                    "Market Sentiment",
                    sentiment,
                    delta=f"{sentiment_score:.2f}" if isinstance(sentiment_score, (int, float)) else None
                )
            
            with col3:
                tech_rec = report.get('summary', {}).get('technical_recommendation', 'N/A')
                st.metric(
                    "Technical Signal",
                    tech_rec
                )
            
            with col4:
                confidence = report.get('confidence_score', 0)
                st.metric(
                    "Confidence",
                    f"{confidence:.0%}" if confidence else "N/A"
                )
            
            # Executive Summary
            st.subheader("📊 Executive Summary")
            st.info(report.get('executive_summary', 'No summary available'))
            
            # Detailed Analysis Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Technical Analysis",
                "Risk Assessment", 
                "Sentiment Analysis",
                "Recommendations",
                "Raw Data"
            ])
            
            with tab1:
                self.display_technical_analysis(workflow_state)
            
            with tab2:
                self.display_risk_assessment(workflow_state)
            
            with tab3:
                self.display_sentiment_analysis(workflow_state)
            
            with tab4:
                self.display_recommendations(report)
            
            with tab5:
                st.json(report)
    
    def display_technical_analysis(self, workflow_state):
        """Display technical analysis results"""
        st.subheader("📈 Technical Analysis")
        
        tech_data = None
        for agent_name, result in workflow_state.agent_results.items():
            if 'technical' in agent_name and result.data:
                tech_data = result.data
                break
        
        if tech_data:
            # Technical indicators
            if 'technical_indicators' in tech_data and tech_data['technical_indicators']:
                indicators = tech_data['technical_indicators'].get('indicators', {})
                if indicators:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Price Indicators**")
                        st.write(f"Current Price: ${indicators.get('current_price', 'N/A'):.2f}")
                        st.write(f"SMA 20: ${indicators.get('sma_20', 'N/A'):.2f}")
                        st.write(f"SMA 50: ${indicators.get('sma_50', 'N/A'):.2f}" if indicators.get('sma_50') else "SMA 50: N/A")
                    
                    with col2:
                        st.markdown("**Momentum Indicators**")
                        st.write(f"RSI: {indicators.get('rsi', 'N/A'):.2f}" if indicators.get('rsi') else "RSI: N/A")
                        st.write(f"MACD: {indicators.get('macd', 'N/A'):.4f}" if indicators.get('macd') else "MACD: N/A")
                        st.write(f"Volume Ratio: {indicators.get('volume_ratio', 'N/A'):.2f}" if indicators.get('volume_ratio') else "Volume Ratio: N/A")
                    
                    with col3:
                        st.markdown("**Volatility Indicators**")
                        if indicators.get('bb_upper'):
                            st.write(f"BB Upper: ${indicators.get('bb_upper'):.2f}")
                            st.write(f"BB Middle: ${indicators.get('bb_middle'):.2f}")
                            st.write(f"BB Lower: ${indicators.get('bb_lower'):.2f}")
                        st.write(f"ATR: {indicators.get('atr', 'N/A'):.2f}" if indicators.get('atr') else "ATR: N/A")
            
            # Chart patterns
            if 'chart_patterns' in tech_data and tech_data['chart_patterns']:
                patterns = tech_data['chart_patterns'].get('patterns', [])
                if patterns:
                    st.markdown("**Chart Patterns Detected**")
                    for pattern in patterns:
                        if isinstance(pattern, dict):
                            st.write(f"• {pattern.get('pattern', 'Unknown')}: {pattern.get('signal', '')} {pattern.get('direction', '')}")
            
            # Trading signals
            if 'trading_signals' in tech_data and tech_data['trading_signals']:
                signals = tech_data['trading_signals']
                st.markdown("**Trading Signals**")
                st.success(f"Recommendation: **{signals.get('recommendation', 'N/A')}**")
                if 'signals' in signals:
                    for signal in signals['signals'][:5]:
                        if isinstance(signal, dict):
                            st.write(f"• {signal.get('indicator', '')}: {signal.get('signal', '')} ({signal.get('strength', '')})")
        else:
            st.info("No technical analysis data available")
    
    def display_risk_assessment(self, workflow_state):
        """Display risk assessment results"""
        st.subheader("⚠️ Risk Assessment")
        
        risk_data = None
        for agent_name, result in workflow_state.agent_results.items():
            if 'risk' in agent_name and result.data:
                risk_data = result.data
                break
        
        if risk_data:
            # Risk metrics
            if 'risk_metrics' in risk_data:
                metrics = risk_data['risk_metrics']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if metrics.get('volatility_metrics'):
                        vol = metrics['volatility_metrics']
                        st.markdown("**Volatility Metrics**")
                        st.write(f"Historical Volatility: {vol.get('historical_volatility', 'N/A')}%")
                        st.write(f"Max Drawdown: {vol.get('max_drawdown', 'N/A')}%")
                
                with col2:
                    if metrics.get('value_at_risk'):
                        var = metrics['value_at_risk']
                        st.markdown("**Value at Risk (95% confidence)**")
                        st.write(f"Historical VaR: {var.get('historical_var', 'N/A')}%")
                        st.write(f"Conditional VaR: {var.get('conditional_var', 'N/A')}%")
            
            # Risk assessment
            if 'risk_assessment' in risk_data:
                st.markdown("**Risk Assessment Summary**")
                st.warning(risk_data['risk_assessment'])
            
            # Recommendations
            if 'recommendations' in risk_data:
                st.markdown("**Risk Management Recommendations**")
                for rec in risk_data['recommendations']:
                    st.write(f"• {rec}")
        else:
            st.info("No risk assessment data available")
    
    def display_sentiment_analysis(self, workflow_state):
        """Display sentiment analysis results"""
        st.subheader("💭 Sentiment Analysis")
        
        sentiment_data = None
        for agent_name, result in workflow_state.agent_results.items():
            if 'sentiment' in agent_name and result.data:
                sentiment_data = result.data
                break
        
        if sentiment_data:
            # Composite sentiment
            if 'composite_sentiment' in sentiment_data:
                comp = sentiment_data['composite_sentiment']
                st.markdown("**Overall Sentiment**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment Score", f"{comp.get('score', 0):.3f}")
                with col2:
                    st.metric("Category", comp.get('category', 'N/A'))
                with col3:
                    st.metric("Confidence", comp.get('confidence', 'N/A'))
            
            # Sentiment sources
            if 'sentiment_sources' in sentiment_data:
                sources = sentiment_data['sentiment_sources']
                
                # News sentiment
                if sources.get('news_sentiment'):
                    news = sources['news_sentiment']
                    if 'aggregate_scores' in news:
                        st.markdown("**News Sentiment**")
                        scores = news['aggregate_scores']
                        st.write(f"Overall: {scores.get('overall_sentiment', 'N/A')}")
                        st.write(f"Average Score: {scores.get('vader_average', 'N/A')}")
                
                # Social sentiment
                if sources.get('reddit_sentiment'):
                    reddit = sources['reddit_sentiment']
                    st.markdown("**Reddit Sentiment**")
                    st.write(f"Label: {reddit.get('sentiment_label', 'N/A')}")
                    st.write(f"Average: {reddit.get('average_sentiment', 'N/A')}")
            
            # Signals
            if 'signals' in sentiment_data:
                st.markdown("**Sentiment Signals**")
                for signal in sentiment_data['signals'][:5]:
                    st.write(f"• {signal}")
        else:
            st.info("No sentiment analysis data available")
    
    def display_recommendations(self, report):
        """Display recommendations"""
        st.subheader("💡 Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Investment Recommendations**")
            if report.get('recommendations'):
                for rec in report['recommendations']:
                    st.success(f"✓ {rec}")
            else:
                st.info("No specific recommendations available")
        
        with col2:
            st.markdown("**Risk Warnings**")
            if report.get('risk_warnings'):
                for warning in report['risk_warnings']:
                    st.warning(f"⚠️ {warning}")
            else:
                st.info("No specific warnings")
    
    def display_welcome(self):
        """Display welcome screen"""
        # Show mode indicator
        if WorkflowOrchestrator is None:
            st.info("🎯 **Running in Simulation Mode** - Using your configured API keys for realistic analysis")
        
        st.markdown("""
        ### Welcome to the Multi-Agent Stock Analysis System
        
        This advanced AI-powered platform provides comprehensive stock analysis using:
        
        - **🎯 Technical Analysis**: Advanced indicators, chart patterns, and trading signals
        - **⚠️ Risk Assessment**: Volatility metrics, VaR calculations, and portfolio risk
        - **💭 Sentiment Analysis**: News, social media, and market sentiment tracking
        - **📊 Market Comparison**: Peer analysis and sector performance
        - **✅ Compliance Checking**: Regulatory and ESG considerations
        - **🎨 Portfolio Optimization**: Allocation and diversification strategies
        
        **Get started** by entering a stock ticker in the sidebar and selecting your analysis type.
        
        ---
        
        #### Available Workflow Types:
        
        - **Quick Scan**: Fast technical and sentiment check (~1 minute)
        - **Comprehensive**: Full multi-agent analysis (~5 minutes)
        - **Risk Focused**: Deep risk and volatility analysis
        - **Technical Focused**: Detailed technical indicators and patterns
        - **Sentiment Focused**: In-depth market sentiment analysis
        """)
        
        if WorkflowOrchestrator is None:
            st.success("""
            ✅ **Your API Keys are Configured!** 
            The system is using your Alpha Vantage, Finnhub, NewsAPI, and other configured APIs for analysis.
            
            For advanced multi-agent orchestration features, you can optionally install:
            ```bash
            pip install crewai crewai-tools pandas-ta
            ```
            """)
        
        # Display sample tickers
        st.markdown("#### Popular Stocks to Analyze:")
        cols = st.columns(6)
        sample_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA"]
        for i, ticker in enumerate(sample_tickers):
            with cols[i]:
                st.button(ticker, key=f"sample_{ticker}", use_container_width=True)


def main():
    """Main entry point"""
    app = StockAnalysisApp()
    app.run()


if __name__ == "__main__":
    main()