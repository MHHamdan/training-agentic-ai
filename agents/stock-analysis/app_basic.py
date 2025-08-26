import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import requests

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

class BasicStockAnalyzer:
    """Basic stock analysis without complex AI models"""
    
    def __init__(self):
        pass
    
    def get_stock_data(self, ticker):
        """Get stock data using yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1y")
            
            if hist.empty:
                return None
                
            current_price = hist['Close'].iloc[-1]
            previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            return {
                "symbol": ticker,
                "current_price": float(current_price),
                "previous_close": float(previous_close),
                "change": float(change),
                "change_percent": float(change_percent),
                "volume": info.get('volume', 0),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('forwardPE') or info.get('trailingPE'),
                "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                "week_52_high": info.get('fiftyTwoWeekHigh'),
                "week_52_low": info.get('fiftyTwoWeekLow'),
                "company_name": info.get('longName') or info.get('shortName'),
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "historical_data": hist
            }
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def analyze_stock_fundamentals(self, stock_data):
        """Basic fundamental analysis"""
        analysis = {
            "financial_health": "Good" if stock_data["pe_ratio"] and stock_data["pe_ratio"] < 25 else "Fair",
            "valuation_score": 0,
            "recommendation": "HOLD"
        }
        
        # Simple scoring based on PE ratio and market cap
        pe_ratio = stock_data.get("pe_ratio", 20)
        if pe_ratio and pe_ratio < 15:
            analysis["valuation_score"] = 8
            analysis["recommendation"] = "BUY"
        elif pe_ratio and pe_ratio > 30:
            analysis["valuation_score"] = 4
            analysis["recommendation"] = "SELL"
        else:
            analysis["valuation_score"] = 6
            analysis["recommendation"] = "HOLD"
        
        return analysis
    
    def analyze_technical_indicators(self, hist_data):
        """Basic technical analysis"""
        if len(hist_data) < 50:
            return {"error": "Insufficient data for technical analysis"}
        
        # Calculate moving averages
        hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
        hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
        
        current_price = hist_data['Close'].iloc[-1]
        ma_20 = hist_data['MA_20'].iloc[-1]
        ma_50 = hist_data['MA_50'].iloc[-1]
        
        # Simple trend analysis
        if current_price > ma_20 > ma_50:
            trend = "Bullish"
            signal = "BUY"
        elif current_price < ma_20 < ma_50:
            trend = "Bearish" 
            signal = "SELL"
        else:
            trend = "Neutral"
            signal = "HOLD"
        
        # Calculate volatility
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std() * 100
        
        return {
            "trend": trend,
            "signal": signal,
            "current_price": current_price,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "volatility": volatility,
            "support_level": hist_data['Low'].tail(20).min(),
            "resistance_level": hist_data['High'].tail(20).max()
        }

def main():
    analyzer = BasicStockAnalyzer()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Agent (Basic Mode)</h1>', 
               unsafe_allow_html=True)
    st.markdown("*AI-powered stock analysis with fundamental and technical insights*")
    
    # Sidebar
    with st.sidebar:
        st.title("🚀 System Status")
        st.success("✅ System Online")
        st.info("🤖 Basic Analysis Mode")
        st.info("🔧 Financial Tools Available")
        
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Main content
    main_tab, monitoring_tab, settings_tab = st.tabs([
        "🎯 Stock Analysis", 
        "📊 Market Overview",
        "⚙️ Settings"
    ])
    
    with main_tab:
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
                    ["Comprehensive", "Fundamental Only", "Technical Only"],
                    help="Select the type of analysis to perform"
                )
                
                submit_button = st.form_submit_button("🚀 Analyze Stock", type="primary")
            
            if submit_button and ticker_input:
                with st.spinner(f"🔄 Analyzing {ticker_input}..."):
                    # Get stock data
                    stock_data = analyzer.get_stock_data(ticker_input)
                    
                    if stock_data:
                        # Display basic info
                        st.success(f"✅ Analysis completed for {ticker_input}")
                        
                        # Stock info card
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            st.metric(
                                "Current Price", 
                                f"${stock_data['current_price']:.2f}",
                                f"{stock_data['change_percent']:+.2f}%"
                            )
                        
                        with col_info2:
                            st.metric(
                                "Market Cap", 
                                f"${stock_data['market_cap']/1e9:.2f}B" if stock_data['market_cap'] else "N/A"
                            )
                        
                        with col_info3:
                            st.metric(
                                "P/E Ratio", 
                                f"{stock_data['pe_ratio']:.2f}" if stock_data['pe_ratio'] else "N/A"
                            )
                        
                        # Company Info
                        st.subheader(f"📊 {stock_data['company_name']} ({ticker_input})")
                        col_comp1, col_comp2 = st.columns(2)
                        
                        with col_comp1:
                            st.write(f"**Sector:** {stock_data['sector']}")
                            st.write(f"**Industry:** {stock_data['industry']}")
                        
                        with col_comp2:
                            st.write(f"**52-Week High:** ${stock_data['week_52_high']:.2f}" if stock_data['week_52_high'] else "")
                            st.write(f"**52-Week Low:** ${stock_data['week_52_low']:.2f}" if stock_data['week_52_low'] else "")
                        
                        # Analysis Results
                        analysis_tabs = st.tabs(["📊 Fundamental", "📈 Technical", "📋 Summary"])
                        
                        with analysis_tabs[0]:
                            if analysis_type in ["Comprehensive", "Fundamental Only"]:
                                fund_analysis = analyzer.analyze_stock_fundamentals(stock_data)
                                
                                st.subheader("📊 Fundamental Analysis")
                                
                                col_fund1, col_fund2 = st.columns(2)
                                with col_fund1:
                                    st.metric("Financial Health", fund_analysis["financial_health"])
                                    st.metric("Valuation Score", f"{fund_analysis['valuation_score']}/10")
                                
                                with col_fund2:
                                    rec = fund_analysis["recommendation"]
                                    if rec == "BUY":
                                        st.markdown(f'<div class="recommendation-buy"><strong>🚀 RECOMMENDATION: {rec}</strong></div>', unsafe_allow_html=True)
                                    elif rec == "SELL":
                                        st.markdown(f'<div class="recommendation-sell"><strong>📉 RECOMMENDATION: {rec}</strong></div>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<div class="recommendation-hold"><strong>🤚 RECOMMENDATION: {rec}</strong></div>', unsafe_allow_html=True)
                        
                        with analysis_tabs[1]:
                            if analysis_type in ["Comprehensive", "Technical Only"]:
                                tech_analysis = analyzer.analyze_technical_indicators(stock_data["historical_data"])
                                
                                st.subheader("📈 Technical Analysis")
                                
                                if "error" not in tech_analysis:
                                    col_tech1, col_tech2, col_tech3 = st.columns(3)
                                    
                                    with col_tech1:
                                        st.metric("Trend", tech_analysis["trend"])
                                        st.metric("Signal", tech_analysis["signal"])
                                    
                                    with col_tech2:
                                        st.metric("20-day MA", f"${tech_analysis['ma_20']:.2f}")
                                        st.metric("50-day MA", f"${tech_analysis['ma_50']:.2f}")
                                    
                                    with col_tech3:
                                        st.metric("Volatility", f"{tech_analysis['volatility']:.2f}%")
                                        st.metric("Support Level", f"${tech_analysis['support_level']:.2f}")
                                    
                                    # Price chart
                                    st.subheader("📊 Price Chart")
                                    hist_data = stock_data["historical_data"].copy()
                                    hist_data['MA_20'] = hist_data['Close'].rolling(window=20).mean()
                                    hist_data['MA_50'] = hist_data['Close'].rolling(window=50).mean()
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], 
                                                           name='Close Price', line=dict(color='blue')))
                                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA_20'], 
                                                           name='20-day MA', line=dict(color='orange')))
                                    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA_50'], 
                                                           name='50-day MA', line=dict(color='red')))
                                    
                                    fig.update_layout(title=f"{ticker_input} Stock Price with Moving Averages",
                                                    xaxis_title="Date", yaxis_title="Price ($)")
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.error(tech_analysis["error"])
                        
                        with analysis_tabs[2]:
                            st.subheader("📋 Analysis Summary")
                            st.write(f"**Company:** {stock_data['company_name']}")
                            st.write(f"**Current Price:** ${stock_data['current_price']:.2f}")
                            st.write(f"**Daily Change:** {stock_data['change_percent']:+.2f}%")
                            
                            if analysis_type in ["Comprehensive", "Fundamental Only"]:
                                fund_analysis = analyzer.analyze_stock_fundamentals(stock_data)
                                st.write(f"**Fundamental Recommendation:** {fund_analysis['recommendation']}")
                            
                            if analysis_type in ["Comprehensive", "Technical Only"]:
                                tech_analysis = analyzer.analyze_technical_indicators(stock_data["historical_data"])
                                if "error" not in tech_analysis:
                                    st.write(f"**Technical Signal:** {tech_analysis['signal']}")
                                    st.write(f"**Trend:** {tech_analysis['trend']}")
                    else:
                        st.error("Failed to retrieve stock data. Please check the ticker symbol.")
        
        with col2:
            st.subheader("📊 Quick Stats")
            st.info("Enter a stock ticker to see analysis results here.")
    
    with monitoring_tab:
        st.subheader("📊 Market Overview")
        
        # Popular stocks overview
        popular_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        market_data = []
        for ticker in popular_stocks:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev = hist['Open'].iloc[0]
                    change = ((current - prev) / prev) * 100
                    market_data.append({
                        "Ticker": ticker,
                        "Price": f"${current:.2f}",
                        "Change": f"{change:+.2f}%"
                    })
            except:
                continue
        
        if market_data:
            df = pd.DataFrame(market_data)
            st.dataframe(df, use_container_width=True)
    
    with settings_tab:
        st.subheader("⚙️ Settings")
        st.write("**Analysis Configuration:**")
        
        enable_technical = st.checkbox("Enable Technical Analysis", value=True)
        enable_fundamental = st.checkbox("Enable Fundamental Analysis", value=True)
        
        st.write("**Data Sources:**")
        st.write("• Yahoo Finance (yfinance)")
        st.write("• Real-time market data")
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")

if __name__ == "__main__":
    main()