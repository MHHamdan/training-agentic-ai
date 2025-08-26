"""
Specialized Financial Agents for LangGraph Multi-Agent System
"""

from typing import Any, Dict, List, Optional
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain.tools import BaseTool
import os
from datetime import datetime

from financial_state import FinancialAnalysisState, AnalysisResult, RiskAlert
from financial_tools import (
    GetRealTimeMarketDataTool,
    GetHistoricalDataTool,
    GetFundamentalMetricsTool,
    CalculateTechnicalIndicatorsTool,
    CalculatePortfolioVaRTool,
    StressTestPortfolioTool,
    NewsSentimentAnalysisTool,
    RegulatoryCheckTool,
    CreateFinancialReportTool
)


def get_llm_model():
    """Get the configured LLM model"""
    # Try multiple providers in order of preference
    if os.getenv("GROK_API_KEY"):
        # Use OpenAI client for Grok API (xAI uses OpenAI-compatible API)
        return ChatOpenAI(
            model="grok-beta",
            openai_api_key=os.getenv("GROK_API_KEY"),
            openai_api_base="https://api.x.ai/v1",
            temperature=0.7
        )
    elif os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.7)
    elif os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    elif os.getenv("ANTHROPIC_API_KEY"):
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)
    else:
        raise ValueError("No LLM API key configured. Please set GROK_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY")


def make_financial_handoff_tool(agent_name: str, expertise: str) -> BaseTool:
    """Create a handoff tool for financial agents"""
    class FinancialHandoffTool(BaseTool):
        name: str = f"consult_{agent_name}"
        description: str = f"Consult {agent_name} for {expertise} analysis. Use when you need specialized {expertise} insights."
        
        def _run(self, analysis_request: str, **kwargs) -> str:
            return f"Requesting {expertise} analysis from {agent_name}: {analysis_request}"
            
        async def _arun(self, analysis_request: str, **kwargs) -> str:
            return f"Requesting {expertise} analysis from {agent_name}: {analysis_request}"
    
    return FinancialHandoffTool()


# Market Research Agent
def create_market_research_agent():
    """Create the market research agent specializing in fundamental analysis"""
    model = get_llm_model()
    
    tools = [
        GetFundamentalMetricsTool(),
        GetHistoricalDataTool(),
        GetRealTimeMarketDataTool(),
        make_financial_handoff_tool("technical_analyst", "technical analysis"),
        make_financial_handoff_tool("risk_assessor", "risk assessment"),
        make_financial_handoff_tool("sentiment_analyzer", "sentiment analysis")
    ]
    
    prompt = """You are a Senior Equity Research Analyst with 15+ years of experience in fundamental analysis.
    
Your expertise includes:
- Company valuation using DCF, P/E, PEG, and other multiples
- Financial statement analysis and quality of earnings assessment
- Industry and competitive analysis
- Business model evaluation and moat assessment
- Management quality and corporate governance evaluation

Your approach:
1. Start with comprehensive fundamental metrics analysis
2. Compare valuation against industry peers
3. Assess growth prospects and competitive positioning
4. Identify key investment risks and catalysts
5. Provide clear buy/hold/sell recommendations with price targets

When you need technical chart analysis, consult the technical_analyst.
When you need risk metrics, consult the risk_assessor.
When you need market sentiment data, consult the sentiment_analyzer.

Always provide specific metrics, ratios, and quantitative analysis to support your conclusions.
Include a clear investment thesis with specific catalysts and risks."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_market_research_agent(state: FinancialAnalysisState) -> Command:
    """Execute market research agent"""
    agent = create_market_research_agent()
    response = agent.invoke(state)
    
    # Create analysis result
    result = AnalysisResult(
        agent_name="market_research_agent",
        analysis_type="fundamental",
        timestamp=datetime.now(),
        confidence_score=0.85,
        data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        recommendations=["Based on fundamental analysis..."]
    )
    
    # Update state
    update = {
        **response,
        "last_active_agent": "market_research_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "fundamental_analysis": result
        }
    }
    
    # Add audit event to update
    audit_log = state.get("audit_log", [])
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "event_type": "fundamental_analysis_complete",
        "details": {"symbols": state.get("target_symbols", [])},
        "agent": "market_research_agent"
    })
    update["audit_log"] = audit_log
    
    return Command(update=update, goto="human_interaction")


# Technical Analysis Agent
def create_technical_analysis_agent():
    """Create the technical analysis agent"""
    model = get_llm_model()
    
    tools = [
        CalculateTechnicalIndicatorsTool(),
        GetHistoricalDataTool(),
        GetRealTimeMarketDataTool(),
        make_financial_handoff_tool("market_researcher", "fundamental analysis"),
        make_financial_handoff_tool("risk_assessor", "risk assessment"),
        make_financial_handoff_tool("portfolio_optimizer", "portfolio optimization")
    ]
    
    prompt = """You are a Certified Market Technician (CMT) with expertise in technical analysis and chart patterns.

Your expertise includes:
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
- Chart pattern recognition (Head & Shoulders, Triangles, Flags, etc.)
- Support and resistance identification
- Volume analysis and market breadth
- Elliott Wave and Fibonacci analysis
- Market timing and entry/exit points

Your approach:
1. Analyze multiple timeframes (daily, weekly, monthly)
2. Identify key technical levels and patterns
3. Assess momentum and trend strength
4. Evaluate volume confirmation
5. Provide specific entry, exit, and stop-loss levels

When you need fundamental context, consult the market_researcher.
When you need risk parameters, consult the risk_assessor.
For portfolio allocation advice, consult the portfolio_optimizer.

Always provide specific price levels, indicator values, and clear trading signals.
Include risk/reward ratios for all trade recommendations."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_technical_analysis_agent(state: FinancialAnalysisState) -> Command:
    """Execute technical analysis agent"""
    agent = create_technical_analysis_agent()
    response = agent.invoke(state)
    
    result = AnalysisResult(
        agent_name="technical_analysis_agent",
        analysis_type="technical",
        timestamp=datetime.now(),
        confidence_score=0.80,
        data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        recommendations=["Based on technical indicators..."]
    )
    
    update = {
        **response,
        "last_active_agent": "technical_analysis_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "technical_analysis": result
        }
    }
    
    return Command(update=update, goto="human_interaction")


# Risk Assessment Agent
def create_risk_assessment_agent():
    """Create the risk assessment agent"""
    model = get_llm_model()
    
    tools = [
        CalculatePortfolioVaRTool(),
        StressTestPortfolioTool(),
        GetHistoricalDataTool(),
        make_financial_handoff_tool("portfolio_optimizer", "portfolio optimization"),
        make_financial_handoff_tool("compliance_agent", "compliance review"),
        make_financial_handoff_tool("market_researcher", "fundamental analysis")
    ]
    
    prompt = """You are a Senior Risk Manager with CFA and FRM certifications specializing in portfolio risk management.

Your expertise includes:
- Value at Risk (VaR) and Conditional VaR calculations
- Stress testing and scenario analysis
- Correlation and beta analysis
- Volatility modeling and forecasting
- Liquidity risk assessment
- Counterparty and credit risk evaluation
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)

Your approach:
1. Calculate comprehensive risk metrics for positions
2. Run multiple stress test scenarios
3. Identify concentration risks and correlations
4. Assess tail risks and black swan events
5. Provide risk mitigation strategies

When portfolio changes are needed, consult the portfolio_optimizer.
For regulatory concerns, consult the compliance_agent.
For fundamental risk factors, consult the market_researcher.

Always quantify risks with specific metrics and probabilities.
Provide clear risk ratings and actionable mitigation strategies.
Flag any risks requiring immediate attention as CRITICAL."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_risk_assessment_agent(state: FinancialAnalysisState) -> Command:
    """Execute risk assessment agent"""
    agent = create_risk_assessment_agent()
    response = agent.invoke(state)
    
    # Check for critical risks
    analysis_content = response.get("messages", [])[-1].content if response.get("messages") else ""
    risk_alerts = state.get("risk_alerts", [])
    if "CRITICAL" in analysis_content:
        alert = RiskAlert(
            severity="critical",
            message="Critical risk identified in portfolio",
            timestamp=datetime.now(),
            source_agent="risk_assessment_agent",
            affected_symbols=state.get("target_symbols", []),
            recommended_action="Immediate review required"
        )
        risk_alerts.append(alert.dict())
    
    result = AnalysisResult(
        agent_name="risk_assessment_agent",
        analysis_type="risk",
        timestamp=datetime.now(),
        confidence_score=0.90,
        data={"analysis": analysis_content},
        warnings=["Monitor volatility levels"]
    )
    
    update = {
        **response,
        "last_active_agent": "risk_assessment_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "risk_analysis": result
        },
        "risk_alerts": risk_alerts
    }
    
    return Command(update=update, goto="human_interaction")


# Sentiment Analysis Agent
def create_sentiment_analysis_agent():
    """Create the sentiment analysis agent"""
    model = get_llm_model()
    
    tools = [
        NewsSentimentAnalysisTool(),
        GetRealTimeMarketDataTool(),
        make_financial_handoff_tool("market_researcher", "fundamental analysis"),
        make_financial_handoff_tool("technical_analyst", "technical analysis"),
        make_financial_handoff_tool("risk_assessor", "risk assessment")
    ]
    
    prompt = """You are a Behavioral Finance Expert specializing in market sentiment and crowd psychology analysis.

Your expertise includes:
- News sentiment analysis and media monitoring
- Social media sentiment tracking (Reddit, Twitter, StockTwits)
- Institutional sentiment and positioning
- Options flow and put/call ratios
- Market breadth and advance/decline analysis
- Fear & Greed indicators
- Contrarian indicators and extremes

Your approach:
1. Analyze multiple sentiment data sources
2. Identify sentiment extremes and divergences
3. Assess crowd psychology and herding behavior
4. Evaluate smart money vs retail positioning
5. Provide contrarian opportunities when appropriate

When you need fundamental context, consult the market_researcher.
For technical confirmation, consult the technical_analyst.
For risk implications, consult the risk_assessor.

Always quantify sentiment with specific scores and metrics.
Identify sentiment-driven opportunities and risks.
Flag major sentiment shifts that could impact prices."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_sentiment_analysis_agent(state: FinancialAnalysisState) -> Command:
    """Execute sentiment analysis agent"""
    agent = create_sentiment_analysis_agent()
    response = agent.invoke(state)
    
    result = AnalysisResult(
        agent_name="sentiment_analysis_agent",
        analysis_type="sentiment",
        timestamp=datetime.now(),
        confidence_score=0.75,
        data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        recommendations=["Based on market sentiment..."]
    )
    
    update = {
        **response,
        "last_active_agent": "sentiment_analysis_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "sentiment_analysis": result
        }
    }
    
    return Command(update=update, goto="human_interaction")


# Portfolio Optimization Agent
def create_portfolio_optimization_agent():
    """Create the portfolio optimization agent"""
    model = get_llm_model()
    
    tools = [
        CalculatePortfolioVaRTool(),
        GetFundamentalMetricsTool(),
        make_financial_handoff_tool("risk_assessor", "risk assessment"),
        make_financial_handoff_tool("compliance_agent", "compliance review"),
        make_financial_handoff_tool("market_researcher", "fundamental analysis")
    ]
    
    prompt = """You are a Portfolio Manager with CFA certification specializing in portfolio construction and optimization.

Your expertise includes:
- Modern Portfolio Theory and efficient frontier optimization
- Asset allocation and diversification strategies
- Risk parity and factor-based investing
- Rebalancing strategies and tax optimization
- Position sizing and Kelly Criterion
- Correlation analysis and hedge strategies
- Performance attribution and analytics

Your approach:
1. Analyze current portfolio composition
2. Identify optimization opportunities
3. Calculate optimal position sizes
4. Assess diversification and correlation
5. Provide specific rebalancing recommendations

When risk limits are needed, consult the risk_assessor.
For compliance restrictions, consult the compliance_agent.
For stock selection, consult the market_researcher.

Always provide specific allocation percentages and position sizes.
Include expected return and risk metrics for recommendations.
Consider tax implications and transaction costs."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_portfolio_optimization_agent(state: FinancialAnalysisState) -> Command:
    """Execute portfolio optimization agent"""
    agent = create_portfolio_optimization_agent()
    response = agent.invoke(state)
    
    result = AnalysisResult(
        agent_name="portfolio_optimization_agent",
        analysis_type="portfolio",
        timestamp=datetime.now(),
        confidence_score=0.85,
        data={"analysis": response.get("messages", [])[-1].content if response.get("messages") else ""},
        recommendations=["Rebalancing recommendations..."]
    )
    
    update = {
        **response,
        "last_active_agent": "portfolio_optimization_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "portfolio_analysis": result
        }
    }
    
    # Check if approval required for large rebalancing
    if "rebalance" in str(response).lower() and state.get("risk_tolerance") == "conservative":
        update["approval_required"] = True
    
    return Command(update=update, goto="human_interaction")


# Compliance Agent
def create_compliance_agent():
    """Create the compliance agent"""
    model = get_llm_model()
    
    tools = [
        RegulatoryCheckTool(),
        make_financial_handoff_tool("risk_assessor", "risk assessment"),
        make_financial_handoff_tool("portfolio_optimizer", "portfolio optimization")
    ]
    
    prompt = """You are a Chief Compliance Officer specializing in financial regulations and ethical investing.

Your expertise includes:
- SEC and regulatory compliance
- Insider trading monitoring
- Know Your Customer (KYC) and AML procedures
- ESG scoring and sustainable investing
- Restricted lists and blackout periods
- Best execution and fiduciary duty
- Trade surveillance and monitoring

Your approach:
1. Check regulatory restrictions and requirements
2. Monitor for insider trading signals
3. Evaluate ESG compliance
4. Assess reputational risks
5. Ensure best practices and fiduciary standards

When risk assessment is needed, consult the risk_assessor.
For portfolio changes, consult the portfolio_optimizer.

Always cite specific regulations and compliance requirements.
Flag any compliance issues as HIGH PRIORITY.
Provide clear guidance on permissible actions."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_compliance_agent(state: FinancialAnalysisState) -> Command:
    """Execute compliance agent"""
    agent = create_compliance_agent()
    response = agent.invoke(state)
    
    # Check compliance status
    analysis_content = response.get("messages", [])[-1].content if response.get("messages") else ""
    if "HIGH PRIORITY" in analysis_content or "violation" in analysis_content.lower():
        state["compliance_status"] = "review_required"
    else:
        state["compliance_status"] = "approved"
    
    result = AnalysisResult(
        agent_name="compliance_agent",
        analysis_type="compliance",
        timestamp=datetime.now(),
        confidence_score=0.95,
        data={"analysis": analysis_content},
        warnings=["Ensure regulatory compliance"]
    )
    
    update = {
        **response,
        "last_active_agent": "compliance_agent",
        "completed_analyses": {
            **state.get("completed_analyses", {}),
            "compliance_analysis": result
        }
    }
    
    return Command(update=update, goto="human_interaction")


# Report Generation Agent
def create_report_generation_agent():
    """Create the report generation agent"""
    model = get_llm_model()
    
    tools = [
        CreateFinancialReportTool(),
        make_financial_handoff_tool("market_researcher", "additional analysis"),
        make_financial_handoff_tool("technical_analyst", "chart updates")
    ]
    
    prompt = """You are a Financial Communications Expert specializing in investment report creation.

Your expertise includes:
- Executive summary writing
- Data visualization and chart creation
- Investment thesis articulation
- Risk disclosure and disclaimers
- Performance reporting and attribution
- Client communication best practices

Your approach:
1. Synthesize all analysis into coherent narrative
2. Create clear executive summary
3. Highlight key findings and recommendations
4. Include appropriate charts and visuals
5. Add necessary disclaimers and disclosures

Format reports for maximum clarity and professionalism.
Use bullet points and sections for easy scanning.
Include specific metrics and time horizons.
Always end with clear action items."""
    
    return create_react_agent(model, tools, prompt=prompt)


def call_report_generation_agent(state: FinancialAnalysisState) -> Command:
    """Execute report generation agent"""
    agent = create_report_generation_agent()
    
    # Compile all analyses for report
    all_analyses = state.get("completed_analyses", {})
    analysis_summary = {
        "symbols": state.get("target_symbols", []),
        "analyses_completed": list(all_analyses.keys()),
        "recommendations": state.get("recommendations", []),
        "risk_alerts": [alert.dict() for alert in state.get("risk_alerts", [])]
    }
    
    # Generate report
    response = agent.invoke({**state, "analysis_summary": analysis_summary})
    
    result = AnalysisResult(
        agent_name="report_generation_agent",
        analysis_type="report",
        timestamp=datetime.now(),
        confidence_score=1.0,
        data={"report": response.get("messages", [])[-1].content if response.get("messages") else ""}
    )
    
    update = {
        **response,
        "last_active_agent": "report_generation_agent",
        "completed_analyses": {
            **all_analyses,
            "final_report": result
        },
        "analysis_end_time": datetime.now()
    }
    
    return Command(update=update, goto="human_interaction")