import logging
from crewai import Task
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StockAnalysisTasks:
    """CrewAI task definitions for stock analysis workflow"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_fundamental_analysis_task(self, agent, ticker: str, financial_data: Dict[str, Any]) -> Task:
        """Create fundamental analysis task"""
        return Task(
            description=f"""
            Perform comprehensive fundamental analysis for {ticker}.
            
            OBJECTIVE:
            Analyze the financial health, valuation, competitive position, and growth prospects
            of {ticker} using the provided financial data.
            
            FINANCIAL DATA PROVIDED:
            {financial_data}
            
            DELIVERABLES:
            1. Financial health assessment with specific metrics
            2. Valuation analysis with industry comparisons
            3. Competitive positioning evaluation
            4. Growth prospects and sustainability analysis
            5. Risk factors identification
            6. Investment recommendation with rationale
            
            REQUIREMENTS:
            - Provide specific quantitative metrics (P/E, ROE, debt ratios, etc.)
            - Compare valuations to industry averages
            - Identify key competitive advantages or weaknesses
            - Assess management effectiveness and strategy
            - Quantify growth opportunities and threats
            - Include regulatory compliance considerations
            
            Expected output: Structured fundamental analysis with actionable insights
            """,
            agent=agent,
            expected_output="Comprehensive fundamental analysis report with investment recommendation"
        )
    
    def create_technical_analysis_task(self, agent, ticker: str, price_data: Dict[str, Any]) -> Task:
        """Create technical analysis task"""
        return Task(
            description=f"""
            Perform comprehensive technical analysis for {ticker}.
            
            OBJECTIVE:
            Analyze price trends, chart patterns, technical indicators, and trading signals
            for {ticker} using the provided price and volume data.
            
            PRICE DATA PROVIDED:
            {price_data}
            
            DELIVERABLES:
            1. Trend analysis with direction and strength assessment
            2. Support and resistance level identification
            3. Chart pattern recognition and completion probability
            4. Technical indicator analysis (RSI, MACD, moving averages)
            5. Volume analysis and confirmation signals
            6. Trading signals with entry/exit points
            7. Risk management recommendations
            
            REQUIREMENTS:
            - Identify specific price levels for support/resistance
            - Calculate technical indicator values and signals
            - Assess chart pattern reliability and targets
            - Provide specific entry/exit recommendations
            - Include position sizing guidelines
            - Consider volatility and risk metrics
            
            Expected output: Detailed technical analysis with trading recommendations
            """,
            agent=agent,
            expected_output="Technical analysis report with specific trading signals and price targets"
        )
    
    def create_news_sentiment_task(self, agent, ticker: str, news_data: List[Dict[str, Any]]) -> Task:
        """Create news sentiment analysis task"""
        return Task(
            description=f"""
            Analyze market sentiment and news impact for {ticker}.
            
            OBJECTIVE:
            Assess the current market sentiment, news impact, and social indicators
            that may affect {ticker}'s stock price and trading activity.
            
            NEWS DATA PROVIDED:
            {news_data}
            
            DELIVERABLES:
            1. Overall sentiment score with confidence level
            2. Key sentiment drivers (positive/negative factors)
            3. News impact assessment on price movements
            4. Social media and retail sentiment indicators
            5. Analyst sentiment and rating changes
            6. Trading volume and volatility implications
            7. Forward-looking sentiment catalysts
            
            REQUIREMENTS:
            - Provide numerical sentiment score (-100 to +100)
            - Identify market-moving news events
            - Correlate news timing with price movements
            - Assess sentiment reliability and consistency
            - Include contrarian sentiment indicators
            - Consider regulatory and compliance implications
            
            Expected output: Sentiment analysis with trading implications and risk assessment
            """,
            agent=agent,
            expected_output="Market sentiment analysis with impact assessment and trading implications"
        )
    
    def create_risk_assessment_task(self, agent, ticker: str, portfolio_data: Dict[str, Any],
                                  analysis_data: Dict[str, Any]) -> Task:
        """Create comprehensive risk assessment task"""
        return Task(
            description=f"""
            Perform comprehensive risk assessment for {ticker} investment.
            
            OBJECTIVE:
            Evaluate all categories of investment risk including market, company-specific,
            liquidity, concentration, and regulatory risks for {ticker}.
            
            DATA PROVIDED:
            Portfolio Data: {portfolio_data}
            Analysis Data: {analysis_data}
            
            DELIVERABLES:
            1. Market risk analysis with VaR calculations
            2. Company-specific risk evaluation
            3. Liquidity risk assessment
            4. Concentration risk analysis
            5. Regulatory compliance validation
            6. Stress testing scenarios and results
            7. Risk mitigation strategies
            8. Overall risk rating and recommendations
            
            REQUIREMENTS:
            - Calculate specific risk metrics (VaR, beta, volatility)
            - Assess regulatory compliance requirements
            - Perform stress testing scenarios
            - Provide quantitative risk measurements
            - Include audit trail and documentation
            - Ensure SEC and FINRA compliance
            
            Expected output: Comprehensive risk assessment with regulatory compliance validation
            """,
            agent=agent,
            expected_output="Risk assessment report with compliance validation and mitigation strategies"
        )
    
    def create_comprehensive_report_task(self, agent, ticker: str, all_analysis_data: Dict[str, Any]) -> Task:
        """Create comprehensive investment report task"""
        return Task(
            description=f"""
            Generate comprehensive investment research report for {ticker}.
            
            OBJECTIVE:
            Synthesize all analysis results into a professional-grade investment research
            report that meets institutional standards and regulatory requirements.
            
            ANALYSIS DATA PROVIDED:
            {all_analysis_data}
            
            DELIVERABLES:
            1. Executive summary with clear recommendation
            2. Company overview and business analysis
            3. Fundamental analysis synthesis
            4. Technical analysis integration
            5. Sentiment and news impact summary
            6. Risk assessment compilation
            7. Financial projections and scenarios
            8. Investment recommendation with rationale
            9. Regulatory disclosures and compliance
            10. Professional formatting and presentation
            
            REQUIREMENTS:
            - Synthesize all analyst inputs cohesively
            - Provide clear BUY/HOLD/SELL recommendation
            - Include specific price targets and timelines
            - Ensure regulatory compliance and disclosures
            - Format for professional distribution
            - Include risk warnings and disclaimers
            
            Expected output: Professional investment research report ready for distribution
            """,
            agent=agent,
            expected_output="Comprehensive investment research report with recommendation and compliance"
        )
    
    def create_portfolio_risk_task(self, agent, portfolio: Dict[str, Any], 
                                 benchmark_data: Dict[str, Any]) -> Task:
        """Create portfolio-level risk assessment task"""
        return Task(
            description=f"""
            Assess portfolio-level risk metrics and exposures.
            
            OBJECTIVE:
            Evaluate the overall risk profile of the investment portfolio including
            diversification, factor exposures, and regulatory compliance.
            
            PORTFOLIO DATA:
            {portfolio}
            
            BENCHMARK DATA:
            {benchmark_data}
            
            DELIVERABLES:
            1. Portfolio risk metrics calculation
            2. Diversification analysis
            3. Factor exposure assessment
            4. Stress testing results
            5. Risk attribution analysis
            6. Liquidity assessment
            7. Regulatory compliance check
            
            REQUIREMENTS:
            - Calculate portfolio beta, volatility, VaR
            - Assess concentration limits compliance
            - Perform multi-factor risk attribution
            - Include stress testing scenarios
            - Validate regulatory position limits
            - Provide rebalancing recommendations
            
            Expected output: Portfolio risk analysis with compliance validation
            """,
            agent=agent,
            expected_output="Portfolio risk assessment with regulatory compliance validation"
        )
    
    def create_compliance_validation_task(self, agent, investment_decision: Dict[str, Any],
                                        client_profile: Dict[str, Any]) -> Task:
        """Create regulatory compliance validation task"""
        return Task(
            description=f"""
            Validate regulatory compliance for investment decision.
            
            OBJECTIVE:
            Ensure all investment recommendations comply with SEC, FINRA, and other
            regulatory requirements including suitability and disclosure obligations.
            
            INVESTMENT DECISION:
            {investment_decision}
            
            CLIENT PROFILE:
            {client_profile}
            
            DELIVERABLES:
            1. Suitability assessment validation
            2. SEC compliance verification
            3. FINRA rule compliance check
            4. Risk disclosure validation
            5. Position and concentration limits
            6. Documentation requirements
            7. Compliance status report
            
            REQUIREMENTS:
            - Validate investment objective alignment
            - Check risk tolerance compatibility
            - Verify disclosure completeness
            - Ensure position limit compliance
            - Document audit trail requirements
            - Provide remediation steps if needed
            
            Expected output: Regulatory compliance validation with pass/fail status
            """,
            agent=agent,
            expected_output="Compliance validation report with regulatory status and remediation"
        )
    
    def create_options_flow_task(self, agent, ticker: str, options_data: Dict[str, Any]) -> Task:
        """Create options flow analysis task"""
        return Task(
            description=f"""
            Analyze options flow and institutional positioning for {ticker}.
            
            OBJECTIVE:
            Assess options market activity, implied volatility, and institutional
            positioning to gauge market sentiment and potential price movements.
            
            OPTIONS DATA:
            {options_data}
            
            DELIVERABLES:
            1. Options flow analysis (call/put ratios)
            2. Implied volatility assessment
            3. Gamma and delta exposure analysis
            4. Unusual activity identification
            5. Institutional positioning indicators
            6. Price range expectations
            7. Strategic trading implications
            
            REQUIREMENTS:
            - Calculate put/call ratios and trends
            - Assess IV rank and term structure
            - Identify large block trades
            - Analyze gamma exposure levels
            - Provide volatility expectations
            - Include market maker positioning
            
            Expected output: Options flow analysis with sentiment and positioning insights
            """,
            agent=agent,
            expected_output="Options flow analysis with market positioning and volatility insights"
        )
    
    def create_earnings_analysis_task(self, agent, ticker: str, earnings_data: Dict[str, Any],
                                    analyst_coverage: List[Dict[str, Any]]) -> Task:
        """Create earnings sentiment analysis task"""
        return Task(
            description=f"""
            Analyze earnings-related sentiment and analyst reactions for {ticker}.
            
            OBJECTIVE:
            Assess the market reaction to recent earnings, management guidance,
            and analyst sentiment shifts following earnings announcements.
            
            EARNINGS DATA:
            {earnings_data}
            
            ANALYST COVERAGE:
            {analyst_coverage}
            
            DELIVERABLES:
            1. Earnings surprise analysis
            2. Management commentary sentiment
            3. Analyst sentiment shifts
            4. Earnings call sentiment
            5. Market reaction assessment
            6. Sector comparison analysis
            7. Forward guidance implications
            
            REQUIREMENTS:
            - Quantify revenue and EPS surprises
            - Assess guidance quality and credibility
            - Track analyst rating and target changes
            - Evaluate market reaction sustainability
            - Compare to sector earnings trends
            - Include forward-looking catalysts
            
            Expected output: Earnings sentiment analysis with forward guidance assessment
            """,
            agent=agent,
            expected_output="Earnings analysis with sentiment assessment and analyst reaction summary"
        )
    
    def create_pattern_recognition_task(self, agent, ticker: str, price_data: Dict[str, Any]) -> Task:
        """Create chart pattern recognition task"""
        return Task(
            description=f"""
            Identify and analyze chart patterns for {ticker}.
            
            OBJECTIVE:
            Recognize continuation and reversal chart patterns, assess their
            completion probability, and provide price targets and trading strategies.
            
            PRICE DATA:
            {price_data}
            
            DELIVERABLES:
            1. Continuation pattern identification
            2. Reversal pattern recognition
            3. Candlestick pattern analysis
            4. Pattern completion assessment
            5. Breakout probability calculation
            6. Price target methodology
            7. Trading strategy recommendations
            
            REQUIREMENTS:
            - Identify specific pattern types
            - Calculate completion probabilities
            - Provide measured move targets
            - Assess pattern reliability
            - Include volume confirmation
            - Consider failure rate statistics
            
            Expected output: Chart pattern analysis with trading recommendations and price targets
            """,
            agent=agent,
            expected_output="Pattern recognition analysis with breakout probability and price targets"
        )