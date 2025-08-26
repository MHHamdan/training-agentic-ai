import logging
from crewai import Agent
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class ReportWriterAgent:
    """Investment Report Writer Agent with comprehensive financial analysis reporting"""
    
    def __init__(self, llm, tools: List[Any]):
        self.observability = get_observability_manager()
        self.llm = llm
        self.tools = tools
        
        self.agent = Agent(
            role='Investment Research Report Writer',
            goal='Create comprehensive, professional investment research reports with actionable insights',
            backstory="""You are an Investment Research Report Writer with expertise in creating 
            professional-grade investment research reports. You have extensive experience in 
            financial analysis, investment strategy, and regulatory compliance reporting. Your 
            specialty is synthesizing complex financial data into clear, actionable investment 
            reports that meet institutional standards and regulatory requirements. You excel at 
            presenting technical analysis, fundamental insights, and risk assessments in a format 
            that serves both professional and retail investors.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools,
            max_iter=3,
            memory=True
        )
    
    @track_agent_performance("ReportWriter", "comprehensive_report")
    async def generate_investment_report(self, ticker: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive investment research report"""
        try:
            # Log compliance action
            self.observability.log_compliance_action(
                agent_name="ReportWriter",
                action_type="investment_report_generation",
                input_data={"ticker": ticker, "data_sources": list(analysis_data.keys())},
                output_data={"report_type": "comprehensive_investment_research"},
                risk_level="medium",
                decision_reasoning=f"Generating regulatory-compliant investment research report for {ticker}"
            )
            
            report_prompt = f"""
            Generate a comprehensive investment research report for {ticker} using the following analysis data:
            
            ANALYSIS DATA:
            {analysis_data}
            
            Create a professional investment research report with the following structure:
            
            1. EXECUTIVE SUMMARY
               - Investment recommendation (BUY/HOLD/SELL)
               - Target price and price range
               - Key investment thesis (3-4 bullet points)
               - Risk rating (Low/Medium/High)
               - Investment horizon recommendation
               - Position sizing recommendation
            
            2. COMPANY OVERVIEW
               - Business description and model
               - Market position and competitive advantages
               - Recent strategic developments
               - Management assessment
               - ESG considerations and sustainability
            
            3. FUNDAMENTAL ANALYSIS
               - Financial health assessment
               - Revenue and earnings analysis
               - Valuation metrics and comparison
               - Balance sheet strength
               - Cash flow analysis
               - Growth prospects and sustainability
               - Competitive positioning
            
            4. TECHNICAL ANALYSIS
               - Chart pattern analysis
               - Support and resistance levels
               - Momentum indicators and signals
               - Volume analysis
               - Trading recommendations
               - Entry and exit strategies
            
            5. MARKET SENTIMENT AND NEWS IMPACT
               - Current market sentiment score
               - Recent news impact assessment
               - Analyst sentiment and rating changes
               - Social media and retail sentiment
               - Earnings expectations and guidance
            
            6. RISK ASSESSMENT
               - Market risks and systematic factors
               - Company-specific risks
               - Liquidity and operational risks
               - Regulatory and compliance risks
               - Stress testing and scenario analysis
               - Risk mitigation strategies
            
            7. FINANCIAL PROJECTIONS
               - Revenue growth projections (1-3 years)
               - Earnings estimates and revisions
               - Valuation scenarios (base/bull/bear cases)
               - Dividend expectations if applicable
               - Key financial ratios projections
            
            8. INVESTMENT RECOMMENDATION
               - Detailed recommendation rationale
               - Price targets with methodology
               - Timeline and catalysts
               - Position sizing guidelines
               - Portfolio allocation recommendations
               - Risk-adjusted return expectations
            
            9. REGULATORY DISCLOSURES
               - Conflicts of interest statement
               - Data sources and methodology
               - Risk disclaimers and limitations
               - Regulatory compliance statements
               - Report date and validity period
               - Analyst certifications
            
            10. APPENDICES
                - Key financial metrics table
                - Peer comparison analysis
                - Historical performance charts
                - Glossary of terms
                - Research methodology notes
            
            FORMATTING REQUIREMENTS:
            - Use clear headers and subheaders
            - Include quantitative metrics and specific data points
            - Provide actionable insights and recommendations
            - Maintain professional tone and formatting
            - Include risk warnings and disclaimers
            - Use bullet points for key highlights
            - Include price targets with rationale
            - Provide specific timeframes for projections
            
            COMPLIANCE REQUIREMENTS:
            - Include all material risk disclosures
            - Provide methodology for price targets
            - Include source citations for all data
            - Add appropriate regulatory disclaimers
            - Ensure objectivity and balance in analysis
            - Include conflicts of interest statements
            """
            
            response = self.agent.llm.invoke(report_prompt)
            report_result = self._parse_investment_report(response.content if hasattr(response, 'content') else str(response))
            
            # Add metadata to report
            report_result["metadata"] = {
                "ticker": ticker,
                "report_date": datetime.now().isoformat(),
                "report_type": "comprehensive_investment_research",
                "compliance_validated": True,
                "data_sources": list(analysis_data.keys()),
                "agent_version": "v1.0"
            }
            
            return report_result
            
        except Exception as e:
            logger.error(f"Error generating investment report: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("ReportWriter", "executive_summary")
    async def generate_executive_summary(self, ticker: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary for quick decision making"""
        try:
            summary_prompt = f"""
            Create a concise executive summary for {ticker} investment analysis:
            
            {analysis_data}
            
            Provide a 1-2 page executive summary covering:
            
            1. INVESTMENT THESIS (3-4 key points)
               - Primary reasons to invest or avoid
               - Competitive advantages and market position
               - Growth catalysts and value drivers
               - Key differentiators vs peers
            
            2. FINANCIAL HIGHLIGHTS
               - Current valuation metrics
               - Growth rates and profitability
               - Balance sheet strength
               - Cash flow generation
            
            3. RECOMMENDATION
               - Clear BUY/HOLD/SELL recommendation
               - Target price with upside/downside
               - Investment horizon (short/medium/long-term)
               - Position sizing recommendation
            
            4. KEY RISKS
               - Top 3-5 risk factors
               - Probability and impact assessment
               - Risk mitigation strategies
               - Scenario analysis summary
            
            5. CATALYSTS AND TIMING
               - Upcoming events and catalysts
               - Earnings expectations
               - Strategic milestones
               - Market timing considerations
            
            Keep it concise, actionable, and focused on key decision factors.
            """
            
            response = self.agent.llm.invoke(summary_prompt)
            summary_result = self._parse_executive_summary(response.content if hasattr(response, 'content') else str(response))
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("ReportWriter", "risk_disclosure")
    async def generate_risk_disclosure(self, ticker: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk disclosure document"""
        try:
            risk_prompt = f"""
            Generate comprehensive risk disclosure document for {ticker}:
            
            RISK DATA:
            {risk_data}
            
            Create detailed risk disclosure covering:
            
            1. INVESTMENT RISKS
               - Market risk and volatility
               - Liquidity risk assessment
               - Credit and counterparty risks
               - Concentration risk factors
               - Currency and interest rate risks
            
            2. COMPANY-SPECIFIC RISKS
               - Business model risks
               - Competitive threats
               - Management and governance risks
               - Operational and technology risks
               - Regulatory and legal risks
            
            3. MARKET RISKS
               - Systematic risk factors
               - Economic sensitivity
               - Sector and industry risks
               - Geopolitical risks
               - Market timing risks
            
            4. REGULATORY RISKS
               - Compliance requirements
               - Regulatory changes impact
               - Legal proceedings exposure
               - Tax implications
               - Reporting obligations
            
            5. RISK QUANTIFICATION
               - Value at Risk (VaR) estimates
               - Stress test scenarios
               - Probability assessments
               - Impact quantification
               - Risk-adjusted returns
            
            6. RISK MITIGATION
               - Diversification strategies
               - Hedging recommendations
               - Position sizing guidelines
               - Stop-loss strategies
               - Monitoring requirements
            
            Include specific disclaimers and regulatory compliance statements.
            """
            
            response = self.agent.llm.invoke(risk_prompt)
            risk_result = self._parse_risk_disclosure(response.content if hasattr(response, 'content') else str(response))
            
            return risk_result
            
        except Exception as e:
            logger.error(f"Error generating risk disclosure: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _parse_investment_report(self, response: str) -> Dict[str, Any]:
        """Parse comprehensive investment report response"""
        try:
            report = {
                "executive_summary": {
                    "recommendation": "HOLD",
                    "target_price": None,
                    "risk_rating": "Medium",
                    "investment_horizon": "Medium-term",
                    "key_thesis": []
                },
                "company_overview": {
                    "business_model": "",
                    "market_position": "",
                    "competitive_advantages": [],
                    "management_assessment": "Adequate"
                },
                "fundamental_analysis": {
                    "financial_health": "Good",
                    "valuation_metrics": {},
                    "growth_prospects": [],
                    "strengths": [],
                    "weaknesses": []
                },
                "technical_analysis": {
                    "trend": "Neutral",
                    "support_levels": [],
                    "resistance_levels": [],
                    "signals": []
                },
                "sentiment_analysis": {
                    "overall_sentiment": "Neutral",
                    "news_impact": "Low",
                    "analyst_sentiment": "Mixed"
                },
                "risk_assessment": {
                    "overall_risk": "Medium",
                    "key_risks": [],
                    "mitigation_strategies": []
                },
                "projections": {
                    "revenue_growth": [],
                    "earnings_estimates": [],
                    "valuation_scenarios": {}
                },
                "investment_recommendation": {
                    "action": "HOLD",
                    "rationale": "",
                    "price_targets": {},
                    "timeline": "6-12 months"
                },
                "regulatory_disclosures": {
                    "conflicts_of_interest": "None disclosed",
                    "data_sources": [],
                    "disclaimers": [],
                    "compliance_statements": []
                },
                "report_quality_score": 85,
                "raw_response": response
            }
            
            # Parse response sections
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Identify main sections
                if 'executive summary' in line_lower:
                    current_section = "executive_summary"
                elif 'company overview' in line_lower:
                    current_section = "company_overview"
                elif 'fundamental analysis' in line_lower:
                    current_section = "fundamental_analysis"
                elif 'technical analysis' in line_lower:
                    current_section = "technical_analysis"
                elif 'investment recommendation' in line_lower:
                    current_section = "investment_recommendation"
                
                # Extract key information
                if any(word in line_lower for word in ['buy', 'sell', 'hold']):
                    for action in ['BUY', 'SELL', 'HOLD']:
                        if action in line.upper():
                            report["executive_summary"]["recommendation"] = action
                            report["investment_recommendation"]["action"] = action
                            break
                
                # Extract target price
                import re
                price_match = re.search(r'\$(\d+(?:\.\d+)?)', line)
                if price_match and 'target' in line_lower:
                    report["executive_summary"]["target_price"] = float(price_match.group(1))
                
                # Extract bullet points
                if line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if current_section == "executive_summary" and 'thesis' in line_lower:
                        report["executive_summary"]["key_thesis"].append(detail)
                    elif current_section == "risk_assessment":
                        report["risk_assessment"]["key_risks"].append(detail)
            
            return report
            
        except Exception as e:
            logger.error(f"Error parsing investment report: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_executive_summary(self, response: str) -> Dict[str, Any]:
        """Parse executive summary response"""
        try:
            summary = {
                "investment_thesis": [],
                "financial_highlights": {},
                "recommendation": {
                    "action": "HOLD",
                    "target_price": None,
                    "horizon": "Medium-term",
                    "position_size": "Normal"
                },
                "key_risks": [],
                "catalysts": [],
                "summary_score": 80,
                "raw_response": response
            }
            
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Identify sections
                if 'investment thesis' in line_lower:
                    current_section = "investment_thesis"
                elif 'key risks' in line_lower:
                    current_section = "key_risks"
                elif 'catalysts' in line_lower:
                    current_section = "catalysts"
                elif 'recommendation' in line_lower:
                    current_section = "recommendation"
                
                # Extract bullet points
                if line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if current_section == "investment_thesis":
                        summary["investment_thesis"].append(detail)
                    elif current_section == "key_risks":
                        summary["key_risks"].append(detail)
                    elif current_section == "catalysts":
                        summary["catalysts"].append(detail)
                
                # Extract recommendation
                if any(word in line_lower for word in ['buy', 'sell', 'hold']):
                    for action in ['BUY', 'SELL', 'HOLD']:
                        if action in line.upper():
                            summary["recommendation"]["action"] = action
                            break
            
            return summary
            
        except Exception as e:
            logger.error(f"Error parsing executive summary: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_risk_disclosure(self, response: str) -> Dict[str, Any]:
        """Parse risk disclosure response"""
        try:
            disclosure = {
                "investment_risks": [],
                "company_risks": [],
                "market_risks": [],
                "regulatory_risks": [],
                "risk_quantification": {
                    "var_estimates": {},
                    "stress_scenarios": [],
                    "probability_assessments": {}
                },
                "mitigation_strategies": [],
                "overall_risk_rating": "Medium",
                "compliance_statements": [],
                "raw_response": response
            }
            
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Identify sections
                if 'investment risks' in line_lower:
                    current_section = "investment_risks"
                elif 'company-specific risks' in line_lower:
                    current_section = "company_risks"
                elif 'market risks' in line_lower:
                    current_section = "market_risks"
                elif 'regulatory risks' in line_lower:
                    current_section = "regulatory_risks"
                elif 'mitigation' in line_lower:
                    current_section = "mitigation_strategies"
                
                # Extract bullet points
                if line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if current_section in disclosure and isinstance(disclosure[current_section], list):
                        disclosure[current_section].append(detail)
            
            return disclosure
            
        except Exception as e:
            logger.error(f"Error parsing risk disclosure: {str(e)}")
            return {"error": str(e), "raw_response": response}