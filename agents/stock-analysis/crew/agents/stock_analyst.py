import logging
from crewai import Agent
from typing import Dict, Any, List, Optional

from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class StockAnalystAgent:
    """Senior Stock Analyst Agent with AgentOps integration"""
    
    def __init__(self, llm, tools: List[Any]):
        self.observability = get_observability_manager()
        self.llm = llm
        self.tools = tools
        
        self.agent = Agent(
            role='Senior Stock Analyst',
            goal='Analyze stock fundamentals and provide comprehensive investment insights',
            backstory="""You are a Senior Stock Analyst with over 15 years of experience in financial markets. 
            You specialize in fundamental analysis, financial statement evaluation, and market trend analysis.
            Your expertise includes identifying undervalued stocks, assessing company financial health,
            and providing data-driven investment recommendations. You always consider risk factors
            and regulatory compliance in your analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools,
            max_iter=3,
            memory=True
        )
    
    @track_agent_performance("StockAnalyst", "fundamental_analysis")
    async def analyze_fundamentals(self, ticker: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive fundamental analysis"""
        try:
            # Log compliance action for financial analysis
            self.observability.log_compliance_action(
                agent_name="StockAnalyst",
                action_type="fundamental_analysis",
                input_data={"ticker": ticker, "data_type": "fundamental"},
                output_data={"analysis_type": "comprehensive"},
                risk_level="medium",
                decision_reasoning=f"Performing fundamental analysis for investment decision on {ticker}"
            )
            
            analysis_prompt = f"""
            Perform comprehensive fundamental analysis for {ticker} based on the following financial data:
            
            {financial_data}
            
            Please provide detailed analysis covering:
            
            1. FINANCIAL HEALTH ASSESSMENT
               - Revenue growth trends
               - Profit margin analysis
               - Debt-to-equity ratios
               - Cash flow evaluation
               - Return on equity (ROE) and return on assets (ROA)
            
            2. VALUATION ANALYSIS
               - P/E ratio comparison to industry average
               - Price-to-book ratio assessment
               - Price-to-sales ratio evaluation
               - PEG ratio calculation
               - Intrinsic value estimation
            
            3. COMPETITIVE ANALYSIS
               - Market position assessment
               - Competitive advantages/moats
               - Industry trends impact
               - Competitive threats
            
            4. GROWTH PROSPECTS
               - Revenue growth sustainability
               - Market expansion opportunities
               - New product/service potential
               - Management effectiveness
            
            5. RISK FACTORS
               - Financial risks
               - Operational risks
               - Market risks
               - Regulatory risks
            
            6. INVESTMENT RECOMMENDATION
               - Buy/Hold/Sell recommendation
               - Target price range
               - Investment time horizon
               - Position size recommendation
            
            Provide specific metrics, ratios, and quantitative analysis where possible.
            """
            
            # Use the agent to generate analysis
            response = self.agent.llm.invoke(analysis_prompt)
            
            # Parse and structure the response
            analysis_result = self._parse_fundamental_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("StockAnalyst", "company_analysis")
    async def analyze_company_profile(self, ticker: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company profile and business model"""
        try:
            profile_prompt = f"""
            Analyze the company profile and business model for {ticker}:
            
            {company_data}
            
            Provide analysis on:
            
            1. BUSINESS MODEL
               - Primary revenue sources
               - Business model sustainability
               - Scalability assessment
               - Competitive positioning
            
            2. MANAGEMENT ASSESSMENT
               - Leadership track record
               - Strategic vision execution
               - Corporate governance
               - Insider ownership
            
            3. MARKET OPPORTUNITY
               - Total addressable market (TAM)
               - Market share and growth potential
               - Industry dynamics
               - Regulatory environment
            
            4. SWOT ANALYSIS
               - Strengths
               - Weaknesses
               - Opportunities
               - Threats
            
            5. ESG CONSIDERATIONS
               - Environmental impact
               - Social responsibility
               - Governance practices
               - Sustainability initiatives
            """
            
            response = self.agent.llm.invoke(profile_prompt)
            result = self._parse_company_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in company analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _parse_fundamental_analysis(self, response: str) -> Dict[str, Any]:
        """Parse fundamental analysis response into structured format"""
        try:
            analysis = {
                "financial_health": {"score": 0, "details": []},
                "valuation": {"score": 0, "metrics": {}, "details": []},
                "competitive_position": {"score": 0, "details": []},
                "growth_prospects": {"score": 0, "details": []},
                "risk_factors": {"level": "medium", "factors": []},
                "recommendation": {"action": "HOLD", "target_price": None, "confidence": 50},
                "key_metrics": {},
                "summary": "",
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
                if 'financial health' in line_lower:
                    current_section = "financial_health"
                elif 'valuation' in line_lower:
                    current_section = "valuation"
                elif 'competitive' in line_lower:
                    current_section = "competitive_position"
                elif 'growth' in line_lower:
                    current_section = "growth_prospects"
                elif 'risk' in line_lower:
                    current_section = "risk_factors"
                elif 'recommendation' in line_lower:
                    current_section = "recommendation"
                
                # Extract recommendation
                if current_section == "recommendation":
                    if any(word in line_upper for word in ['BUY', 'SELL', 'HOLD'] for line_upper in [line.upper()]):
                        for action in ['BUY', 'SELL', 'HOLD']:
                            if action in line.upper():
                                analysis["recommendation"]["action"] = action
                                break
                    
                    # Extract target price
                    import re
                    price_match = re.search(r'\$(\d+(?:\.\d+)?)', line)
                    if price_match:
                        analysis["recommendation"]["target_price"] = float(price_match.group(1))
                
                # Add details to appropriate section
                if current_section and line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if current_section in analysis and "details" in analysis[current_section]:
                        analysis[current_section]["details"].append(detail)
                    elif current_section == "risk_factors":
                        analysis["risk_factors"]["factors"].append(detail)
            
            # Generate summary
            analysis["summary"] = f"Analysis completed for fundamental assessment. Recommendation: {analysis['recommendation']['action']}"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing fundamental analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_company_analysis(self, response: str) -> Dict[str, Any]:
        """Parse company analysis response"""
        try:
            analysis = {
                "business_model": {"sustainability": "medium", "scalability": "medium", "details": []},
                "management": {"quality": "medium", "track_record": [], "governance": "adequate"},
                "market_opportunity": {"tam": "unknown", "growth_potential": "medium", "details": []},
                "swot": {"strengths": [], "weaknesses": [], "opportunities": [], "threats": []},
                "esg": {"environmental": "neutral", "social": "neutral", "governance": "neutral"},
                "overall_score": 70,
                "raw_response": response
            }
            
            lines = response.split('\n')
            current_section = ""
            current_swot = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Identify main sections
                if 'business model' in line_lower:
                    current_section = "business_model"
                elif 'management' in line_lower:
                    current_section = "management"
                elif 'market opportunity' in line_lower:
                    current_section = "market_opportunity"
                elif 'swot' in line_lower:
                    current_section = "swot"
                elif 'esg' in line_lower:
                    current_section = "esg"
                
                # SWOT subsections
                if current_section == "swot":
                    if 'strengths' in line_lower:
                        current_swot = "strengths"
                    elif 'weaknesses' in line_lower:
                        current_swot = "weaknesses"
                    elif 'opportunities' in line_lower:
                        current_swot = "opportunities"
                    elif 'threats' in line_lower:
                        current_swot = "threats"
                    elif line.startswith('-') or line.startswith('•'):
                        if current_swot and current_swot in analysis["swot"]:
                            analysis["swot"][current_swot].append(line[1:].strip())
                
                # Add details to other sections
                elif current_section and (line.startswith('-') or line.startswith('•')):
                    detail = line[1:].strip()
                    if current_section in analysis and "details" in analysis[current_section]:
                        analysis[current_section]["details"].append(detail)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing company analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}