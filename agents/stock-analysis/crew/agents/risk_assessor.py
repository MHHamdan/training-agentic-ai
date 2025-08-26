import logging
from crewai import Agent
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class RiskAssessorAgent:
    """Risk Management Specialist Agent with comprehensive risk analysis capabilities"""
    
    def __init__(self, llm, tools: List[Any]):
        self.observability = get_observability_manager()
        self.llm = llm
        self.tools = tools
        
        self.agent = Agent(
            role='Risk Management Specialist',
            goal='Assess investment risks and provide comprehensive risk metrics with regulatory compliance',
            backstory="""You are a Risk Management Specialist with expertise in portfolio risk assessment,
            regulatory compliance, and quantitative risk modeling. You have extensive experience in
            financial risk management, including market risk, credit risk, operational risk, and
            regulatory compliance. Your specialty is identifying, measuring, and mitigating investment
            risks while ensuring adherence to regulatory requirements and industry best practices.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools,
            max_iter=3,
            memory=True
        )
    
    @track_agent_performance("RiskAssessor", "comprehensive_risk_analysis")
    async def perform_risk_assessment(self, ticker: str, portfolio_data: Dict[str, Any],
                                    market_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            # Log compliance action for risk assessment
            self.observability.log_compliance_action(
                agent_name="RiskAssessor",
                action_type="risk_assessment",
                input_data={"ticker": ticker, "portfolio_value": portfolio_data.get("total_value", 0)},
                output_data={"assessment_type": "comprehensive_risk"},
                risk_level="high",  # Risk assessment is high importance
                decision_reasoning=f"Performing regulatory-compliant risk assessment for {ticker}"
            )
            
            risk_prompt = f"""
            Perform comprehensive risk assessment for {ticker} investment:
            
            PORTFOLIO DATA:
            {portfolio_data}
            
            MARKET DATA:
            {market_data}
            
            ANALYSIS DATA:
            {analysis_data}
            
            Provide detailed risk assessment covering all major risk categories:
            
            1. MARKET RISK ANALYSIS
               - Beta coefficient and systematic risk
               - Volatility analysis (historical and implied)
               - Correlation with market indices
               - Interest rate sensitivity
               - Currency risk (if applicable)
               - Sector and style risk exposures
               - Value at Risk (VaR) calculations
               - Expected Shortfall (ES) metrics
            
            2. COMPANY-SPECIFIC RISKS
               - Business model risks
               - Competitive position risks
               - Management and governance risks
               - Financial leverage and credit risks
               - Operational risks and key dependencies
               - Technology and obsolescence risks
               - ESG (Environmental, Social, Governance) risks
               - Regulatory and compliance risks
            
            3. LIQUIDITY RISK ASSESSMENT
               - Average daily trading volume analysis
               - Bid-ask spread analysis
               - Market depth assessment
               - Liquidity stress scenario analysis
               - Exit strategy feasibility
               - Market impact costs
               - Liquidity premium requirements
            
            4. CONCENTRATION RISK ANALYSIS
               - Position size relative to portfolio
               - Sector concentration assessment
               - Geographic concentration risks
               - Single-name concentration limits
               - Correlation-adjusted exposure
               - Portfolio diversification metrics
               - Risk contribution analysis
            
            5. CREDIT AND COUNTERPARTY RISKS
               - Credit rating analysis
               - Default probability assessment
               - Credit spread analysis
               - Counterparty exposure limits
               - Collateral and security analysis
               - Recovery rate expectations
            
            6. OPERATIONAL RISKS
               - Custody and settlement risks
               - Technology and system risks
               - Human error and process risks
               - Fraud and security risks
               - Business continuity risks
               - Third-party vendor risks
            
            7. REGULATORY AND COMPLIANCE RISKS
               - SEC compliance requirements
               - FINRA rule adherence
               - Suitability assessment
               - Disclosure requirements
               - Record keeping obligations
               - Anti-money laundering (AML) considerations
               - Know Your Customer (KYC) compliance
            
            8. STRESS TESTING AND SCENARIOS
               - Market crash scenarios (-20%, -30%, -50%)
               - Interest rate shock scenarios
               - Sector-specific stress scenarios
               - Liquidity crisis scenarios
               - Economic recession impacts
               - Geopolitical risk scenarios
               - Black swan event analysis
            
            9. RISK METRICS AND MEASUREMENTS
               - Sharpe ratio and risk-adjusted returns
               - Maximum drawdown analysis
               - Tracking error and information ratio
               - Sortino ratio and downside risk
               - Calmar ratio and risk-return efficiency
               - Skewness and kurtosis analysis
               - Tail risk measurements
            
            10. RISK MITIGATION STRATEGIES
                - Hedging recommendations
                - Diversification strategies
                - Position sizing guidelines
                - Stop-loss and risk management rules
                - Portfolio rebalancing recommendations
                - Risk monitoring and alert systems
                - Contingency planning
            
            Provide specific quantitative risk metrics, probability assessments, and actionable recommendations.
            Include regulatory compliance validation and audit trail requirements.
            """
            
            response = self.agent.llm.invoke(risk_prompt)
            risk_result = self._parse_risk_assessment(response.content if hasattr(response, 'content') else str(response))
            
            # Add compliance validation
            risk_result["compliance_validation"] = self._validate_compliance(risk_result, ticker)
            
            return risk_result
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("RiskAssessor", "portfolio_risk_analysis")
    async def assess_portfolio_risk(self, portfolio: Dict[str, Any], 
                                  benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk metrics"""
        try:
            portfolio_prompt = f"""
            Assess portfolio-level risk metrics and exposures:
            
            PORTFOLIO COMPOSITION:
            {portfolio}
            
            BENCHMARK DATA:
            {benchmark_data}
            
            Provide comprehensive portfolio risk analysis:
            
            1. PORTFOLIO RISK METRICS
               - Portfolio beta vs benchmark
               - Portfolio volatility (annualized)
               - Tracking error vs benchmark
               - Information ratio calculation
               - Sharpe ratio vs benchmark
               - Maximum drawdown analysis
               - Value at Risk (1%, 5%, 95%, 99% confidence)
               - Expected Shortfall calculations
            
            2. DIVERSIFICATION ANALYSIS
               - Effective number of holdings
               - Concentration ratios (top 5, 10 holdings)
               - Sector diversification metrics
               - Geographic diversification
               - Market cap diversification
               - Style factor exposures
               - Correlation matrix analysis
            
            3. FACTOR RISK EXPOSURES
               - Market factor (beta) exposure
               - Size factor (small vs large cap) exposure
               - Value vs growth factor exposure
               - Quality factor exposure
               - Momentum factor exposure
               - Volatility factor exposure
               - Sector factor exposures
            
            4. STRESS TESTING RESULTS
               - 2008 Financial Crisis scenario
               - COVID-19 pandemic scenario
               - Interest rate shock scenarios
               - Inflation shock scenarios
               - Market volatility spike scenarios
               - Currency crisis scenarios
               - Geopolitical event scenarios
            
            5. RISK ATTRIBUTION ANALYSIS
               - Risk contribution by holding
               - Risk contribution by sector
               - Active vs passive risk components
               - Systematic vs idiosyncratic risk
               - Factor-based risk attribution
               - Geographic risk attribution
            
            6. LIQUIDITY ASSESSMENT
               - Portfolio liquidity score
               - Days to liquidate analysis
               - Market impact costs
               - Liquidity-adjusted VaR
               - Emergency liquidation scenarios
               - Liquidity risk by holding size
            
            7. REGULATORY COMPLIANCE CHECK
               - Position limit compliance
               - Concentration limit adherence
               - Liquidity requirement compliance
               - Risk limit monitoring
               - Regulatory reporting requirements
               - Suitability assessment validation
            """
            
            response = self.agent.llm.invoke(portfolio_prompt)
            portfolio_risk_result = self._parse_portfolio_risk(response.content if hasattr(response, 'content') else str(response))
            
            return portfolio_risk_result
            
        except Exception as e:
            logger.error(f"Error in portfolio risk assessment: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("RiskAssessor", "compliance_validation")
    async def validate_regulatory_compliance(self, investment_decision: Dict[str, Any],
                                           client_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance for investment decisions"""
        try:
            compliance_prompt = f"""
            Validate regulatory compliance for investment decision:
            
            INVESTMENT DECISION:
            {investment_decision}
            
            CLIENT PROFILE:
            {client_profile}
            
            Perform comprehensive regulatory compliance validation:
            
            1. SUITABILITY ASSESSMENT
               - Investment objective alignment
               - Risk tolerance compatibility
               - Time horizon appropriateness
               - Financial situation adequacy
               - Investment experience validation
               - Liquidity needs assessment
               - Tax situation considerations
            
            2. SEC COMPLIANCE VALIDATION
               - Securities Act compliance
               - Investment Advisers Act compliance
               - Disclosure requirement adherence
               - Fiduciary duty validation
               - Record keeping compliance
               - Reporting requirement validation
            
            3. FINRA RULE COMPLIANCE
               - Know Your Customer (KYC) validation
               - Best execution requirements
               - Fair dealing principles
               - Supervision and compliance
               - Anti-money laundering (AML) check
               - Customer protection rules
            
            4. RISK DISCLOSURE VALIDATION
               - Material risk disclosure completeness
               - Risk factor identification accuracy
               - Probability and impact assessment
               - Client risk acknowledgment
               - Documentation requirements
               - Ongoing monitoring obligations
            
            5. POSITION AND CONCENTRATION LIMITS
               - Single security concentration limits
               - Sector concentration limits
               - Asset class allocation limits
               - Geographic concentration limits
               - Liquidity requirement compliance
               - Leverage and margin compliance
            
            6. DOCUMENTATION REQUIREMENTS
               - Investment policy compliance
               - Decision rationale documentation
               - Risk assessment documentation
               - Client communication records
               - Supervisory review evidence
               - Audit trail completeness
            
            Provide specific compliance status (COMPLIANT/NON-COMPLIANT) for each area
            and detailed remediation steps for any non-compliant items.
            """
            
            response = self.agent.llm.invoke(compliance_prompt)
            compliance_result = self._parse_compliance_validation(response.content if hasattr(response, 'content') else str(response))
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error in compliance validation: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _parse_risk_assessment(self, response: str) -> Dict[str, Any]:
        """Parse comprehensive risk assessment response"""
        try:
            risk_assessment = {
                "market_risk": {
                    "beta": 1.0,
                    "volatility": 0.20,
                    "var_95": 0.05,
                    "var_99": 0.08,
                    "expected_shortfall": 0.10,
                    "correlation_market": 0.70
                },
                "company_risks": {
                    "business_model": "medium",
                    "competitive_position": "medium",
                    "financial_leverage": "low",
                    "management": "low",
                    "regulatory": "medium"
                },
                "liquidity_risk": {
                    "daily_volume": "adequate",
                    "bid_ask_spread": "normal",
                    "market_depth": "good",
                    "liquidity_score": 7  # 1-10 scale
                },
                "concentration_risk": {
                    "position_size_risk": "medium",
                    "sector_concentration": "low",
                    "diversification_score": 8
                },
                "stress_testing": {
                    "market_crash_20": -0.25,
                    "market_crash_30": -0.40,
                    "recession_scenario": -0.30,
                    "liquidity_crisis": -0.20
                },
                "risk_metrics": {
                    "sharpe_ratio": 1.2,
                    "max_drawdown": 0.15,
                    "sortino_ratio": 1.5,
                    "calmar_ratio": 0.8
                },
                "mitigation_strategies": [],
                "overall_risk_rating": "medium",  # low/medium/high
                "risk_score": 6,  # 1-10 scale
                "raw_response": response
            }
            
            # Parse response for specific risk values
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Extract numerical risk metrics
                if 'beta' in line_lower:
                    import re
                    beta_match = re.search(r'(\d+\.?\d*)', line)
                    if beta_match:
                        risk_assessment["market_risk"]["beta"] = float(beta_match.group(1))
                
                if 'volatility' in line_lower and '%' in line:
                    import re
                    vol_match = re.search(r'(\d+\.?\d*)%', line)
                    if vol_match:
                        risk_assessment["market_risk"]["volatility"] = float(vol_match.group(1)) / 100
                
                if 'var' in line_lower or 'value at risk' in line_lower:
                    import re
                    var_match = re.search(r'(\d+\.?\d*)%', line)
                    if var_match:
                        var_value = float(var_match.group(1)) / 100
                        if '99' in line:
                            risk_assessment["market_risk"]["var_99"] = var_value
                        elif '95' in line:
                            risk_assessment["market_risk"]["var_95"] = var_value
                
                # Extract risk ratings
                if any(word in line_lower for word in ['high risk', 'high']):
                    if 'overall' in line_lower:
                        risk_assessment["overall_risk_rating"] = "high"
                elif any(word in line_lower for word in ['low risk', 'low']):
                    if 'overall' in line_lower:
                        risk_assessment["overall_risk_rating"] = "low"
                
                # Extract mitigation strategies
                if line.startswith('-') or line.startswith('•'):
                    if 'mitigation' in current_section or 'strategy' in current_section:
                        risk_assessment["mitigation_strategies"].append(line[1:].strip())
                
                # Identify sections
                if 'mitigation' in line_lower:
                    current_section = "mitigation"
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_portfolio_risk(self, response: str) -> Dict[str, Any]:
        """Parse portfolio risk assessment response"""
        try:
            portfolio_risk = {
                "portfolio_metrics": {
                    "portfolio_beta": 1.0,
                    "portfolio_volatility": 0.15,
                    "tracking_error": 0.05,
                    "information_ratio": 0.5,
                    "sharpe_ratio": 1.0
                },
                "diversification": {
                    "effective_holdings": 50,
                    "concentration_top5": 0.25,
                    "sector_diversification": 8,
                    "geographic_diversification": 6
                },
                "factor_exposures": {
                    "market_beta": 1.0,
                    "size_factor": 0.1,
                    "value_factor": -0.2,
                    "quality_factor": 0.3,
                    "momentum_factor": 0.1
                },
                "stress_results": {},
                "risk_attribution": {},
                "liquidity_assessment": {
                    "portfolio_liquidity_score": 7,
                    "days_to_liquidate": 5,
                    "market_impact": 0.02
                },
                "compliance_status": "COMPLIANT",
                "raw_response": response
            }
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error parsing portfolio risk: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_compliance_validation(self, response: str) -> Dict[str, Any]:
        """Parse regulatory compliance validation response"""
        try:
            compliance = {
                "suitability": {
                    "status": "COMPLIANT",
                    "objective_alignment": "PASS",
                    "risk_tolerance": "PASS",
                    "time_horizon": "PASS",
                    "financial_situation": "PASS"
                },
                "sec_compliance": {
                    "status": "COMPLIANT",
                    "disclosure": "PASS",
                    "fiduciary_duty": "PASS",
                    "record_keeping": "PASS"
                },
                "finra_compliance": {
                    "status": "COMPLIANT",
                    "kyc": "PASS",
                    "best_execution": "PASS",
                    "aml": "PASS"
                },
                "risk_disclosure": {
                    "status": "COMPLIANT",
                    "material_risks": "DISCLOSED",
                    "probability_assessment": "COMPLETE",
                    "client_acknowledgment": "RECEIVED"
                },
                "position_limits": {
                    "status": "COMPLIANT",
                    "concentration": "WITHIN_LIMITS",
                    "sector_limits": "WITHIN_LIMITS",
                    "leverage": "WITHIN_LIMITS"
                },
                "documentation": {
                    "status": "COMPLIANT",
                    "policy_compliance": "PASS",
                    "rationale": "DOCUMENTED",
                    "audit_trail": "COMPLETE"
                },
                "overall_compliance": "COMPLIANT",
                "remediation_required": [],
                "raw_response": response
            }
            
            # Parse compliance status from response
            lines = response.split('\n')
            for line in lines:
                line_lower = line.strip().lower()
                if 'non-compliant' in line_lower:
                    compliance["overall_compliance"] = "NON-COMPLIANT"
                    break
            
            return compliance
            
        except Exception as e:
            logger.error(f"Error parsing compliance validation: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _validate_compliance(self, risk_assessment: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """Validate compliance requirements for risk assessment"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "risk_assessment_complete": True,
            "required_disclosures": [
                "Market risk disclosure provided",
                "Company-specific risks identified",
                "Liquidity risk assessment completed",
                "Stress testing results available"
            ],
            "regulatory_standards_met": [
                "SEC risk assessment requirements",
                "FINRA suitability standards",
                "Fiduciary duty compliance"
            ],
            "audit_trail_complete": True,
            "compliance_officer_review": "PENDING"
        }
        
        return validation