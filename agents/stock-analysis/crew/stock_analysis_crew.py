import logging
import asyncio
from crewai import Crew, Process
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agents.stock_analyst import StockAnalystAgent
from .agents.news_researcher import NewsResearcherAgent
from .agents.technical_analyst import TechnicalAnalystAgent
from .agents.risk_assessor import RiskAssessorAgent
from .agents.report_writer import ReportWriterAgent
from .tasks import StockAnalysisTasks

from models.model_manager import ModelManager
from tools.market_data import MarketDataTool, HistoricalDataTool, MarketScreenerTool, AlphaVantageDataTool
from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class StockAnalysisCrew:
    """CrewAI orchestrated stock analysis system with specialized agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.observability = get_observability_manager()
        self.model_manager = ModelManager(config)
        self.tasks = StockAnalysisTasks()
        
        # Initialize tools
        self.tools = [
            MarketDataTool(),
            HistoricalDataTool(),
            MarketScreenerTool(),
            AlphaVantageDataTool()
        ]
        
        # Initialize agents
        self._initialize_agents()
        
        # Initialize crew
        self._initialize_crew()
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        try:
            # Get LLM for agents
            primary_llm = self.model_manager.get_model("financial_reasoning")
            
            # Initialize all agents
            self.stock_analyst = StockAnalystAgent(primary_llm, self.tools)
            self.news_researcher = NewsResearcherAgent(primary_llm, self.tools)
            self.technical_analyst = TechnicalAnalystAgent(primary_llm, self.tools)
            self.risk_assessor = RiskAssessorAgent(primary_llm, self.tools)
            self.report_writer = ReportWriterAgent(primary_llm, self.tools)
            
            logger.info("Successfully initialized all specialized agents")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    def _initialize_crew(self):
        """Initialize CrewAI crew with agents and process"""
        try:
            self.agents = [
                self.stock_analyst.agent,
                self.news_researcher.agent,
                self.technical_analyst.agent,
                self.risk_assessor.agent,
                self.report_writer.agent
            ]
            
            logger.info("Successfully initialized CrewAI system")
            
        except Exception as e:
            logger.error(f"Error initializing crew: {str(e)}")
            raise
    
    @track_agent_performance("StockAnalysisCrew", "comprehensive_analysis")
    async def analyze_stock(self, ticker: str, analysis_type: str = "comprehensive",
                          client_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive stock analysis using all agents"""
        try:
            # Log compliance action for analysis
            self.observability.log_compliance_action(
                agent_name="StockAnalysisCrew",
                action_type="comprehensive_stock_analysis",
                input_data={"ticker": ticker, "analysis_type": analysis_type},
                output_data={"crew_size": len(self.agents)},
                risk_level="medium",
                decision_reasoning=f"Performing comprehensive multi-agent analysis for {ticker}"
            )
            
            # Step 1: Gather market data
            market_data = await self._gather_market_data(ticker)
            
            # Step 2: Perform parallel analysis by specialized agents
            analysis_results = await self._perform_parallel_analysis(ticker, market_data)
            
            # Step 3: Generate comprehensive report
            final_report = await self._generate_final_report(ticker, analysis_results)
            
            # Step 4: Validate compliance if client profile provided
            if client_profile:
                compliance_validation = await self._validate_compliance(final_report, client_profile)
                final_report["compliance_validation"] = compliance_validation
            
            # Step 5: Add metadata and tracking
            final_report["metadata"] = {
                "ticker": ticker,
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_type": analysis_type,
                "agents_used": [agent.role for agent in self.agents],
                "crew_version": "v1.0",
                "compliance_validated": client_profile is not None
            }
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error in comprehensive stock analysis: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _gather_market_data(self, ticker: str) -> Dict[str, Any]:
        """Gather all necessary market data for analysis"""
        try:
            # Gather different types of market data in parallel
            market_data_tool = MarketDataTool()
            historical_tool = HistoricalDataTool()
            alpha_vantage_tool = AlphaVantageDataTool()
            
            # Get current market data
            current_data = market_data_tool.get_stock_data(ticker)
            
            # Get historical data for different periods
            historical_1y = historical_tool._run(f"{ticker},1y")
            historical_3m = historical_tool._run(f"{ticker},3mo")
            
            # Get fundamental data
            fundamental_data = alpha_vantage_tool._run(ticker)
            
            # Mock news data (in production, this would come from news APIs)
            news_data = [
                {
                    "title": f"Recent news for {ticker}",
                    "source": "Financial News",
                    "date": datetime.now().isoformat(),
                    "snippet": f"Latest developments for {ticker} stock"
                }
            ]
            
            return {
                "current_data": current_data.__dict__ if current_data else {},
                "historical_1y": historical_1y,
                "historical_3m": historical_3m,
                "fundamental_data": fundamental_data,
                "news_data": news_data,
                "options_data": {},  # Placeholder for options data
                "earnings_data": {},  # Placeholder for earnings data
                "analyst_coverage": []  # Placeholder for analyst data
            }
            
        except Exception as e:
            logger.error(f"Error gathering market data: {str(e)}")
            return {"error": str(e)}
    
    async def _perform_parallel_analysis(self, ticker: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform parallel analysis using all specialized agents"""
        try:
            # Create analysis tasks
            tasks = []
            
            # Fundamental analysis
            if market_data.get("fundamental_data"):
                fundamental_task = asyncio.create_task(
                    self.stock_analyst.analyze_fundamentals(
                        ticker, 
                        market_data["fundamental_data"]
                    )
                )
                tasks.append(("fundamental", fundamental_task))
            
            # Technical analysis
            if market_data.get("historical_1y"):
                technical_task = asyncio.create_task(
                    self.technical_analyst.perform_technical_analysis(
                        ticker,
                        {"prices": [], "historical_data": market_data["historical_1y"]}
                    )
                )
                tasks.append(("technical", technical_task))
            
            # Sentiment analysis
            if market_data.get("news_data"):
                sentiment_task = asyncio.create_task(
                    self.news_researcher.analyze_market_sentiment(
                        ticker,
                        market_data["news_data"]
                    )
                )
                tasks.append(("sentiment", sentiment_task))
            
            # Risk assessment
            portfolio_data = {"total_value": 1000000, "positions": {ticker: 0.1}}
            risk_task = asyncio.create_task(
                self.risk_assessor.perform_risk_assessment(
                    ticker,
                    portfolio_data,
                    market_data,
                    {"fundamental": {}, "technical": {}}
                )
            )
            tasks.append(("risk", risk_task))
            
            # Execute all tasks in parallel
            results = {}
            for task_name, task in tasks:
                try:
                    result = await task
                    results[task_name] = result
                except Exception as e:
                    logger.error(f"Error in {task_name} analysis: {str(e)}")
                    results[task_name] = {"error": str(e), "status": "failed"}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel analysis: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_final_report(self, ticker: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final comprehensive report"""
        try:
            # Use report writer to synthesize all analysis
            final_report = await self.report_writer.generate_investment_report(
                ticker, 
                analysis_results
            )
            
            # Add individual analysis results
            final_report["detailed_analysis"] = analysis_results
            
            # Generate executive summary
            executive_summary = await self.report_writer.generate_executive_summary(
                ticker,
                analysis_results
            )
            final_report["executive_summary"] = executive_summary
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error generating final report: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def _validate_compliance(self, investment_report: Dict[str, Any], 
                                 client_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate regulatory compliance"""
        try:
            compliance_result = await self.risk_assessor.validate_regulatory_compliance(
                investment_report,
                client_profile
            )
            
            return compliance_result
            
        except Exception as e:
            logger.error(f"Error in compliance validation: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("StockAnalysisCrew", "portfolio_analysis")
    async def analyze_portfolio(self, portfolio: Dict[str, Any], 
                              benchmark: str = "SPY") -> Dict[str, Any]:
        """Perform portfolio-level risk and performance analysis"""
        try:
            # Get benchmark data
            benchmark_data = await self._gather_market_data(benchmark)
            
            # Perform portfolio risk assessment
            portfolio_risk = await self.risk_assessor.assess_portfolio_risk(
                portfolio,
                benchmark_data
            )
            
            return {
                "portfolio_risk": portfolio_risk,
                "benchmark": benchmark,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    async def quick_analysis(self, ticker: str) -> Dict[str, Any]:
        """Perform quick analysis with executive summary"""
        try:
            # Gather essential data
            market_data = await self._gather_market_data(ticker)
            
            # Perform fundamental analysis only
            fundamental_result = await self.stock_analyst.analyze_fundamentals(
                ticker,
                market_data.get("fundamental_data", {})
            )
            
            # Generate executive summary
            executive_summary = await self.report_writer.generate_executive_summary(
                ticker,
                {"fundamental": fundamental_result}
            )
            
            return {
                "ticker": ticker,
                "executive_summary": executive_summary,
                "fundamental_analysis": fundamental_result,
                "analysis_type": "quick",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in quick analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get status of all agents and crew"""
        return {
            "agents": {
                "stock_analyst": {"status": "active", "role": "Senior Stock Analyst"},
                "news_researcher": {"status": "active", "role": "Financial News Researcher"},
                "technical_analyst": {"status": "active", "role": "Technical Analysis Expert"},
                "risk_assessor": {"status": "active", "role": "Risk Management Specialist"},
                "report_writer": {"status": "active", "role": "Investment Research Report Writer"}
            },
            "tools_available": len(self.tools),
            "model_manager": "active",
            "observability": "enabled",
            "crew_version": "v1.0",
            "last_updated": datetime.now().isoformat()
        }