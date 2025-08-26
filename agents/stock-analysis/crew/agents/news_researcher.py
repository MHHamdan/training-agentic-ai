import logging
from crewai import Agent
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class NewsResearcherAgent:
    """Financial News Researcher Agent with sentiment analysis capabilities"""
    
    def __init__(self, llm, tools: List[Any]):
        self.observability = get_observability_manager()
        self.llm = llm
        self.tools = tools
        
        self.agent = Agent(
            role='Financial News Researcher',
            goal='Gather and analyze latest market news and sentiment to assess impact on stock performance',
            backstory="""You are a Financial News Researcher with expertise in market sentiment analysis 
            and news impact assessment. You have deep understanding of how news events affect stock prices
            and market psychology. Your specialties include identifying market-moving news, analyzing
            sentiment trends, and correlating news events with price movements. You monitor multiple
            news sources and social media platforms to provide comprehensive market intelligence.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools,
            max_iter=3,
            memory=True
        )
    
    @track_agent_performance("NewsResearcher", "sentiment_analysis")
    async def analyze_market_sentiment(self, ticker: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze market sentiment from news and social media"""
        try:
            # Log compliance action
            self.observability.log_compliance_action(
                agent_name="NewsResearcher",
                action_type="sentiment_analysis",
                input_data={"ticker": ticker, "news_count": len(news_data)},
                output_data={"analysis_type": "market_sentiment"},
                risk_level="low",
                decision_reasoning=f"Analyzing public market sentiment for {ticker} from news sources"
            )
            
            # Prepare news summary for analysis
            news_summary = self._prepare_news_summary(news_data)
            
            sentiment_prompt = f"""
            Analyze the market sentiment for {ticker} based on the following recent news and information:
            
            {news_summary}
            
            Provide comprehensive sentiment analysis including:
            
            1. OVERALL SENTIMENT SCORE
               - Numerical score from -100 (very negative) to +100 (very positive)
               - Confidence level (0-100%)
               - Sentiment trend (improving/declining/stable)
            
            2. SENTIMENT DRIVERS
               - Key positive factors influencing sentiment
               - Key negative factors influencing sentiment
               - Neutral factors with potential impact
            
            3. NEWS IMPACT ANALYSIS
               - Market-moving headlines and their potential impact
               - Earnings-related sentiment
               - Management/strategic developments
               - Industry and sector sentiment
               - Regulatory or legal developments
            
            4. SOCIAL SENTIMENT INDICATORS
               - Social media buzz level (high/medium/low)
               - Retail investor sentiment
               - Professional analyst sentiment
               - Institutional sentiment indicators
            
            5. SENTIMENT RELIABILITY
               - Source credibility assessment
               - News volume and frequency
               - Sentiment consistency across sources
               - Potential bias or manipulation indicators
            
            6. TRADING IMPLICATIONS
               - Short-term price impact prediction
               - Volume impact expectations
               - Volatility implications
               - Options market sentiment indicators
            
            7. RISK ASSESSMENT
               - Sentiment reversal risks
               - News catalyst potential
               - Event-driven volatility risks
               - Market timing considerations
            
            Provide specific examples and quantitative measures where possible.
            """
            
            response = self.agent.llm.invoke(sentiment_prompt)
            sentiment_result = self._parse_sentiment_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("NewsResearcher", "news_impact_assessment")
    async def assess_news_impact(self, ticker: str, news_data: List[Dict[str, Any]], 
                                price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the impact of specific news events on stock price"""
        try:
            news_summary = self._prepare_news_summary(news_data)
            
            impact_prompt = f"""
            Assess the impact of recent news events on {ticker} stock price and trading activity:
            
            NEWS DATA:
            {news_summary}
            
            PRICE DATA:
            {price_data}
            
            Provide detailed impact assessment covering:
            
            1. NEWS-PRICE CORRELATION
               - Direct correlation between specific news and price movements
               - Time lag analysis between news release and price reaction
               - Volume correlation with news events
               - Price volatility changes post-news
            
            2. EVENT IMPACT SCORING
               - Rate each major news event impact (1-10 scale)
               - Positive vs negative news balance
               - Event significance and market attention
               - Duration of impact (immediate/short-term/long-term)
            
            3. MARKET REACTION ANALYSIS
               - Initial market reaction (first hour/day)
               - Sustained impact analysis
               - Reversal patterns after initial reaction
               - Comparison to historical similar events
            
            4. TRADING PATTERN CHANGES
               - Volume spikes correlation with news
               - Institutional vs retail trading patterns
               - Options activity changes
               - Short interest changes
            
            5. FORWARD-LOOKING IMPLICATIONS
               - Anticipated follow-up news or events
               - Earnings guidance impact
               - Strategic direction changes
               - Regulatory or legal implications
            
            6. RISK FACTORS
               - Ongoing news-related risks
               - Potential negative developments
               - Market overreaction risks
               - Fundamental vs sentiment-driven moves
            """
            
            response = self.agent.llm.invoke(impact_prompt)
            impact_result = self._parse_impact_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return impact_result
            
        except Exception as e:
            logger.error(f"Error in news impact assessment: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("NewsResearcher", "earnings_analysis")
    async def analyze_earnings_sentiment(self, ticker: str, earnings_data: Dict[str, Any],
                                       analyst_coverage: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze earnings-related sentiment and analyst reactions"""
        try:
            earnings_prompt = f"""
            Analyze earnings-related sentiment and market reactions for {ticker}:
            
            EARNINGS DATA:
            {earnings_data}
            
            ANALYST COVERAGE:
            {analyst_coverage}
            
            Provide comprehensive earnings sentiment analysis:
            
            1. EARNINGS SURPRISE ANALYSIS
               - Revenue surprise vs expectations
               - EPS surprise vs expectations
               - Guidance surprise vs expectations
               - Quality of earnings assessment
            
            2. MANAGEMENT COMMENTARY SENTIMENT
               - Management tone and confidence
               - Forward guidance sentiment
               - Strategic initiative sentiment
               - Risk disclosure analysis
            
            3. ANALYST SENTIMENT SHIFT
               - Rating changes post-earnings
               - Target price revisions
               - Estimate revisions (upward/downward)
               - Analyst sentiment consensus
            
            4. EARNINGS CALL SENTIMENT
               - Management presentation sentiment
               - Q&A session tone and concerns
               - Analyst question focus areas
               - Management response quality
            
            5. MARKET REACTION ANALYSIS
               - Pre-market reaction to earnings
               - Day-of trading sentiment
               - After-hours reaction analysis
               - Multi-day sentiment evolution
            
            6. SECTOR COMPARISON
               - Earnings performance vs sector peers
               - Relative sentiment vs industry
               - Sector-wide trend implications
               - Competitive positioning impact
            """
            
            response = self.agent.llm.invoke(earnings_prompt)
            earnings_result = self._parse_earnings_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return earnings_result
            
        except Exception as e:
            logger.error(f"Error in earnings analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _prepare_news_summary(self, news_data: List[Dict[str, Any]]) -> str:
        """Prepare news data for analysis"""
        if not news_data:
            return "No recent news data available."
        
        summary = "RECENT NEWS SUMMARY:\n\n"
        
        for i, news_item in enumerate(news_data[:10], 1):  # Limit to 10 most recent
            title = news_item.get('title', 'No title')
            source = news_item.get('source', 'Unknown source')
            date = news_item.get('date', 'Unknown date')
            snippet = news_item.get('snippet', news_item.get('description', 'No description'))
            
            summary += f"{i}. {title}\n"
            summary += f"   Source: {source} | Date: {date}\n"
            summary += f"   Summary: {snippet[:200]}...\n\n"
        
        return summary
    
    def _parse_sentiment_analysis(self, response: str) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        try:
            analysis = {
                "overall_sentiment": {
                    "score": 0,  # -100 to +100
                    "confidence": 50,
                    "trend": "stable",
                    "classification": "neutral"
                },
                "sentiment_drivers": {
                    "positive": [],
                    "negative": [],
                    "neutral": []
                },
                "news_impact": {
                    "market_moving": [],
                    "earnings_related": [],
                    "strategic": [],
                    "regulatory": []
                },
                "social_indicators": {
                    "buzz_level": "medium",
                    "retail_sentiment": "neutral",
                    "analyst_sentiment": "neutral",
                    "institutional_sentiment": "neutral"
                },
                "reliability": {
                    "source_credibility": "high",
                    "consistency": "medium",
                    "bias_indicators": []
                },
                "trading_implications": {
                    "short_term_impact": "neutral",
                    "volume_impact": "medium",
                    "volatility": "medium"
                },
                "raw_response": response
            }
            
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Extract overall sentiment score
                if 'sentiment score' in line_lower or 'overall sentiment' in line_lower:
                    import re
                    score_match = re.search(r'[-+]?\d+', line)
                    if score_match:
                        score = int(score_match.group())
                        analysis["overall_sentiment"]["score"] = max(-100, min(100, score))
                        
                        # Classify sentiment
                        if score > 20:
                            analysis["overall_sentiment"]["classification"] = "positive"
                        elif score < -20:
                            analysis["overall_sentiment"]["classification"] = "negative"
                        else:
                            analysis["overall_sentiment"]["classification"] = "neutral"
                
                # Extract confidence
                if 'confidence' in line_lower:
                    import re
                    conf_match = re.search(r'(\d+)%', line)
                    if conf_match:
                        analysis["overall_sentiment"]["confidence"] = int(conf_match.group(1))
                
                # Identify sections
                if 'sentiment drivers' in line_lower:
                    current_section = "sentiment_drivers"
                elif 'news impact' in line_lower:
                    current_section = "news_impact"
                elif 'social' in line_lower and 'indicators' in line_lower:
                    current_section = "social_indicators"
                elif 'trading implications' in line_lower:
                    current_section = "trading_implications"
                
                # Extract bullet points
                if line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if current_section == "sentiment_drivers":
                        if 'positive' in line_lower:
                            analysis["sentiment_drivers"]["positive"].append(detail)
                        elif 'negative' in line_lower:
                            analysis["sentiment_drivers"]["negative"].append(detail)
                        else:
                            analysis["sentiment_drivers"]["neutral"].append(detail)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing sentiment analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_impact_analysis(self, response: str) -> Dict[str, Any]:
        """Parse news impact analysis response"""
        try:
            analysis = {
                "correlation_score": 5,  # 1-10 scale
                "event_impacts": [],
                "market_reaction": {
                    "initial": "neutral",
                    "sustained": "neutral",
                    "reversal_risk": "medium"
                },
                "trading_patterns": {
                    "volume_change": "normal",
                    "volatility_change": "normal",
                    "institutional_activity": "normal"
                },
                "forward_implications": [],
                "risk_factors": [],
                "raw_response": response
            }
            
            # Parse response for key metrics and insights
            lines = response.split('\n')
            current_section = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Extract correlation score
                if 'correlation' in line_lower and any(char.isdigit() for char in line):
                    import re
                    score_match = re.search(r'(\d+)', line)
                    if score_match:
                        score = int(score_match.group(1))
                        if 1 <= score <= 10:
                            analysis["correlation_score"] = score
                
                # Identify sections and extract details
                if line.startswith('-') or line.startswith('•'):
                    detail = line[1:].strip()
                    if 'risk' in current_section:
                        analysis["risk_factors"].append(detail)
                    elif 'implications' in current_section:
                        analysis["forward_implications"].append(detail)
                    elif 'event' in current_section:
                        analysis["event_impacts"].append(detail)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing impact analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_earnings_analysis(self, response: str) -> Dict[str, Any]:
        """Parse earnings sentiment analysis response"""
        try:
            analysis = {
                "earnings_surprise": {
                    "revenue_surprise": 0,
                    "eps_surprise": 0,
                    "guidance_surprise": 0,
                    "quality_score": 5
                },
                "management_sentiment": "neutral",
                "analyst_sentiment": {
                    "rating_changes": [],
                    "target_changes": [],
                    "consensus": "hold"
                },
                "market_reaction": {
                    "pre_market": "neutral",
                    "day_of": "neutral",
                    "sustained": "neutral"
                },
                "sector_comparison": "in-line",
                "raw_response": response
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing earnings analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}