import logging
from crewai import Agent
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from utils.observability import track_agent_performance, get_observability_manager

logger = logging.getLogger(__name__)

class TechnicalAnalystAgent:
    """Technical Analysis Expert Agent with pattern recognition and indicator analysis"""
    
    def __init__(self, llm, tools: List[Any]):
        self.observability = get_observability_manager()
        self.llm = llm
        self.tools = tools
        
        self.agent = Agent(
            role='Technical Analysis Expert',
            goal='Perform comprehensive technical analysis and identify trading patterns and signals',
            backstory="""You are a Technical Analysis Expert with advanced expertise in chart patterns,
            technical indicators, and quantitative analysis. You have deep knowledge of market psychology,
            price action analysis, and risk management. Your specialties include identifying support and
            resistance levels, trend analysis, momentum indicators, and volatility patterns. You excel
            at providing actionable trading signals while maintaining strict risk management principles.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=tools,
            max_iter=3,
            memory=True
        )
    
    @track_agent_performance("TechnicalAnalyst", "technical_analysis")
    async def perform_technical_analysis(self, ticker: str, price_data: Dict[str, Any], 
                                       timeframe: str = "daily") -> Dict[str, Any]:
        """Perform comprehensive technical analysis"""
        try:
            # Log compliance action
            self.observability.log_compliance_action(
                agent_name="TechnicalAnalyst",
                action_type="technical_analysis",
                input_data={"ticker": ticker, "timeframe": timeframe},
                output_data={"analysis_type": "comprehensive_technical"},
                risk_level="low",
                decision_reasoning=f"Performing technical analysis for trading signals on {ticker}"
            )
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(price_data)
            
            analysis_prompt = f"""
            Perform comprehensive technical analysis for {ticker} using the following data and indicators:
            
            PRICE DATA:
            {price_data}
            
            TECHNICAL INDICATORS:
            {indicators}
            
            TIMEFRAME: {timeframe}
            
            Provide detailed technical analysis covering:
            
            1. TREND ANALYSIS
               - Primary trend direction (uptrend/downtrend/sideways)
               - Trend strength (weak/moderate/strong)
               - Trend sustainability assessment
               - Key trend lines and channels
               - Trend reversal probability
            
            2. SUPPORT AND RESISTANCE LEVELS
               - Major support levels with price targets
               - Major resistance levels with price targets
               - Dynamic support/resistance (moving averages)
               - Volume-confirmed levels
               - Historical significance of levels
            
            3. CHART PATTERNS
               - Identified chart patterns (triangles, flags, head & shoulders, etc.)
               - Pattern completion probability
               - Price targets from patterns
               - Pattern reliability assessment
               - Time frame for pattern completion
            
            4. MOMENTUM INDICATORS
               - RSI analysis and divergences
               - MACD signal and histogram analysis
               - Stochastic oscillator signals
               - Momentum divergences
               - Overbought/oversold conditions
            
            5. VOLUME ANALYSIS
               - Volume trend analysis
               - Volume-price relationships
               - Accumulation/distribution patterns
               - Volume breakout confirmations
               - Volume-based support/resistance
            
            6. MOVING AVERAGE ANALYSIS
               - Short-term MA signals (20, 50 day)
               - Long-term MA signals (100, 200 day)
               - Moving average crossovers
               - Price vs MA relationships
               - MA slope and convergence/divergence
            
            7. VOLATILITY ANALYSIS
               - Current volatility vs historical
               - Bollinger Bands analysis
               - ATR-based position sizing
               - Volatility breakout potential
               - Risk assessment based on volatility
            
            8. TRADING SIGNALS
               - Buy/sell signal strength (1-10 scale)
               - Entry points with specific prices
               - Stop loss recommendations
               - Profit target levels
               - Position sizing recommendations
            
            9. RISK MANAGEMENT
               - Risk/reward ratios
               - Maximum drawdown expectations
               - Position sizing guidelines
               - Stop loss placement
               - Risk mitigation strategies
            
            10. TIME-BASED ANALYSIS
                - Optimal entry timing
                - Expected move duration
                - Seasonal patterns if applicable
                - Earnings/event impact timing
                - Weekly/daily pattern analysis
            
            Provide specific price levels, percentages, and quantitative measures.
            """
            
            response = self.agent.llm.invoke(analysis_prompt)
            technical_result = self._parse_technical_analysis(response.content if hasattr(response, 'content') else str(response))
            
            # Add calculated indicators to result
            technical_result["calculated_indicators"] = indicators
            
            return technical_result
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("TechnicalAnalyst", "pattern_recognition")
    async def identify_chart_patterns(self, ticker: str, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and analyze chart patterns"""
        try:
            pattern_prompt = f"""
            Identify and analyze chart patterns for {ticker} using the price data:
            
            {price_data}
            
            Focus on identifying and analyzing:
            
            1. CONTINUATION PATTERNS
               - Flags and pennants
               - Triangles (ascending, descending, symmetrical)
               - Rectangles and consolidation patterns
               - Wedges (rising, falling)
               - Cup and handle patterns
            
            2. REVERSAL PATTERNS
               - Head and shoulders (regular and inverted)
               - Double tops and bottoms
               - Triple tops and bottoms
               - Rounding tops and bottoms
               - V-shaped reversals
            
            3. CANDLESTICK PATTERNS
               - Single candlestick patterns (doji, hammer, shooting star)
               - Two-candle patterns (engulfing, harami)
               - Three-candle patterns (morning/evening star, three soldiers)
               - Pattern reliability in current context
            
            4. PATTERN ANALYSIS
               - Pattern completion status
               - Volume confirmation
               - Price targets and measurements
               - Failure rates and reliability
               - Time frame expectations
            
            5. BREAKOUT ANALYSIS
               - Breakout probability assessment
               - Volume requirements for confirmation
               - False breakout risks
               - Measured move calculations
               - Follow-through expectations
            
            For each identified pattern, provide:
            - Pattern name and type
            - Completion probability (%)
            - Price target calculation
            - Stop loss placement
            - Timeline expectations
            - Risk assessment
            """
            
            response = self.agent.llm.invoke(pattern_prompt)
            pattern_result = self._parse_pattern_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return pattern_result
            
        except Exception as e:
            logger.error(f"Error in pattern recognition: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    @track_agent_performance("TechnicalAnalyst", "options_analysis")
    async def analyze_options_flow(self, ticker: str, options_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options flow and sentiment"""
        try:
            options_prompt = f"""
            Analyze options flow and sentiment for {ticker}:
            
            {options_data}
            
            Provide comprehensive options analysis:
            
            1. OPTIONS FLOW ANALYSIS
               - Call vs put volume ratios
               - Unusual options activity
               - Large block trades
               - Smart money indicators
               - Institutional vs retail flow
            
            2. IMPLIED VOLATILITY ANALYSIS
               - IV rank and percentile
               - IV skew analysis
               - Term structure analysis
               - IV vs historical volatility
               - Volatility expectations
            
            3. GAMMA AND DELTA ANALYSIS
               - Gamma exposure levels
               - Delta hedging implications
               - Pin risk analysis
               - Market maker positioning
               - Expiration impact
            
            4. SENTIMENT INDICATORS
               - Put/call ratios
               - Volatility sentiment
               - Contrarian indicators
               - Fear/greed measurements
               - Positioning extremes
            
            5. STRATEGIC IMPLICATIONS
               - Expected price range
               - Breakout probability
               - Support/resistance from strikes
               - Event-driven expectations
               - Risk reversal insights
            """
            
            response = self.agent.llm.invoke(options_prompt)
            options_result = self._parse_options_analysis(response.content if hasattr(response, 'content') else str(response))
            
            return options_result
            
        except Exception as e:
            logger.error(f"Error in options analysis: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    def _calculate_technical_indicators(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic technical indicators from price data"""
        try:
            indicators = {}
            
            # Extract price arrays (mock calculation for demonstration)
            if 'prices' in price_data:
                prices = price_data['prices']
                high = price_data.get('high', prices)
                low = price_data.get('low', prices)
                volume = price_data.get('volume', [1000000] * len(prices))
                
                if len(prices) >= 20:
                    # Simple Moving Averages
                    indicators['sma_20'] = sum(prices[-20:]) / 20
                    indicators['sma_50'] = sum(prices[-50:]) / 50 if len(prices) >= 50 else None
                    indicators['sma_200'] = sum(prices[-200:]) / 200 if len(prices) >= 200 else None
                    
                    # RSI calculation (simplified)
                    if len(prices) >= 14:
                        gains = []
                        losses = []
                        for i in range(1, min(15, len(prices))):
                            change = prices[-i] - prices[-i-1]
                            if change > 0:
                                gains.append(change)
                            else:
                                losses.append(abs(change))
                        
                        avg_gain = sum(gains) / len(gains) if gains else 0
                        avg_loss = sum(losses) / len(losses) if losses else 1
                        rs = avg_gain / avg_loss if avg_loss != 0 else 100
                        indicators['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Current price vs moving averages
                    current_price = prices[-1]
                    indicators['current_price'] = current_price
                    indicators['price_vs_sma20'] = ((current_price - indicators['sma_20']) / indicators['sma_20']) * 100
                    
                    # Volatility (simplified)
                    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(21, len(prices)))]
                    indicators['volatility'] = np.std(returns) * 100 if len(returns) > 1 else 0
                    
                    # Volume analysis
                    indicators['avg_volume'] = sum(volume[-20:]) / 20
                    indicators['volume_ratio'] = volume[-1] / indicators['avg_volume'] if indicators['avg_volume'] > 0 else 1
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {"error": str(e)}
    
    def _parse_technical_analysis(self, response: str) -> Dict[str, Any]:
        """Parse technical analysis response"""
        try:
            analysis = {
                "trend_analysis": {
                    "direction": "sideways",
                    "strength": "moderate",
                    "sustainability": "medium",
                    "reversal_probability": 30
                },
                "support_resistance": {
                    "support_levels": [],
                    "resistance_levels": [],
                    "key_level": None
                },
                "chart_patterns": [],
                "momentum": {
                    "rsi_signal": "neutral",
                    "macd_signal": "neutral",
                    "overall": "neutral"
                },
                "trading_signals": {
                    "signal_strength": 5,  # 1-10 scale
                    "action": "hold",
                    "entry_price": None,
                    "stop_loss": None,
                    "target": None
                },
                "risk_management": {
                    "risk_reward": 1.0,
                    "position_size": "normal",
                    "max_risk": 2.0  # percentage
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
                
                # Identify sections
                if 'trend analysis' in line_lower:
                    current_section = "trend"
                elif 'support' in line_lower and 'resistance' in line_lower:
                    current_section = "levels"
                elif 'trading signals' in line_lower:
                    current_section = "signals"
                elif 'momentum' in line_lower:
                    current_section = "momentum"
                
                # Extract trend information
                if current_section == "trend":
                    if 'uptrend' in line_lower:
                        analysis["trend_analysis"]["direction"] = "uptrend"
                    elif 'downtrend' in line_lower:
                        analysis["trend_analysis"]["direction"] = "downtrend"
                    
                    if 'strong' in line_lower:
                        analysis["trend_analysis"]["strength"] = "strong"
                    elif 'weak' in line_lower:
                        analysis["trend_analysis"]["strength"] = "weak"
                
                # Extract trading signals
                if current_section == "signals":
                    if any(word in line_lower for word in ['buy', 'bullish']):
                        analysis["trading_signals"]["action"] = "buy"
                    elif any(word in line_lower for word in ['sell', 'bearish']):
                        analysis["trading_signals"]["action"] = "sell"
                    
                    # Extract price levels
                    import re
                    price_matches = re.findall(r'\$(\d+(?:\.\d+)?)', line)
                    if price_matches and current_section == "signals":
                        if 'entry' in line_lower:
                            analysis["trading_signals"]["entry_price"] = float(price_matches[0])
                        elif 'stop' in line_lower:
                            analysis["trading_signals"]["stop_loss"] = float(price_matches[0])
                        elif 'target' in line_lower:
                            analysis["trading_signals"]["target"] = float(price_matches[0])
                
                # Extract support/resistance levels
                if current_section == "levels" and (line.startswith('-') or line.startswith('•')):
                    import re
                    price_match = re.search(r'\$(\d+(?:\.\d+)?)', line)
                    if price_match:
                        price = float(price_match.group(1))
                        if 'support' in line_lower:
                            analysis["support_resistance"]["support_levels"].append(price)
                        elif 'resistance' in line_lower:
                            analysis["support_resistance"]["resistance_levels"].append(price)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing technical analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_pattern_analysis(self, response: str) -> Dict[str, Any]:
        """Parse chart pattern analysis response"""
        try:
            analysis = {
                "identified_patterns": [],
                "pattern_reliability": "medium",
                "breakout_probability": 50,
                "price_targets": [],
                "time_expectations": "1-3 weeks",
                "raw_response": response
            }
            
            # Parse patterns from response
            lines = response.split('\n')
            current_pattern = {}
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Identify pattern names
                pattern_keywords = ['triangle', 'flag', 'pennant', 'head', 'shoulder', 'double', 'cup', 'handle']
                if any(keyword in line.lower() for keyword in pattern_keywords):
                    if current_pattern:
                        analysis["identified_patterns"].append(current_pattern)
                    
                    current_pattern = {
                        "name": line,
                        "type": "continuation" if any(word in line.lower() for word in ['flag', 'triangle', 'pennant']) else "reversal",
                        "completion": 70,
                        "target": None,
                        "confidence": "medium"
                    }
                
                # Extract numerical values
                if current_pattern and '%' in line:
                    import re
                    percent_match = re.search(r'(\d+)%', line)
                    if percent_match and 'completion' in line.lower():
                        current_pattern["completion"] = int(percent_match.group(1))
            
            if current_pattern:
                analysis["identified_patterns"].append(current_pattern)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing pattern analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}
    
    def _parse_options_analysis(self, response: str) -> Dict[str, Any]:
        """Parse options flow analysis response"""
        try:
            analysis = {
                "flow_sentiment": "neutral",
                "put_call_ratio": 1.0,
                "iv_analysis": {
                    "iv_rank": 50,
                    "iv_trend": "stable",
                    "iv_vs_hv": "normal"
                },
                "gamma_analysis": {
                    "gamma_exposure": "neutral",
                    "pin_risk": "low"
                },
                "strategic_implications": [],
                "raw_response": response
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing options analysis: {str(e)}")
            return {"error": str(e), "raw_response": response}