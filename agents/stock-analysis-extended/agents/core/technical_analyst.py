"""Technical Analysis Agent for advanced technical indicators and chart patterns"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import pandas_ta as ta

from agents.base import BaseStockAgent, AgentResult
from crewai.tools import BaseTool


class TechnicalIndicatorsTool(BaseTool):
    name: str = "TechnicalIndicatorsCalculator"
    description: str = "Calculate various technical indicators for stock analysis"
    
    def _run(self, ticker: str, period: str = "3mo") -> Dict[str, Any]:
        """Calculate technical indicators"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty:
                return {"error": f"No data found for {ticker}"}
            
            # Calculate various indicators
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = float(df['Close'].rolling(window=20).mean().iloc[-1])
            indicators['sma_50'] = float(df['Close'].rolling(window=50).mean().iloc[-1]) if len(df) >= 50 else None
            indicators['sma_200'] = float(df['Close'].rolling(window=200).mean().iloc[-1]) if len(df) >= 200 else None
            indicators['ema_12'] = float(df['Close'].ewm(span=12).mean().iloc[-1])
            indicators['ema_26'] = float(df['Close'].ewm(span=26).mean().iloc[-1])
            
            # RSI
            rsi = ta.rsi(df['Close'], length=14)
            indicators['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else None
            
            # MACD
            macd = ta.macd(df['Close'])
            if macd is not None and not macd.empty:
                indicators['macd'] = float(macd['MACD_12_26_9'].iloc[-1]) if 'MACD_12_26_9' in macd.columns else None
                indicators['macd_signal'] = float(macd['MACDs_12_26_9'].iloc[-1]) if 'MACDs_12_26_9' in macd.columns else None
                indicators['macd_histogram'] = float(macd['MACDh_12_26_9'].iloc[-1]) if 'MACDh_12_26_9' in macd.columns else None
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], length=20, std=2)
            if bb is not None and not bb.empty:
                indicators['bb_upper'] = float(bb['BBU_20_2.0'].iloc[-1]) if 'BBU_20_2.0' in bb.columns else None
                indicators['bb_middle'] = float(bb['BBM_20_2.0'].iloc[-1]) if 'BBM_20_2.0' in bb.columns else None
                indicators['bb_lower'] = float(bb['BBL_20_2.0'].iloc[-1]) if 'BBL_20_2.0' in bb.columns else None
            
            # Stochastic Oscillator
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            if stoch is not None and not stoch.empty:
                indicators['stoch_k'] = float(stoch['STOCHk_14_3_3'].iloc[-1]) if 'STOCHk_14_3_3' in stoch.columns else None
                indicators['stoch_d'] = float(stoch['STOCHd_14_3_3'].iloc[-1]) if 'STOCHd_14_3_3' in stoch.columns else None
            
            # ATR (Average True Range)
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            indicators['atr'] = float(atr.iloc[-1]) if atr is not None and not atr.empty else None
            
            # Volume indicators
            indicators['volume_sma'] = float(df['Volume'].rolling(window=20).mean().iloc[-1])
            indicators['volume_ratio'] = float(df['Volume'].iloc[-1] / indicators['volume_sma']) if indicators['volume_sma'] else None
            
            # OBV (On-Balance Volume)
            obv = ta.obv(df['Close'], df['Volume'])
            indicators['obv'] = float(obv.iloc[-1]) if obv is not None and not obv.empty else None
            
            # Current price info
            indicators['current_price'] = float(df['Close'].iloc[-1])
            indicators['price_change'] = float(df['Close'].pct_change().iloc[-1] * 100)
            
            return {
                'ticker': ticker,
                'indicators': indicators,
                'last_updated': df.index[-1].strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            return {"error": str(e), "ticker": ticker}


class ChartPatternTool(BaseTool):
    name: str = "ChartPatternRecognizer"
    description: str = "Identify chart patterns and price formations"
    
    def _run(self, ticker: str, period: str = "6mo") -> Dict[str, Any]:
        """Identify chart patterns"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df.empty or len(df) < 20:
                return {"error": f"Insufficient data for {ticker}"}
            
            patterns = []
            
            # Support and Resistance levels
            support, resistance = self._find_support_resistance(df)
            patterns.append({
                'pattern': 'Support/Resistance',
                'support_levels': support,
                'resistance_levels': resistance
            })
            
            # Trend detection
            trend = self._detect_trend(df)
            patterns.append({
                'pattern': 'Trend',
                'direction': trend['direction'],
                'strength': trend['strength']
            })
            
            # Moving Average Crossovers
            ma_cross = self._detect_ma_crossover(df)
            if ma_cross:
                patterns.append(ma_cross)
            
            # Price patterns
            head_shoulders = self._detect_head_shoulders(df)
            if head_shoulders:
                patterns.append(head_shoulders)
            
            # Candlestick patterns
            candlestick = self._detect_candlestick_patterns(df)
            if candlestick:
                patterns.extend(candlestick)
            
            return {
                'ticker': ticker,
                'patterns': patterns,
                'current_price': float(df['Close'].iloc[-1]),
                'analysis_period': period
            }
            
        except Exception as e:
            return {"error": str(e), "ticker": ticker}
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        # Simple approach: use recent lows for support and highs for resistance
        window = 20
        
        # Rolling min/max
        support_levels = []
        resistance_levels = []
        
        for i in range(0, len(df), window):
            subset = df.iloc[i:i+window]
            if len(subset) > 0:
                support_levels.append(float(subset['Low'].min()))
                resistance_levels.append(float(subset['High'].max()))
        
        # Remove duplicates and sort
        support_levels = sorted(list(set([round(s, 2) for s in support_levels])))[:3]
        resistance_levels = sorted(list(set([round(r, 2) for r in resistance_levels])), reverse=True)[:3]
        
        return support_levels, resistance_levels
    
    def _detect_trend(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detect price trend"""
        # Use linear regression on closing prices
        from scipy import stats
        
        x = np.arange(len(df))
        y = df['Close'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend direction
        if slope > 0:
            direction = "UPTREND"
        elif slope < 0:
            direction = "DOWNTREND"
        else:
            direction = "SIDEWAYS"
        
        # Determine trend strength based on R-squared
        r_squared = r_value ** 2
        if r_squared > 0.7:
            strength = "STRONG"
        elif r_squared > 0.3:
            strength = "MODERATE"
        else:
            strength = "WEAK"
        
        return {'direction': direction, 'strength': strength}
    
    def _detect_ma_crossover(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect moving average crossovers"""
        if len(df) < 50:
            return None
        
        # Calculate MAs
        ma_20 = df['Close'].rolling(window=20).mean()
        ma_50 = df['Close'].rolling(window=50).mean()
        
        # Check for recent crossover (last 5 days)
        for i in range(-5, 0):
            if i-1 >= -len(df):
                # Golden cross (bullish)
                if ma_20.iloc[i-1] <= ma_50.iloc[i-1] and ma_20.iloc[i] > ma_50.iloc[i]:
                    return {
                        'pattern': 'Golden Cross',
                        'signal': 'BULLISH',
                        'date': df.index[i].strftime('%Y-%m-%d')
                    }
                # Death cross (bearish)
                elif ma_20.iloc[i-1] >= ma_50.iloc[i-1] and ma_20.iloc[i] < ma_50.iloc[i]:
                    return {
                        'pattern': 'Death Cross',
                        'signal': 'BEARISH',
                        'date': df.index[i].strftime('%Y-%m-%d')
                    }
        
        return None
    
    def _detect_head_shoulders(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect head and shoulders pattern"""
        # Simplified detection - look for three peaks
        if len(df) < 30:
            return None
        
        # Find local maxima
        highs = df['High'].rolling(window=5).max()
        
        # This is a simplified version - real implementation would be more complex
        return None
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect candlestick patterns"""
        patterns = []
        
        # Last few candles
        for i in range(-3, 0):
            if i >= -len(df):
                open_price = df['Open'].iloc[i]
                close_price = df['Close'].iloc[i]
                high_price = df['High'].iloc[i]
                low_price = df['Low'].iloc[i]
                
                body = abs(close_price - open_price)
                range_hl = high_price - low_price
                
                # Doji
                if body / range_hl < 0.1 and range_hl > 0:
                    patterns.append({
                        'pattern': 'Doji',
                        'signal': 'NEUTRAL/REVERSAL',
                        'date': df.index[i].strftime('%Y-%m-%d')
                    })
                
                # Hammer (bullish)
                if close_price > open_price and (low_price < open_price - body) and (high_price - close_price < body * 0.3):
                    patterns.append({
                        'pattern': 'Hammer',
                        'signal': 'BULLISH',
                        'date': df.index[i].strftime('%Y-%m-%d')
                    })
                
                # Shooting Star (bearish)
                if close_price < open_price and (high_price > open_price + body) and (close_price - low_price < body * 0.3):
                    patterns.append({
                        'pattern': 'Shooting Star',
                        'signal': 'BEARISH',
                        'date': df.index[i].strftime('%Y-%m-%d')
                    })
        
        return patterns[:3]  # Return max 3 most recent patterns


class SignalGeneratorTool(BaseTool):
    name: str = "TechnicalSignalGenerator"
    description: str = "Generate buy/sell signals based on technical indicators"
    
    def _run(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals from indicators"""
        signals = []
        score = 0
        
        # RSI signals
        if indicators.get('rsi'):
            rsi = indicators['rsi']
            if rsi < 30:
                signals.append({'indicator': 'RSI', 'signal': 'OVERSOLD', 'strength': 'STRONG'})
                score += 2
            elif rsi < 40:
                signals.append({'indicator': 'RSI', 'signal': 'OVERSOLD', 'strength': 'MODERATE'})
                score += 1
            elif rsi > 70:
                signals.append({'indicator': 'RSI', 'signal': 'OVERBOUGHT', 'strength': 'STRONG'})
                score -= 2
            elif rsi > 60:
                signals.append({'indicator': 'RSI', 'signal': 'OVERBOUGHT', 'strength': 'MODERATE'})
                score -= 1
        
        # MACD signals
        if indicators.get('macd') and indicators.get('macd_signal'):
            macd = indicators['macd']
            signal = indicators['macd_signal']
            if macd > signal:
                signals.append({'indicator': 'MACD', 'signal': 'BULLISH CROSSOVER', 'strength': 'MODERATE'})
                score += 1
            else:
                signals.append({'indicator': 'MACD', 'signal': 'BEARISH CROSSOVER', 'strength': 'MODERATE'})
                score -= 1
        
        # Bollinger Bands signals
        if indicators.get('current_price') and indicators.get('bb_lower') and indicators.get('bb_upper'):
            price = indicators['current_price']
            bb_lower = indicators['bb_lower']
            bb_upper = indicators['bb_upper']
            
            if price < bb_lower:
                signals.append({'indicator': 'Bollinger Bands', 'signal': 'OVERSOLD', 'strength': 'STRONG'})
                score += 2
            elif price > bb_upper:
                signals.append({'indicator': 'Bollinger Bands', 'signal': 'OVERBOUGHT', 'strength': 'STRONG'})
                score -= 2
        
        # Moving Average signals
        if indicators.get('current_price') and indicators.get('sma_50') and indicators.get('sma_200'):
            price = indicators['current_price']
            sma_50 = indicators['sma_50']
            sma_200 = indicators['sma_200']
            
            if sma_50 and sma_200:
                if sma_50 > sma_200:
                    signals.append({'indicator': 'MA Cross', 'signal': 'GOLDEN CROSS', 'strength': 'STRONG'})
                    score += 2
                else:
                    signals.append({'indicator': 'MA Cross', 'signal': 'DEATH CROSS', 'strength': 'STRONG'})
                    score -= 2
            
            if sma_50 and price > sma_50:
                signals.append({'indicator': 'Price/MA', 'signal': 'ABOVE MA50', 'strength': 'MODERATE'})
                score += 1
        
        # Volume signals
        if indicators.get('volume_ratio'):
            vol_ratio = indicators['volume_ratio']
            if vol_ratio > 1.5:
                signals.append({'indicator': 'Volume', 'signal': 'HIGH VOLUME', 'strength': 'MODERATE'})
        
        # Overall recommendation
        if score >= 3:
            recommendation = "STRONG BUY"
        elif score >= 1:
            recommendation = "BUY"
        elif score <= -3:
            recommendation = "STRONG SELL"
        elif score <= -1:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return {
            'signals': signals,
            'score': score,
            'recommendation': recommendation,
            'confidence': self._calculate_confidence(len(signals))
        }
    
    def _calculate_confidence(self, signal_count: int) -> str:
        """Calculate confidence based on number of signals"""
        if signal_count >= 5:
            return "HIGH"
        elif signal_count >= 3:
            return "MEDIUM"
        else:
            return "LOW"


class TechnicalAnalysisAgent(BaseStockAgent):
    """Agent specialized in technical analysis and chart patterns"""
    
    def __init__(self, **kwargs):
        # Initialize tools
        self.indicators_tool = TechnicalIndicatorsTool()
        self.pattern_tool = ChartPatternTool()
        self.signal_tool = SignalGeneratorTool()
        
        super().__init__(
            name="TechnicalAnalysisAgent",
            role="Technical Analyst",
            goal="Perform advanced technical analysis using indicators and chart patterns",
            backstory="""You are a veteran technical analyst with decades of experience in 
            chart reading and technical indicator interpretation. You specialize in identifying 
            trading opportunities through price patterns, momentum indicators, and volume analysis. 
            Your expertise includes RSI, MACD, Bollinger Bands, moving averages, and complex 
            chart patterns.""",
            tools=[self.indicators_tool, self.pattern_tool, self.signal_tool],
            **kwargs
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return 'ticker' in input_data
    
    async def analyze(self, input_data: Dict[str, Any]) -> AgentResult:
        """Perform comprehensive technical analysis"""
        ticker = input_data.get('ticker')
        period = input_data.get('period', '3mo')
        
        try:
            # Calculate indicators and patterns in parallel
            tasks = [
                self._calculate_indicators(ticker, period),
                self._identify_patterns(ticker, period)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            indicators_result = results[0] if not isinstance(results[0], Exception) else None
            patterns_result = results[1] if not isinstance(results[1], Exception) else None
            
            # Generate signals if we have indicators
            signals_result = None
            if indicators_result and 'indicators' in indicators_result:
                signals_result = await self._generate_signals(indicators_result['indicators'])
            
            # Compile technical analysis
            analysis = self._compile_technical_analysis(
                indicators_result,
                patterns_result,
                signals_result
            )
            
            # Generate technical summary
            summary = self._generate_technical_summary(analysis)
            
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'technical_analysis'),
                status='completed',
                data={
                    'ticker': ticker,
                    'technical_indicators': indicators_result,
                    'chart_patterns': patterns_result,
                    'trading_signals': signals_result,
                    'analysis': analysis,
                    'summary': summary,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'technical_analysis'),
                status='failed',
                errors=[str(e)]
            )
    
    async def _calculate_indicators(self, ticker: str, period: str) -> Dict[str, Any]:
        """Calculate technical indicators"""
        return await asyncio.to_thread(self.indicators_tool._run, ticker, period)
    
    async def _identify_patterns(self, ticker: str, period: str) -> Dict[str, Any]:
        """Identify chart patterns"""
        return await asyncio.to_thread(self.pattern_tool._run, ticker, period)
    
    async def _generate_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals"""
        return await asyncio.to_thread(self.signal_tool._run, indicators)
    
    def _compile_technical_analysis(
        self,
        indicators: Optional[Dict],
        patterns: Optional[Dict],
        signals: Optional[Dict]
    ) -> Dict[str, Any]:
        """Compile comprehensive technical analysis"""
        analysis = {
            'strength_indicators': [],
            'weakness_indicators': [],
            'key_levels': {},
            'trend_analysis': None,
            'momentum_status': None
        }
        
        # Analyze indicators
        if indicators and 'indicators' in indicators:
            ind = indicators['indicators']
            
            # Trend strength
            if ind.get('rsi'):
                if ind['rsi'] > 50:
                    analysis['strength_indicators'].append('RSI above 50')
                else:
                    analysis['weakness_indicators'].append('RSI below 50')
            
            # Key levels
            if ind.get('sma_50'):
                analysis['key_levels']['sma_50'] = ind['sma_50']
            if ind.get('sma_200'):
                analysis['key_levels']['sma_200'] = ind['sma_200']
            if ind.get('bb_upper'):
                analysis['key_levels']['resistance'] = ind['bb_upper']
            if ind.get('bb_lower'):
                analysis['key_levels']['support'] = ind['bb_lower']
        
        # Analyze patterns
        if patterns and 'patterns' in patterns:
            for pattern in patterns['patterns']:
                if pattern.get('pattern') == 'Trend':
                    analysis['trend_analysis'] = pattern
        
        # Analyze signals
        if signals:
            analysis['momentum_status'] = signals.get('recommendation')
        
        return analysis
    
    def _generate_technical_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate technical analysis summary"""
        summary_parts = []
        
        # Trend summary
        if analysis.get('trend_analysis'):
            trend = analysis['trend_analysis']
            summary_parts.append(f"{trend['strength']} {trend['direction']} detected")
        
        # Momentum summary
        if analysis.get('momentum_status'):
            summary_parts.append(f"Technical indicators suggest {analysis['momentum_status']}")
        
        # Strength/weakness balance
        strengths = len(analysis.get('strength_indicators', []))
        weaknesses = len(analysis.get('weakness_indicators', []))
        
        if strengths > weaknesses:
            summary_parts.append("Overall technical picture is BULLISH")
        elif weaknesses > strengths:
            summary_parts.append("Overall technical picture is BEARISH")
        else:
            summary_parts.append("Technical indicators are MIXED")
        
        # Key levels
        if analysis.get('key_levels'):
            levels = analysis['key_levels']
            if 'support' in levels and 'resistance' in levels:
                summary_parts.append(
                    f"Key levels: Support at {levels['support']:.2f}, "
                    f"Resistance at {levels['resistance']:.2f}"
                )
        
        return ". ".join(summary_parts)