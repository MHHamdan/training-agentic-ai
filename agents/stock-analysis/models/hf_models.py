import logging
import json
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FinancialAnalysisResult:
    """Structured result for financial analysis"""
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    key_points: List[str] = None
    risk_factors: List[str] = None
    market_impact: Optional[str] = None
    investment_outlook: Optional[str] = None
    raw_response: Optional[str] = None
    
    def __post_init__(self):
        if self.key_points is None:
            self.key_points = []
        if self.risk_factors is None:
            self.risk_factors = []

class HuggingFaceModel:
    """Enhanced HuggingFace model with financial analysis capabilities"""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        financial_optimized: bool = False
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.financial_optimized = financial_optimized
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def generate(self, prompt: str) -> str:
        """Generate text using the HuggingFace model"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(self.max_tokens, 1024),  # HF API limits
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if isinstance(result, list) and len(result) > 0:
                            if "generated_text" in result[0]:
                                return result[0]["generated_text"]
                            elif "label" in result[0]:
                                # Handle classification models (sentiment analysis)
                                return json.dumps(result)
                        
                        return str(result)
                        
                    else:
                        error_text = await response.text()
                        logger.error(f"HuggingFace API error for {self.model_name}: {error_text}")
                        
                        if response.status == 503:
                            # Model is loading
                            return f"Model {self.model_name} is currently loading. Please try again in a few minutes."
                        
                        return f"Error: {response.status} - {error_text[:200]}"
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout calling HuggingFace API for {self.model_name}")
                return "Error: Request timeout. The model may be under heavy load."
                
            except Exception as e:
                logger.error(f"Error calling HuggingFace API for {self.model_name}: {str(e)}")
                return f"Error: {str(e)}"
    
    async def analyze_sentiment(self, text: str) -> FinancialAnalysisResult:
        """Specialized sentiment analysis for financial text"""
        if "sentiment" not in self.model_name.lower() and not self.financial_optimized:
            logger.warning(f"Model {self.model_name} may not be optimized for sentiment analysis")
        
        # Use classification endpoint for sentiment models
        if "sentiment" in self.model_name.lower() or "finbert" in self.model_name.lower():
            payload = {
                "inputs": text[:512],  # Limit text length for sentiment models
            }
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if isinstance(result, list) and len(result) > 0:
                                if isinstance(result[0], list):
                                    # Multiple classifications
                                    best_result = max(result[0], key=lambda x: x.get("score", 0))
                                    return FinancialAnalysisResult(
                                        sentiment=best_result.get("label", "UNKNOWN"),
                                        confidence=best_result.get("score", 0.0) * 100,
                                        raw_response=json.dumps(result)
                                    )
                                else:
                                    # Single classification
                                    return FinancialAnalysisResult(
                                        sentiment=result[0].get("label", "UNKNOWN"),
                                        confidence=result[0].get("score", 0.0) * 100,
                                        raw_response=json.dumps(result)
                                    )
                
                except Exception as e:
                    logger.error(f"Sentiment analysis error: {str(e)}")
        
        # Fallback to text generation for non-sentiment models
        prompt = f"""Analyze the financial sentiment of this text. Provide:
1. Sentiment: POSITIVE, NEGATIVE, or NEUTRAL
2. Confidence: 0-100
3. Key factors affecting sentiment
4. Market implications

Text: {text[:1000]}

Analysis:"""
        
        response = await self.generate(prompt)
        return self._parse_sentiment_response(response)
    
    async def analyze_financial_text(self, text: str, analysis_type: str = "comprehensive") -> FinancialAnalysisResult:
        """Comprehensive financial text analysis"""
        if analysis_type == "sentiment":
            return await self.analyze_sentiment(text)
        
        prompt = f"""Provide comprehensive financial analysis of this text:

Text: {text[:2000]}

Please analyze and provide:

1. SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL] with confidence score
2. KEY POINTS: List the most important financial information
3. RISK FACTORS: Identify potential risks or concerns
4. MARKET IMPACT: Assess likely market implications
5. INVESTMENT OUTLOOK: Provide investment perspective

Format your response clearly with each section labeled."""
        
        response = await self.generate(prompt)
        return self._parse_comprehensive_response(response)
    
    async def generate_investment_report(self, stock_data: Dict[str, Any], 
                                       analysis_data: Dict[str, Any]) -> str:
        """Generate structured investment report"""
        prompt = f"""Generate a professional investment analysis report based on the following data:

STOCK DATA:
{json.dumps(stock_data, indent=2)}

ANALYSIS DATA:
{json.dumps(analysis_data, indent=2)}

Please provide a comprehensive investment report including:

1. EXECUTIVE SUMMARY
   - Investment recommendation (BUY/HOLD/SELL)
   - Target price and timeline
   - Key risk factors

2. FUNDAMENTAL ANALYSIS
   - Financial metrics analysis
   - Company strengths and weaknesses
   - Competitive position

3. TECHNICAL ANALYSIS
   - Price trends and patterns
   - Support and resistance levels
   - Technical indicators summary

4. RISK ASSESSMENT
   - Investment risks
   - Market risks
   - Company-specific risks

5. CONCLUSION AND RECOMMENDATIONS
   - Investment thesis
   - Action items
   - Monitoring points

Format as a professional financial report."""
        
        return await self.generate(prompt)
    
    def _parse_sentiment_response(self, response: str) -> FinancialAnalysisResult:
        """Parse sentiment analysis response"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{') or response.strip().startswith('['):
                data = json.loads(response)
                if isinstance(data, list) and len(data) > 0:
                    return FinancialAnalysisResult(
                        sentiment=data[0].get("label", "UNKNOWN"),
                        confidence=data[0].get("score", 0.0) * 100,
                        raw_response=response
                    )
            
            # Parse text response
            sentiment = "NEUTRAL"
            confidence = 50.0
            key_points = []
            
            lines = response.split('\n')
            for line in lines:
                line_lower = line.lower().strip()
                if 'sentiment:' in line_lower:
                    parts = line.split(':')
                    if len(parts) > 1:
                        sentiment_text = parts[1].strip().upper()
                        if any(word in sentiment_text for word in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']):
                            for word in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                                if word in sentiment_text:
                                    sentiment = word
                                    break
                
                elif 'confidence:' in line_lower:
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            confidence = float(parts[1].strip().replace('%', ''))
                        except:
                            pass
                
                elif any(keyword in line_lower for keyword in ['key', 'important', 'factor']):
                    if len(line.strip()) > 10:
                        key_points.append(line.strip())
            
            return FinancialAnalysisResult(
                sentiment=sentiment,
                confidence=confidence,
                key_points=key_points[:5],
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {str(e)}")
            return FinancialAnalysisResult(
                sentiment="UNKNOWN",
                confidence=0.0,
                raw_response=response
            )
    
    def _parse_comprehensive_response(self, response: str) -> FinancialAnalysisResult:
        """Parse comprehensive financial analysis response"""
        try:
            sentiment = "NEUTRAL"
            confidence = 50.0
            key_points = []
            risk_factors = []
            market_impact = ""
            investment_outlook = ""
            
            # Split response into sections
            sections = response.split('\n')
            current_section = ""
            
            for line in sections:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Identify sections
                if any(keyword in line_lower for keyword in ['sentiment', '1.']):
                    current_section = "sentiment"
                    if ':' in line:
                        sentiment_part = line.split(':', 1)[1].strip().upper()
                        for word in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                            if word in sentiment_part:
                                sentiment = word
                                break
                        
                        # Extract confidence if present
                        if '%' in sentiment_part or 'confidence' in line_lower:
                            import re
                            conf_match = re.search(r'(\d+(?:\.\d+)?)%?', sentiment_part)
                            if conf_match:
                                confidence = float(conf_match.group(1))
                
                elif any(keyword in line_lower for keyword in ['key points', '2.', 'key']):
                    current_section = "key_points"
                
                elif any(keyword in line_lower for keyword in ['risk', '3.']):
                    current_section = "risk_factors"
                
                elif any(keyword in line_lower for keyword in ['market impact', '4.']):
                    current_section = "market_impact"
                
                elif any(keyword in line_lower for keyword in ['investment outlook', '5.', 'outlook']):
                    current_section = "investment_outlook"
                
                else:
                    # Add content to current section
                    if current_section == "key_points" and len(line) > 10:
                        if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                            key_points.append(line[1:].strip())
                        elif len(key_points) < 5:
                            key_points.append(line)
                    
                    elif current_section == "risk_factors" and len(line) > 10:
                        if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                            risk_factors.append(line[1:].strip())
                        elif len(risk_factors) < 5:
                            risk_factors.append(line)
                    
                    elif current_section == "market_impact" and len(line) > 10:
                        market_impact += line + " "
                    
                    elif current_section == "investment_outlook" and len(line) > 10:
                        investment_outlook += line + " "
            
            return FinancialAnalysisResult(
                sentiment=sentiment,
                confidence=confidence,
                key_points=key_points[:5],
                risk_factors=risk_factors[:5],
                market_impact=market_impact.strip()[:500],
                investment_outlook=investment_outlook.strip()[:500],
                raw_response=response
            )
            
        except Exception as e:
            logger.error(f"Error parsing comprehensive response: {str(e)}")
            return FinancialAnalysisResult(
                sentiment="UNKNOWN",
                confidence=0.0,
                raw_response=response
            )