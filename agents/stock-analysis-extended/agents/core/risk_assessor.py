"""Risk Assessment Agent for portfolio and stock risk analysis"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import asyncio

from agents.base import BaseStockAgent, AgentResult
from tools.analysis.risk_metrics import RiskMetricsCalculator
from crewai.tools import BaseTool


class VolatilityTool(BaseTool):
    name: str = "VolatilityCalculator"
    description: str = "Calculate historical and implied volatility for stocks"
    
    def _run(self, ticker: str, period: str = "1y") -> Dict[str, float]:
        """Calculate volatility metrics"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate returns
        returns = hist['Close'].pct_change().dropna()
        
        # Historical volatility (annualized)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else annual_vol
        
        return {
            "ticker": ticker,
            "historical_volatility": round(annual_vol * 100, 2),
            "current_volatility": round(current_vol * 100, 2),
            "daily_volatility": round(daily_vol * 100, 2),
            "max_drawdown": round(self._calculate_max_drawdown(hist['Close']) * 100, 2)
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class CorrelationTool(BaseTool):
    name: str = "CorrelationAnalyzer"
    description: str = "Analyze correlation between stocks and market indices"
    
    def _run(self, tickers: List[str], period: str = "6mo") -> Dict[str, Any]:
        """Calculate correlation matrix"""
        data = {}
        
        # Download data for all tickers
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if len(data) < 2:
            return {"error": "Insufficient data for correlation analysis"}
        
        # Create DataFrame and calculate returns
        df = pd.DataFrame(data)
        returns = df.pct_change().dropna()
        
        # Calculate correlation matrix
        correlation_matrix = returns.corr()
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "average_correlation": float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()),
            "tickers": tickers
        }


class VaRTool(BaseTool):
    name: str = "ValueAtRiskCalculator"
    description: str = "Calculate Value at Risk (VaR) and Conditional VaR (CVaR)"
    
    def _run(self, ticker: str, confidence_level: float = 0.95, period: str = "1y") -> Dict[str, float]:
        """Calculate VaR and CVaR"""
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {"error": f"No data found for {ticker}"}
        
        # Calculate returns
        returns = hist['Close'].pct_change().dropna()
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns, var_percentile)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()
        
        # Calculate parametric VaR (assuming normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        parametric_var = mean_return - z_score * std_return
        
        return {
            "ticker": ticker,
            "historical_var": round(var * 100, 2),
            "conditional_var": round(cvar * 100, 2),
            "parametric_var": round(parametric_var * 100, 2),
            "confidence_level": confidence_level * 100
        }


class RiskAssessmentAgent(BaseStockAgent):
    """Agent specialized in risk assessment and portfolio risk analysis"""
    
    def __init__(self, **kwargs):
        # Initialize tools
        self.volatility_tool = VolatilityTool()
        self.correlation_tool = CorrelationTool()
        self.var_tool = VaRTool()
        self.risk_calculator = RiskMetricsCalculator()
        
        super().__init__(
            name="RiskAssessmentAgent",
            role="Risk Analyst",
            goal="Evaluate portfolio risk, volatility metrics, and market correlations",
            backstory="""You are a senior risk analyst with expertise in quantitative risk 
            management, portfolio theory, and financial mathematics. You specialize in 
            calculating risk metrics like VaR, CVaR, Sharpe ratios, and volatility measures. 
            Your analysis helps investors understand and manage their portfolio risk.""",
            tools=[self.volatility_tool, self.correlation_tool, self.var_tool],
            **kwargs
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        required_fields = ['ticker']
        return all(field in input_data for field in required_fields)
    
    async def analyze(self, input_data: Dict[str, Any]) -> AgentResult:
        """Perform comprehensive risk assessment"""
        ticker = input_data.get('ticker')
        portfolio = input_data.get('portfolio', [ticker])
        period = input_data.get('period', '1y')
        benchmark = input_data.get('benchmark', 'SPY')
        
        try:
            # Gather all risk metrics in parallel
            tasks = [
                self._get_volatility_metrics(ticker, period),
                self._get_var_metrics(ticker, period),
                self._get_correlation_metrics(portfolio + [benchmark], period),
                self._get_risk_ratios(ticker, benchmark, period),
                self._get_beta_alpha(ticker, benchmark, period)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            risk_data = {
                'volatility_metrics': results[0] if not isinstance(results[0], Exception) else None,
                'value_at_risk': results[1] if not isinstance(results[1], Exception) else None,
                'correlation_analysis': results[2] if not isinstance(results[2], Exception) else None,
                'risk_ratios': results[3] if not isinstance(results[3], Exception) else None,
                'market_risk': results[4] if not isinstance(results[4], Exception) else None
            }
            
            # Generate risk assessment summary
            risk_assessment = self._generate_risk_assessment(risk_data)
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(risk_data)
            
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'risk_assessment'),
                status='completed',
                data={
                    'ticker': ticker,
                    'risk_metrics': risk_data,
                    'risk_assessment': risk_assessment,
                    'risk_score': risk_score,
                    'recommendations': self._generate_recommendations(risk_score, risk_data)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'risk_assessment'),
                status='failed',
                errors=[str(e)]
            )
    
    async def _get_volatility_metrics(self, ticker: str, period: str) -> Dict[str, Any]:
        """Get volatility metrics"""
        return await asyncio.to_thread(self.volatility_tool._run, ticker, period)
    
    async def _get_var_metrics(self, ticker: str, period: str) -> Dict[str, Any]:
        """Get Value at Risk metrics"""
        return await asyncio.to_thread(self.var_tool._run, ticker, 0.95, period)
    
    async def _get_correlation_metrics(self, tickers: List[str], period: str) -> Dict[str, Any]:
        """Get correlation metrics"""
        return await asyncio.to_thread(self.correlation_tool._run, tickers, period)
    
    async def _get_risk_ratios(self, ticker: str, benchmark: str, period: str) -> Dict[str, Any]:
        """Calculate risk-adjusted return ratios"""
        return await asyncio.to_thread(
            self.risk_calculator.calculate_risk_ratios, 
            ticker, 
            benchmark, 
            period
        )
    
    async def _get_beta_alpha(self, ticker: str, benchmark: str, period: str) -> Dict[str, Any]:
        """Calculate beta and alpha"""
        return await asyncio.to_thread(
            self.risk_calculator.calculate_beta_alpha,
            ticker,
            benchmark,
            period
        )
    
    def _generate_risk_assessment(self, risk_data: Dict[str, Any]) -> str:
        """Generate textual risk assessment"""
        assessment_parts = []
        
        # Volatility assessment
        if risk_data.get('volatility_metrics'):
            vol = risk_data['volatility_metrics'].get('historical_volatility', 0)
            if vol > 40:
                assessment_parts.append("HIGH VOLATILITY: Stock shows significant price fluctuations.")
            elif vol > 20:
                assessment_parts.append("MODERATE VOLATILITY: Stock has average price movements.")
            else:
                assessment_parts.append("LOW VOLATILITY: Stock is relatively stable.")
        
        # VaR assessment
        if risk_data.get('value_at_risk'):
            var = abs(risk_data['value_at_risk'].get('historical_var', 0))
            assessment_parts.append(f"Daily VaR at 95% confidence: {var}%")
        
        # Correlation assessment
        if risk_data.get('correlation_analysis'):
            avg_corr = risk_data['correlation_analysis'].get('average_correlation', 0)
            if avg_corr > 0.7:
                assessment_parts.append("HIGH CORRELATION: Strong relationship with market/portfolio.")
            elif avg_corr > 0.3:
                assessment_parts.append("MODERATE CORRELATION: Some diversification benefits.")
            else:
                assessment_parts.append("LOW CORRELATION: Good diversification potential.")
        
        return " ".join(assessment_parts)
    
    def _calculate_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100)"""
        score_components = []
        
        # Volatility component (0-40 points)
        if risk_data.get('volatility_metrics'):
            vol = risk_data['volatility_metrics'].get('historical_volatility', 0)
            vol_score = min(vol, 40)  # Cap at 40
            score_components.append(vol_score)
        
        # VaR component (0-30 points)
        if risk_data.get('value_at_risk'):
            var = abs(risk_data['value_at_risk'].get('historical_var', 0))
            var_score = min(var * 10, 30)  # Scale and cap at 30
            score_components.append(var_score)
        
        # Drawdown component (0-30 points)
        if risk_data.get('volatility_metrics'):
            dd = abs(risk_data['volatility_metrics'].get('max_drawdown', 0))
            dd_score = min(dd, 30)  # Cap at 30
            score_components.append(dd_score)
        
        # Calculate weighted average
        if score_components:
            return round(sum(score_components) / len(score_components) * (100/40), 2)
        return 50.0  # Default moderate risk
    
    def _generate_recommendations(self, risk_score: float, risk_data: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if risk_score > 70:
            recommendations.append("Consider reducing position size due to high risk")
            recommendations.append("Implement stop-loss orders to limit downside")
            recommendations.append("Diversify portfolio to reduce concentration risk")
        elif risk_score > 40:
            recommendations.append("Monitor position regularly for risk changes")
            recommendations.append("Consider hedging strategies for protection")
        else:
            recommendations.append("Risk levels are acceptable for most investors")
            recommendations.append("Consider increasing position if fits investment goals")
        
        # Specific recommendations based on metrics
        if risk_data.get('volatility_metrics', {}).get('max_drawdown', 0) < -20:
            recommendations.append("Historical drawdown suggests potential for significant losses")
        
        if risk_data.get('risk_ratios', {}).get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Risk-adjusted returns are below optimal levels")
        
        return recommendations