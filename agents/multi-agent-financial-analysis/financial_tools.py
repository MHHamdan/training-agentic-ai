"""
Comprehensive Financial Tools Library for Multi-Agent System
"""

import json
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
import requests
from pydantic import Field


class GetRealTimeMarketDataTool(BaseTool):
    """Fetch real-time market data for analysis"""
    name: str = "get_real_time_market_data"
    description: str = "Fetch real-time market data including price, volume, and key metrics for given symbols"
    
    def _run(self, symbols: str) -> str:
        """Execute the tool"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            results = {}
            
            for symbol in symbol_list:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                results[symbol] = {
                    "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                    "volume": info.get("volume", info.get("regularMarketVolume", 0)),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                    "52_week_low": info.get("fiftyTwoWeekLow", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "beta": info.get("beta", 1.0)
                }
            
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error fetching market data: {str(e)}"


class GetHistoricalDataTool(BaseTool):
    """Get historical price and volume data"""
    name: str = "get_historical_data"
    description: str = "Get historical OHLCV data for a symbol over specified period"
    
    def _run(self, symbol: str, period: str = "1mo", interval: str = "1d") -> str:
        """Execute the tool"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                return f"No historical data available for {symbol}"
            
            # Calculate basic statistics
            returns = hist['Close'].pct_change().dropna()
            
            result = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "latest_close": float(hist['Close'].iloc[-1]),
                "period_return": float((hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100),
                "volatility": float(returns.std() * np.sqrt(252) * 100),  # Annualized
                "average_volume": float(hist['Volume'].mean()),
                "high": float(hist['High'].max()),
                "low": float(hist['Low'].min()),
                "data_points": len(hist)
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error fetching historical data: {str(e)}"


class GetFundamentalMetricsTool(BaseTool):
    """Get company fundamental data"""
    name: str = "get_fundamental_metrics"
    description: str = "Get fundamental metrics like P/E, EPS, revenue, profit margins for a symbol"
    
    def _run(self, symbol: str) -> str:
        """Execute the tool"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                "symbol": symbol,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "trailing_pe": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "price_to_sales": info.get("priceToSalesTrailing12Months", 0),
                "profit_margin": info.get("profitMargins", 0),
                "operating_margin": info.get("operatingMargins", 0),
                "return_on_equity": info.get("returnOnEquity", 0),
                "return_on_assets": info.get("returnOnAssets", 0),
                "revenue_growth": info.get("revenueGrowth", 0),
                "earnings_growth": info.get("earningsGrowth", 0),
                "current_ratio": info.get("currentRatio", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "free_cash_flow": info.get("freeCashflow", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "payout_ratio": info.get("payoutRatio", 0),
                "beta": info.get("beta", 1.0),
                "recommendation": info.get("recommendationKey", "none")
            }
            
            return json.dumps(fundamentals, indent=2)
        except Exception as e:
            return f"Error fetching fundamental metrics: {str(e)}"


class CalculateTechnicalIndicatorsTool(BaseTool):
    """Calculate technical indicators"""
    name: str = "calculate_technical_indicators"
    description: str = "Calculate RSI, MACD, Bollinger Bands, and other technical indicators"
    
    def _run(self, symbol: str, period: str = "3mo") -> str:
        """Execute the tool"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return f"No data available for technical analysis of {symbol}"
            
            close_prices = hist['Close']
            
            # RSI Calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Moving Averages
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            ema_12 = close_prices.ewm(span=12, adjust=False).mean()
            ema_26 = close_prices.ewm(span=26, adjust=False).mean()
            
            # MACD
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            # Bollinger Bands
            bb_middle = sma_20
            bb_std = close_prices.rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            
            # Latest values
            latest_price = float(close_prices.iloc[-1])
            
            indicators = {
                "symbol": symbol,
                "current_price": latest_price,
                "rsi": float(rsi.iloc[-1]) if not rsi.empty else None,
                "macd": float(macd_line.iloc[-1]) if not macd_line.empty else None,
                "macd_signal": float(signal_line.iloc[-1]) if not signal_line.empty else None,
                "macd_histogram": float(macd_histogram.iloc[-1]) if not macd_histogram.empty else None,
                "sma_20": float(sma_20.iloc[-1]) if not sma_20.empty else None,
                "sma_50": float(sma_50.iloc[-1]) if not sma_50.empty else None,
                "bb_upper": float(bb_upper.iloc[-1]) if not bb_upper.empty else None,
                "bb_middle": float(bb_middle.iloc[-1]) if not bb_middle.empty else None,
                "bb_lower": float(bb_lower.iloc[-1]) if not bb_lower.empty else None,
                "volume_sma": float(hist['Volume'].rolling(window=20).mean().iloc[-1]),
                
                # Signals
                "rsi_signal": "oversold" if rsi.iloc[-1] < 30 else "overbought" if rsi.iloc[-1] > 70 else "neutral",
                "macd_signal": "bullish" if macd_histogram.iloc[-1] > 0 else "bearish",
                "bb_signal": "oversold" if latest_price < bb_lower.iloc[-1] else "overbought" if latest_price > bb_upper.iloc[-1] else "neutral",
                "trend": "bullish" if sma_20.iloc[-1] > sma_50.iloc[-1] else "bearish"
            }
            
            return json.dumps(indicators, indent=2)
        except Exception as e:
            return f"Error calculating technical indicators: {str(e)}"


class CalculatePortfolioVaRTool(BaseTool):
    """Calculate Value at Risk for portfolio"""
    name: str = "calculate_portfolio_var"
    description: str = "Calculate Value at Risk (VaR) and Conditional VaR for a portfolio using Monte Carlo simulation"
    
    def _run(self, portfolio_json: str, confidence: float = 0.95, days: int = 1) -> str:
        """Execute the tool"""
        try:
            portfolio = json.loads(portfolio_json)
            
            # Fetch historical data for all symbols
            returns_data = {}
            weights = []
            
            for position in portfolio:
                symbol = position['symbol']
                weight = position['weight']
                weights.append(weight)
                
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                returns = hist['Close'].pct_change().dropna()
                returns_data[symbol] = returns
            
            # Create returns DataFrame
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate portfolio metrics
            mean_returns = returns_df.mean()
            cov_matrix = returns_df.cov()
            
            # Monte Carlo simulation
            num_simulations = 10000
            portfolio_returns = []
            
            for _ in range(num_simulations):
                # Generate random returns based on historical statistics
                random_returns = np.random.multivariate_normal(
                    mean_returns * days,
                    cov_matrix * days
                )
                portfolio_return = np.dot(weights, random_returns)
                portfolio_returns.append(portfolio_return)
            
            portfolio_returns = np.array(portfolio_returns)
            
            # Calculate VaR and CVaR
            var_percentile = (1 - confidence) * 100
            var = np.percentile(portfolio_returns, var_percentile)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            
            # Calculate other risk metrics
            result = {
                "confidence_level": confidence,
                "time_horizon_days": days,
                "value_at_risk": float(var * 100),  # Convert to percentage
                "conditional_var": float(cvar * 100),  # Convert to percentage
                "expected_return": float(portfolio_returns.mean() * 100),
                "volatility": float(portfolio_returns.std() * 100),
                "sharpe_ratio": float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)),
                "max_loss": float(portfolio_returns.min() * 100),
                "max_gain": float(portfolio_returns.max() * 100),
                "skewness": float(pd.Series(portfolio_returns).skew()),
                "kurtosis": float(pd.Series(portfolio_returns).kurt())
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error calculating portfolio VaR: {str(e)}"


class StressTestPortfolioTool(BaseTool):
    """Run stress tests on portfolio positions"""
    name: str = "stress_test_portfolio"
    description: str = "Run historical scenario stress tests on portfolio to assess potential losses"
    
    def _run(self, portfolio_json: str, scenarios: str = "2008_crisis,covid_crash,dot_com_bubble") -> str:
        """Execute the tool"""
        try:
            portfolio = json.loads(portfolio_json)
            scenario_list = [s.strip() for s in scenarios.split(',')]
            
            # Define historical stress scenarios (simplified - in production use real data)
            stress_scenarios = {
                "2008_crisis": {"market_drop": -0.45, "volatility_spike": 3.0, "period": "2008 Financial Crisis"},
                "covid_crash": {"market_drop": -0.34, "volatility_spike": 2.5, "period": "COVID-19 Crash"},
                "dot_com_bubble": {"market_drop": -0.49, "volatility_spike": 2.0, "period": "Dot-com Bubble"},
                "black_monday": {"market_drop": -0.22, "volatility_spike": 5.0, "period": "Black Monday 1987"},
                "moderate_correction": {"market_drop": -0.10, "volatility_spike": 1.5, "period": "10% Correction"}
            }
            
            results = {}
            
            for scenario_name in scenario_list:
                if scenario_name not in stress_scenarios:
                    continue
                    
                scenario = stress_scenarios[scenario_name]
                portfolio_impact = 0
                position_impacts = []
                
                for position in portfolio:
                    symbol = position['symbol']
                    weight = position['weight']
                    beta = position.get('beta', 1.0)
                    
                    # Calculate position impact based on beta and market drop
                    position_loss = scenario["market_drop"] * beta * weight
                    portfolio_impact += position_loss
                    
                    position_impacts.append({
                        "symbol": symbol,
                        "weight": weight,
                        "beta": beta,
                        "estimated_loss": float(position_loss * 100)
                    })
                
                results[scenario_name] = {
                    "scenario": scenario["period"],
                    "market_drop": float(scenario["market_drop"] * 100),
                    "portfolio_impact": float(portfolio_impact * 100),
                    "volatility_multiplier": scenario["volatility_spike"],
                    "position_impacts": position_impacts,
                    "risk_assessment": "High Risk" if abs(portfolio_impact) > 0.20 else "Moderate Risk" if abs(portfolio_impact) > 0.10 else "Low Risk"
                }
            
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error running stress test: {str(e)}"


class NewsSentimentAnalysisTool(BaseTool):
    """Analyze news sentiment for given symbols"""
    name: str = "news_sentiment_analysis"
    description: str = "Analyze recent news sentiment and market psychology for given symbols"
    
    def _run(self, symbols: str) -> str:
        """Execute the tool - using mock data for demonstration"""
        try:
            symbol_list = [s.strip() for s in symbols.split(',')]
            results = {}
            
            for symbol in symbol_list:
                # In production, integrate with real news APIs
                # For demo, generate realistic mock sentiment
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Mock sentiment based on recent performance
                recent_change = info.get("regularMarketChangePercent", 0)
                
                if recent_change > 2:
                    sentiment_score = random.uniform(0.5, 0.9)
                    sentiment_label = "Very Positive"
                elif recent_change > 0:
                    sentiment_score = random.uniform(0.2, 0.5)
                    sentiment_label = "Positive"
                elif recent_change > -2:
                    sentiment_score = random.uniform(-0.2, 0.2)
                    sentiment_label = "Neutral"
                else:
                    sentiment_score = random.uniform(-0.9, -0.2)
                    sentiment_label = "Negative"
                
                results[symbol] = {
                    "sentiment_score": float(sentiment_score),
                    "sentiment_label": sentiment_label,
                    "news_volume": random.randint(10, 100),
                    "social_media_mentions": random.randint(100, 10000),
                    "analyst_rating": info.get("recommendationKey", "hold"),
                    "key_topics": ["earnings", "growth", "market share", "innovation"],
                    "confidence": random.uniform(0.6, 0.95)
                }
            
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"


class RegulatoryCheckTool(BaseTool):
    """Check regulatory compliance and restrictions"""
    name: str = "regulatory_check"
    description: str = "Check for regulatory restrictions, insider trading activity, and compliance issues"
    
    def _run(self, symbol: str) -> str:
        """Execute the tool - mock implementation"""
        try:
            # In production, integrate with real regulatory data sources
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            compliance_result = {
                "symbol": symbol,
                "compliance_status": "approved",
                "restricted_list": False,
                "insider_activity": {
                    "recent_buys": random.randint(0, 5),
                    "recent_sells": random.randint(0, 10),
                    "net_activity": "neutral"
                },
                "regulatory_flags": [],
                "esg_score": random.uniform(60, 95),
                "governance_score": random.uniform(50, 90),
                "controversy_level": "low",
                "trading_restrictions": None,
                "notes": "No regulatory issues identified"
            }
            
            return json.dumps(compliance_result, indent=2)
        except Exception as e:
            return f"Error checking regulatory compliance: {str(e)}"


class CreateFinancialReportTool(BaseTool):
    """Create formatted financial reports"""
    name: str = "create_financial_report"
    description: str = "Generate a comprehensive financial analysis report with charts and recommendations"
    
    def _run(self, analysis_data: str, report_type: str = "comprehensive") -> str:
        """Execute the tool"""
        try:
            data = json.loads(analysis_data)
            
            report = {
                "report_type": report_type,
                "generated_at": datetime.now().isoformat(),
                "executive_summary": "Based on comprehensive multi-agent analysis...",
                "sections": {
                    "market_overview": {
                        "title": "Market Overview",
                        "content": "Current market conditions and trends analysis"
                    },
                    "technical_analysis": {
                        "title": "Technical Analysis",
                        "content": "Key technical indicators and chart patterns"
                    },
                    "fundamental_analysis": {
                        "title": "Fundamental Analysis",
                        "content": "Company financials and valuation metrics"
                    },
                    "risk_assessment": {
                        "title": "Risk Assessment",
                        "content": "Portfolio risk metrics and stress test results"
                    },
                    "recommendations": {
                        "title": "Investment Recommendations",
                        "content": "Actionable investment recommendations based on analysis"
                    }
                },
                "charts_generated": [
                    "price_chart",
                    "technical_indicators",
                    "risk_heatmap",
                    "portfolio_allocation"
                ],
                "disclaimer": "This report is for informational purposes only and does not constitute investment advice."
            }
            
            return json.dumps(report, indent=2)
        except Exception as e:
            return f"Error creating financial report: {str(e)}"


def get_all_financial_tools() -> List[BaseTool]:
    """Get all financial tools for agents"""
    return [
        GetRealTimeMarketDataTool(),
        GetHistoricalDataTool(),
        GetFundamentalMetricsTool(),
        CalculateTechnicalIndicatorsTool(),
        CalculatePortfolioVaRTool(),
        StressTestPortfolioTool(),
        NewsSentimentAnalysisTool(),
        RegulatoryCheckTool(),
        CreateFinancialReportTool()
    ]