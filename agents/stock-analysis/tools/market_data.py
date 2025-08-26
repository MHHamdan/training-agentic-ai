import yfinance as yf
import requests
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from dataclasses import dataclass

from crewai_tools import BaseTool
from config import config
from utils.observability import track_tool_usage, get_observability_manager

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    """Structured stock data"""
    symbol: str
    current_price: Optional[float] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    company_name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    
    @property
    def summary(self) -> str:
        """Get formatted summary"""
        return f"""
{self.company_name} ({self.symbol})
Current Price: ${self.current_price:.2f} ({self.change_percent:+.2f}%)
Volume: {self.volume:,} | Market Cap: ${self.market_cap/1e9:.2f}B
P/E Ratio: {self.pe_ratio} | Dividend Yield: {self.dividend_yield:.2f}%
52-Week Range: ${self.week_52_low:.2f} - ${self.week_52_high:.2f}
Sector: {self.sector} | Industry: {self.industry}
        """.strip()

class MarketDataTool(BaseTool):
    """Enhanced market data tool with multiple data sources"""
    
    name: str = "MarketDataSearcher"
    description: str = """Get comprehensive real-time and historical stock data using yfinance and Alpha Vantage.
    Input should be a stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')."""
    
    def __init__(self):
        super().__init__()
        self.observability = get_observability_manager()
    
    @track_tool_usage("MarketDataTool")
    def _run(self, ticker: str) -> str:
        """Get market data for a stock ticker"""
        try:
            ticker = ticker.upper().strip()
            stock_data = self.get_stock_data(ticker)
            
            if stock_data:
                # Log compliance action
                self.observability.log_compliance_action(
                    agent_name="MarketDataTool",
                    action_type="market_data_access",
                    input_data={"ticker": ticker},
                    output_data={"price": stock_data.current_price, "volume": stock_data.volume},
                    risk_level="low",
                    decision_reasoning=f"Accessing public market data for {ticker}"
                )
                
                return stock_data.summary
            else:
                return f"Could not retrieve market data for {ticker}. Please check the ticker symbol."
        
        except Exception as e:
            logger.error(f"Error in market data tool: {str(e)}")
            return f"Error retrieving market data for {ticker}: {str(e)}"
    
    def get_stock_data(self, ticker: str) -> Optional[StockData]:
        """Get comprehensive stock data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get current price data
            hist = stock.history(period="5d")
            if hist.empty:
                logger.error(f"No historical data found for {ticker}")
                return None
            
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price)
            
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
            
            stock_data = StockData(
                symbol=ticker,
                current_price=float(current_price),
                previous_close=float(previous_close),
                change=float(change),
                change_percent=float(change_percent),
                volume=info.get('volume', int(hist['Volume'].iloc[-1])),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('forwardPE') or info.get('trailingPE'),
                dividend_yield=info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                week_52_high=info.get('fiftyTwoWeekHigh'),
                week_52_low=info.get('fiftyTwoWeekLow'),
                company_name=info.get('longName') or info.get('shortName'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                description=info.get('longBusinessSummary', '')[:500]
            )
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {str(e)}")
            return None

class HistoricalDataTool(BaseTool):
    """Tool for historical stock data analysis"""
    
    name: str = "HistoricalDataAnalyzer"
    description: str = """Get historical stock price data for technical analysis.
    Input should be: 'ticker,period' where period is '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y'"""
    
    @track_tool_usage("HistoricalDataTool")
    def _run(self, query: str) -> str:
        """Get historical data for analysis"""
        try:
            parts = query.split(',')
            if len(parts) != 2:
                return "Please provide input as 'ticker,period' (e.g., 'AAPL,3mo')"
            
            ticker, period = parts[0].strip().upper(), parts[1].strip()
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                return f"No historical data found for {ticker} with period {period}"
            
            # Calculate basic statistics
            current_price = hist['Close'].iloc[-1]
            period_start = hist['Close'].iloc[0]
            period_return = ((current_price - period_start) / period_start) * 100
            
            high = hist['High'].max()
            low = hist['Low'].min()
            avg_volume = hist['Volume'].mean()
            
            # Calculate volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            summary = f"""
Historical Analysis for {ticker} ({period}):
Period Return: {period_return:+.2f}%
Price Range: ${low:.2f} - ${high:.2f}
Current Price: ${current_price:.2f}
Average Volume: {avg_volume:,.0f}
Volatility: {volatility:.2f}%
Data Points: {len(hist)} trading days
            """.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in historical data tool: {str(e)}")
            return f"Error retrieving historical data: {str(e)}"

class MarketScreenerTool(BaseTool):
    """Tool for screening stocks based on criteria"""
    
    name: str = "MarketScreener"
    description: str = """Screen stocks based on market criteria like market cap, sector, performance.
    Input should be screening criteria like 'large_cap', 'tech_stocks', 'high_volume', 'gainers', 'losers'"""
    
    @track_tool_usage("MarketScreenerTool")
    def _run(self, criteria: str) -> str:
        """Screen stocks based on criteria"""
        try:
            criteria = criteria.lower().strip()
            
            if criteria in ['gainers', 'top_gainers']:
                return self._get_top_gainers()
            elif criteria in ['losers', 'top_losers']:
                return self._get_top_losers()
            elif criteria in ['high_volume', 'volume']:
                return self._get_high_volume_stocks()
            elif criteria in ['tech_stocks', 'technology']:
                return self._get_tech_stocks()
            elif criteria in ['large_cap', 'mega_cap']:
                return self._get_large_cap_stocks()
            else:
                return f"Screening criteria '{criteria}' not supported. Try: gainers, losers, high_volume, tech_stocks, large_cap"
        
        except Exception as e:
            logger.error(f"Error in market screener: {str(e)}")
            return f"Error screening stocks: {str(e)}"
    
    def _get_top_gainers(self) -> str:
        """Get top performing stocks"""
        try:
            # Popular tickers for screening
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM']
            gainers = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Open'].iloc[0]
                        change_percent = ((current - previous) / previous) * 100
                        
                        if change_percent > 0:
                            gainers.append((ticker, change_percent, current))
                except:
                    continue
            
            gainers.sort(key=lambda x: x[1], reverse=True)
            
            result = "Top Gainers Today:\n"
            for ticker, change, price in gainers[:5]:
                result += f"{ticker}: ${price:.2f} (+{change:.2f}%)\n"
            
            return result
            
        except Exception as e:
            return f"Error getting top gainers: {str(e)}"
    
    def _get_top_losers(self) -> str:
        """Get worst performing stocks"""
        try:
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM']
            losers = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        previous = hist['Open'].iloc[0]
                        change_percent = ((current - previous) / previous) * 100
                        
                        if change_percent < 0:
                            losers.append((ticker, change_percent, current))
                except:
                    continue
            
            losers.sort(key=lambda x: x[1])
            
            result = "Top Losers Today:\n"
            for ticker, change, price in losers[:5]:
                result += f"{ticker}: ${price:.2f} ({change:.2f}%)\n"
            
            return result
            
        except Exception as e:
            return f"Error getting top losers: {str(e)}"
    
    def _get_high_volume_stocks(self) -> str:
        """Get high volume stocks"""
        try:
            tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'CRM']
            volume_stocks = []
            
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period="1d")
                    if not hist.empty:
                        volume = hist['Volume'].iloc[-1]
                        price = hist['Close'].iloc[-1]
                        volume_stocks.append((ticker, volume, price))
                except:
                    continue
            
            volume_stocks.sort(key=lambda x: x[1], reverse=True)
            
            result = "High Volume Stocks Today:\n"
            for ticker, volume, price in volume_stocks[:5]:
                result += f"{ticker}: ${price:.2f} | Volume: {volume:,.0f}\n"
            
            return result
            
        except Exception as e:
            return f"Error getting high volume stocks: {str(e)}"
    
    def _get_tech_stocks(self) -> str:
        """Get technology sector stocks"""
        tech_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'AMD', 'CRM', 'ORCL', 'ADBE']
        
        result = "Technology Sector Stocks:\n"
        for ticker in tech_tickers[:5]:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    result += f"{ticker}: ${price:.2f} | Volume: {volume:,.0f}\n"
            except:
                continue
        
        return result
    
    def _get_large_cap_stocks(self) -> str:
        """Get large cap stocks"""
        large_cap_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'BRK-B', 'META', 'NVDA', 'UNH', 'JNJ']
        
        result = "Large Cap Stocks:\n"
        for ticker in large_cap_tickers[:5]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1d")
                
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    market_cap = info.get('marketCap', 0)
                    result += f"{ticker}: ${price:.2f} | Market Cap: ${market_cap/1e9:.1f}B\n"
            except:
                continue
        
        return result

class AlphaVantageDataTool(BaseTool):
    """Alpha Vantage API integration for additional financial data"""
    
    name: str = "AlphaVantageData"
    description: str = """Get detailed financial data from Alpha Vantage API including earnings, news sentiment.
    Input should be a stock ticker symbol."""
    
    def __init__(self):
        super().__init__()
        self.api_key = config.financial.alpha_vantage_api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    @track_tool_usage("AlphaVantageDataTool")
    def _run(self, ticker: str) -> str:
        """Get Alpha Vantage data"""
        if not self.api_key:
            return "Alpha Vantage API key not configured"
        
        try:
            ticker = ticker.upper().strip()
            
            # Get company overview
            overview = self._get_company_overview(ticker)
            
            # Get earnings data
            earnings = self._get_earnings_data(ticker)
            
            result = f"Alpha Vantage Data for {ticker}:\n\n"
            result += "COMPANY OVERVIEW:\n"
            result += overview + "\n\n"
            result += "RECENT EARNINGS:\n"
            result += earnings
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Alpha Vantage tool: {str(e)}")
            return f"Error retrieving Alpha Vantage data: {str(e)}"
    
    def _get_company_overview(self, ticker: str) -> str:
        """Get company overview from Alpha Vantage"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'Symbol' not in data:
                return "Company overview not available"
            
            overview = f"""
Company: {data.get('Name', 'N/A')}
Sector: {data.get('Sector', 'N/A')} | Industry: {data.get('Industry', 'N/A')}
Market Cap: {data.get('MarketCapitalization', 'N/A')}
P/E Ratio: {data.get('PERatio', 'N/A')} | P/B Ratio: {data.get('PriceToBookRatio', 'N/A')}
Dividend Yield: {data.get('DividendYield', 'N/A')}
ROE: {data.get('ReturnOnEquityTTM', 'N/A')} | Profit Margin: {data.get('ProfitMargin', 'N/A')}
            """.strip()
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting company overview: {str(e)}")
            return "Error retrieving company overview"
    
    def _get_earnings_data(self, ticker: str) -> str:
        """Get earnings data from Alpha Vantage"""
        try:
            params = {
                'function': 'EARNINGS',
                'symbol': ticker,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            if 'quarterlyEarnings' not in data:
                return "Earnings data not available"
            
            earnings = "Recent Quarterly Earnings:\n"
            for quarter in data['quarterlyEarnings'][:4]:  # Last 4 quarters
                earnings += f"Q{quarter.get('fiscalDateEnding', 'N/A')}: "
                earnings += f"EPS ${quarter.get('reportedEPS', 'N/A')} "
                earnings += f"(Est: ${quarter.get('estimatedEPS', 'N/A')})\n"
            
            return earnings
            
        except Exception as e:
            logger.error(f"Error getting earnings data: {str(e)}")
            return "Error retrieving earnings data"