"""Risk metrics calculation tools"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
from scipy import stats


class RiskMetricsCalculator:
    """Calculate various risk metrics for stocks and portfolios"""
    
    def calculate_risk_ratios(self, ticker: str, benchmark: str = 'SPY', period: str = '1y') -> Dict[str, float]:
        """Calculate Sharpe, Sortino, and other risk-adjusted return ratios"""
        try:
            # Get data
            stock = yf.Ticker(ticker)
            bench = yf.Ticker(benchmark)
            
            stock_hist = stock.history(period=period)
            bench_hist = bench.history(period=period)
            
            if stock_hist.empty or bench_hist.empty:
                return {"error": "Insufficient data"}
            
            # Calculate returns
            stock_returns = stock_hist['Close'].pct_change().dropna()
            bench_returns = bench_hist['Close'].pct_change().dropna()
            
            # Risk-free rate (approximate using 3-month T-bill)
            risk_free_rate = 0.05 / 252  # Daily risk-free rate
            
            # Sharpe Ratio
            excess_returns = stock_returns - risk_free_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / stock_returns.std()
            
            # Sortino Ratio (downside deviation)
            downside_returns = stock_returns[stock_returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else stock_returns.std()
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std
            
            # Calmar Ratio (return over max drawdown)
            annual_return = (1 + stock_returns.mean()) ** 252 - 1
            max_dd = self._calculate_max_drawdown(stock_hist['Close'])
            calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
            
            # Information Ratio
            active_returns = stock_returns - bench_returns[:len(stock_returns)]
            tracking_error = active_returns.std()
            information_ratio = np.sqrt(252) * active_returns.mean() / tracking_error if tracking_error > 0 else 0
            
            return {
                'sharpe_ratio': round(sharpe_ratio, 3),
                'sortino_ratio': round(sortino_ratio, 3),
                'calmar_ratio': round(calmar_ratio, 3),
                'information_ratio': round(information_ratio, 3)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_beta_alpha(self, ticker: str, benchmark: str = 'SPY', period: str = '1y') -> Dict[str, float]:
        """Calculate beta and alpha relative to benchmark"""
        try:
            # Get data
            stock = yf.Ticker(ticker)
            bench = yf.Ticker(benchmark)
            
            stock_hist = stock.history(period=period)
            bench_hist = bench.history(period=period)
            
            if stock_hist.empty or bench_hist.empty:
                return {"error": "Insufficient data"}
            
            # Calculate returns
            stock_returns = stock_hist['Close'].pct_change().dropna()
            bench_returns = bench_hist['Close'].pct_change().dropna()
            
            # Align data
            aligned_data = pd.DataFrame({
                'stock': stock_returns,
                'benchmark': bench_returns
            }).dropna()
            
            # Calculate beta using covariance
            covariance = aligned_data['stock'].cov(aligned_data['benchmark'])
            benchmark_variance = aligned_data['benchmark'].var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            # Calculate alpha (Jensen's alpha)
            risk_free_rate = 0.05  # Annual risk-free rate
            stock_annual_return = (1 + aligned_data['stock'].mean()) ** 252 - 1
            bench_annual_return = (1 + aligned_data['benchmark'].mean()) ** 252 - 1
            
            alpha = stock_annual_return - (risk_free_rate + beta * (bench_annual_return - risk_free_rate))
            
            # R-squared (goodness of fit)
            correlation = aligned_data['stock'].corr(aligned_data['benchmark'])
            r_squared = correlation ** 2
            
            return {
                'beta': round(beta, 3),
                'alpha': round(alpha * 100, 3),  # As percentage
                'r_squared': round(r_squared, 3),
                'correlation': round(correlation, 3)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def calculate_portfolio_metrics(self, tickers: list, weights: Optional[list] = None, period: str = '1y') -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics"""
        try:
            if weights is None:
                weights = [1/len(tickers)] * len(tickers)
            
            # Ensure weights sum to 1
            weights = np.array(weights) / sum(weights)
            
            # Get data for all tickers
            data = {}
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[ticker] = hist['Close']
            
            if not data:
                return {"error": "No data available"}
            
            # Create DataFrame and calculate returns
            df = pd.DataFrame(data)
            returns = df.pct_change().dropna()
            
            # Portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Portfolio metrics
            annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
            annual_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Correlation matrix
            correlation_matrix = returns.corr()
            
            # Covariance matrix
            covariance_matrix = returns.cov() * 252  # Annualized
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            # Diversification ratio
            weighted_avg_volatility = sum(weights * returns.std() * np.sqrt(252))
            diversification_ratio = weighted_avg_volatility / portfolio_std if portfolio_std > 0 else 1
            
            return {
                'annual_return': round(annual_return * 100, 2),
                'annual_volatility': round(annual_volatility * 100, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'diversification_ratio': round(diversification_ratio, 3),
                'portfolio_variance': round(portfolio_variance, 4),
                'effective_number_of_stocks': round(1 / sum(weights**2), 2)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()