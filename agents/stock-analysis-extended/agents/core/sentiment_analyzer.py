"""Sentiment Analysis Agent for market sentiment and social media analysis"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from duckduckgo_search import DDGS
import requests
import json

from agents.base import BaseStockAgent, AgentResult
from crewai.tools import BaseTool
from config.settings import settings


class NewsSentimentTool(BaseTool):
    name: str = "NewsSentimentAnalyzer"
    description: str = "Analyze sentiment from news articles about stocks"
    
    def __init__(self):
        super().__init__()
        self.vader = SentimentIntensityAnalyzer()
    
    def _run(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Analyze news sentiment"""
        try:
            # Search for news using DuckDuckGo
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{query} stock news", max_results=max_results))
            
            if not results:
                return {"error": "No news found", "query": query}
            
            sentiments = []
            for article in results:
                # Analyze sentiment using VADER
                text = article.get('body', '')
                vader_scores = self.vader.polarity_scores(text)
                
                # Analyze sentiment using TextBlob
                blob = TextBlob(text)
                
                sentiments.append({
                    'title': article.get('title', ''),
                    'snippet': text[:200],
                    'vader_compound': vader_scores['compound'],
                    'vader_positive': vader_scores['pos'],
                    'vader_negative': vader_scores['neg'],
                    'vader_neutral': vader_scores['neu'],
                    'textblob_polarity': blob.sentiment.polarity,
                    'textblob_subjectivity': blob.sentiment.subjectivity
                })
            
            # Calculate aggregate sentiment
            avg_vader = sum(s['vader_compound'] for s in sentiments) / len(sentiments)
            avg_textblob = sum(s['textblob_polarity'] for s in sentiments) / len(sentiments)
            
            # Determine overall sentiment
            if avg_vader > 0.1:
                overall_sentiment = "POSITIVE"
            elif avg_vader < -0.1:
                overall_sentiment = "NEGATIVE"
            else:
                overall_sentiment = "NEUTRAL"
            
            return {
                'query': query,
                'article_count': len(sentiments),
                'sentiments': sentiments,
                'aggregate_scores': {
                    'vader_average': round(avg_vader, 3),
                    'textblob_average': round(avg_textblob, 3),
                    'overall_sentiment': overall_sentiment
                },
                'sentiment_distribution': self._calculate_distribution(sentiments)
            }
            
        except Exception as e:
            return {"error": str(e), "query": query}
    
    def _calculate_distribution(self, sentiments: List[Dict]) -> Dict[str, int]:
        """Calculate sentiment distribution"""
        positive = sum(1 for s in sentiments if s['vader_compound'] > 0.1)
        negative = sum(1 for s in sentiments if s['vader_compound'] < -0.1)
        neutral = len(sentiments) - positive - negative
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'positive_percentage': round(positive / len(sentiments) * 100, 1),
            'negative_percentage': round(negative / len(sentiments) * 100, 1),
            'neutral_percentage': round(neutral / len(sentiments) * 100, 1)
        }


class SocialMediaSentimentTool(BaseTool):
    name: str = "SocialMediaSentimentAnalyzer"
    description: str = "Analyze sentiment from social media platforms"
    
    def __init__(self):
        super().__init__()
        self.vader = SentimentIntensityAnalyzer()
    
    def _run(self, ticker: str, platform: str = "reddit") -> Dict[str, Any]:
        """Analyze social media sentiment"""
        try:
            if platform == "reddit":
                return self._analyze_reddit_sentiment(ticker)
            elif platform == "stocktwits":
                return self._analyze_stocktwits_sentiment(ticker)
            else:
                # Fallback to web search for social sentiment
                return self._analyze_web_social_sentiment(ticker)
                
        except Exception as e:
            return {"error": str(e), "ticker": ticker, "platform": platform}
    
    def _analyze_reddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze Reddit sentiment (simulated without API)"""
        # Search for Reddit discussions
        with DDGS() as ddgs:
            results = list(ddgs.text(f"site:reddit.com {ticker} stock", max_results=10))
        
        if not results:
            return {"error": "No Reddit data found", "ticker": ticker}
        
        sentiments = []
        for post in results:
            text = post.get('body', '')
            scores = self.vader.polarity_scores(text)
            sentiments.append({
                'text': text[:200],
                'compound': scores['compound'],
                'source': 'reddit'
            })
        
        avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            'ticker': ticker,
            'platform': 'reddit',
            'post_count': len(sentiments),
            'average_sentiment': round(avg_sentiment, 3),
            'sentiment_label': self._get_sentiment_label(avg_sentiment),
            'posts': sentiments[:5]  # Return top 5 posts
        }
    
    def _analyze_stocktwits_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Analyze StockTwits sentiment"""
        # StockTwits API endpoint (public, no auth required for basic data)
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return self._analyze_web_social_sentiment(ticker)
            
            data = response.json()
            messages = data.get('messages', [])
            
            if not messages:
                return {"error": "No StockTwits data", "ticker": ticker}
            
            sentiments = []
            for msg in messages[:20]:  # Analyze last 20 messages
                text = msg.get('body', '')
                scores = self.vader.polarity_scores(text)
                
                sentiments.append({
                    'text': text[:200],
                    'compound': scores['compound'],
                    'likes': msg.get('likes', {}).get('total', 0),
                    'sentiment_label': msg.get('entities', {}).get('sentiment', {}).get('basic', 'neutral')
                })
            
            avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments)
            
            return {
                'ticker': ticker,
                'platform': 'stocktwits',
                'message_count': len(sentiments),
                'average_sentiment': round(avg_sentiment, 3),
                'sentiment_label': self._get_sentiment_label(avg_sentiment),
                'messages': sentiments[:5]
            }
            
        except Exception:
            return self._analyze_web_social_sentiment(ticker)
    
    def _analyze_web_social_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Fallback web-based social sentiment analysis"""
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{ticker} stock forum discussion", max_results=10))
        
        if not results:
            return {"error": "No social data found", "ticker": ticker}
        
        sentiments = []
        for post in results:
            text = post.get('body', '')
            scores = self.vader.polarity_scores(text)
            sentiments.append({
                'text': text[:200],
                'compound': scores['compound'],
                'source': 'web'
            })
        
        avg_sentiment = sum(s['compound'] for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            'ticker': ticker,
            'platform': 'web_forums',
            'post_count': len(sentiments),
            'average_sentiment': round(avg_sentiment, 3),
            'sentiment_label': self._get_sentiment_label(avg_sentiment)
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return "BULLISH"
        elif score < -0.1:
            return "BEARISH"
        else:
            return "NEUTRAL"


class TrendAnalysisTool(BaseTool):
    name: str = "TrendAnalyzer"
    description: str = "Analyze trending topics and momentum for stocks"
    
    def _run(self, ticker: str) -> Dict[str, Any]:
        """Analyze trending metrics"""
        try:
            # Search for trending discussions
            with DDGS() as ddgs:
                recent_results = list(ddgs.text(f"{ticker} stock", max_results=20, timelimit='d'))
                week_results = list(ddgs.text(f"{ticker} stock", max_results=20, timelimit='w'))
            
            # Calculate momentum (recent vs week)
            recent_count = len(recent_results)
            week_count = len(week_results)
            
            momentum = "INCREASING" if recent_count > week_count * 0.3 else "STABLE"
            if recent_count > week_count * 0.5:
                momentum = "RAPIDLY INCREASING"
            elif recent_count < week_count * 0.1:
                momentum = "DECREASING"
            
            # Extract key topics
            all_text = ' '.join([r.get('body', '') for r in recent_results])
            topics = self._extract_topics(all_text)
            
            return {
                'ticker': ticker,
                'daily_mentions': recent_count,
                'weekly_mentions': week_count,
                'momentum': momentum,
                'trending_topics': topics,
                'buzz_score': self._calculate_buzz_score(recent_count, week_count)
            }
            
        except Exception as e:
            return {"error": str(e), "ticker": ticker}
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple keyword extraction
        keywords = ['earnings', 'merger', 'acquisition', 'dividend', 'buyback', 
                   'guidance', 'revenue', 'profit', 'loss', 'growth', 'decline',
                   'upgrade', 'downgrade', 'analyst', 'target', 'price']
        
        text_lower = text.lower()
        found_topics = [kw for kw in keywords if kw in text_lower]
        
        return found_topics[:5]  # Return top 5 topics
    
    def _calculate_buzz_score(self, recent: int, weekly: int) -> float:
        """Calculate buzz score (0-100)"""
        # Simple buzz calculation
        base_score = min(recent * 5, 50)  # Up to 50 points for volume
        momentum_bonus = 0
        
        if recent > weekly * 0.3:
            momentum_bonus = 20
        if recent > weekly * 0.5:
            momentum_bonus = 50
        
        return min(base_score + momentum_bonus, 100)


class SentimentAnalysisAgent(BaseStockAgent):
    """Agent specialized in sentiment analysis from multiple sources"""
    
    def __init__(self, **kwargs):
        # Initialize tools
        self.news_tool = NewsSentimentTool()
        self.social_tool = SocialMediaSentimentTool()
        self.trend_tool = TrendAnalysisTool()
        
        super().__init__(
            name="SentimentAnalysisAgent",
            role="Sentiment Analyst",
            goal="Analyze market sentiment from news, social media, and online discussions",
            backstory="""You are an expert in natural language processing and sentiment analysis 
            with a focus on financial markets. You specialize in extracting insights from news 
            articles, social media posts, and online forums to gauge market sentiment and investor 
            psychology. Your analysis helps predict market movements based on public sentiment.""",
            tools=[self.news_tool, self.social_tool, self.trend_tool],
            **kwargs
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return 'ticker' in input_data
    
    async def analyze(self, input_data: Dict[str, Any]) -> AgentResult:
        """Perform comprehensive sentiment analysis"""
        ticker = input_data.get('ticker')
        company_name = input_data.get('company_name', ticker)
        
        try:
            # Gather sentiment from multiple sources in parallel
            tasks = [
                self._analyze_news_sentiment(f"{company_name} {ticker}"),
                self._analyze_social_sentiment(ticker, 'reddit'),
                self._analyze_social_sentiment(ticker, 'stocktwits'),
                self._analyze_trends(ticker)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            sentiment_data = {
                'news_sentiment': results[0] if not isinstance(results[0], Exception) else None,
                'reddit_sentiment': results[1] if not isinstance(results[1], Exception) else None,
                'stocktwits_sentiment': results[2] if not isinstance(results[2], Exception) else None,
                'trend_analysis': results[3] if not isinstance(results[3], Exception) else None
            }
            
            # Calculate composite sentiment score
            composite_score = self._calculate_composite_sentiment(sentiment_data)
            
            # Generate sentiment summary
            sentiment_summary = self._generate_sentiment_summary(sentiment_data, composite_score)
            
            # Determine sentiment signals
            signals = self._generate_sentiment_signals(composite_score, sentiment_data)
            
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'sentiment_analysis'),
                status='completed',
                data={
                    'ticker': ticker,
                    'sentiment_sources': sentiment_data,
                    'composite_sentiment': composite_score,
                    'sentiment_summary': sentiment_summary,
                    'signals': signals,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                task_id=input_data.get('task_id', 'sentiment_analysis'),
                status='failed',
                errors=[str(e)]
            )
    
    async def _analyze_news_sentiment(self, query: str) -> Dict[str, Any]:
        """Analyze news sentiment"""
        return await asyncio.to_thread(self.news_tool._run, query)
    
    async def _analyze_social_sentiment(self, ticker: str, platform: str) -> Dict[str, Any]:
        """Analyze social media sentiment"""
        return await asyncio.to_thread(self.social_tool._run, ticker, platform)
    
    async def _analyze_trends(self, ticker: str) -> Dict[str, Any]:
        """Analyze trending metrics"""
        return await asyncio.to_thread(self.trend_tool._run, ticker)
    
    def _calculate_composite_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite sentiment score from all sources"""
        scores = []
        weights = []
        
        # News sentiment (weight: 40%)
        if sentiment_data.get('news_sentiment'):
            news_score = sentiment_data['news_sentiment'].get('aggregate_scores', {}).get('vader_average', 0)
            scores.append(news_score)
            weights.append(0.4)
        
        # Reddit sentiment (weight: 20%)
        if sentiment_data.get('reddit_sentiment'):
            reddit_score = sentiment_data['reddit_sentiment'].get('average_sentiment', 0)
            scores.append(reddit_score)
            weights.append(0.2)
        
        # StockTwits sentiment (weight: 20%)
        if sentiment_data.get('stocktwits_sentiment'):
            st_score = sentiment_data['stocktwits_sentiment'].get('average_sentiment', 0)
            scores.append(st_score)
            weights.append(0.2)
        
        # Trend momentum (weight: 20%)
        if sentiment_data.get('trend_analysis'):
            buzz = sentiment_data['trend_analysis'].get('buzz_score', 50) / 100
            # Convert buzz to sentiment (-1 to 1)
            trend_score = (buzz - 0.5) * 2
            scores.append(trend_score)
            weights.append(0.2)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            weighted_score = 0
        
        # Determine sentiment category
        if weighted_score > 0.2:
            category = "VERY BULLISH"
        elif weighted_score > 0.05:
            category = "BULLISH"
        elif weighted_score < -0.2:
            category = "VERY BEARISH"
        elif weighted_score < -0.05:
            category = "BEARISH"
        else:
            category = "NEUTRAL"
        
        return {
            'score': round(weighted_score, 3),
            'category': category,
            'confidence': self._calculate_confidence(sentiment_data)
        }
    
    def _calculate_confidence(self, sentiment_data: Dict[str, Any]) -> str:
        """Calculate confidence level based on data availability and consistency"""
        available_sources = sum(1 for v in sentiment_data.values() if v and not isinstance(v, dict) or (isinstance(v, dict) and 'error' not in v))
        
        if available_sources >= 3:
            return "HIGH"
        elif available_sources >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_sentiment_summary(self, sentiment_data: Dict[str, Any], composite: Dict[str, Any]) -> str:
        """Generate textual sentiment summary"""
        summary_parts = []
        
        # Overall sentiment
        summary_parts.append(f"Overall market sentiment is {composite['category']} with a score of {composite['score']}")
        
        # News sentiment
        if sentiment_data.get('news_sentiment'):
            news = sentiment_data['news_sentiment'].get('aggregate_scores', {})
            if news:
                summary_parts.append(f"News sentiment is {news.get('overall_sentiment', 'UNKNOWN')}")
        
        # Social sentiment
        if sentiment_data.get('reddit_sentiment'):
            reddit = sentiment_data['reddit_sentiment']
            summary_parts.append(f"Reddit community sentiment is {reddit.get('sentiment_label', 'UNKNOWN')}")
        
        # Trend analysis
        if sentiment_data.get('trend_analysis'):
            trend = sentiment_data['trend_analysis']
            summary_parts.append(f"Discussion momentum is {trend.get('momentum', 'UNKNOWN')}")
        
        return ". ".join(summary_parts)
    
    def _generate_sentiment_signals(self, composite: Dict[str, Any], sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate trading signals based on sentiment"""
        signals = []
        
        score = composite['score']
        
        # Strong signals
        if score > 0.3:
            signals.append("STRONG BUY signal from overwhelmingly positive sentiment")
        elif score > 0.1:
            signals.append("BUY signal from positive market sentiment")
        elif score < -0.3:
            signals.append("STRONG SELL signal from overwhelmingly negative sentiment")
        elif score < -0.1:
            signals.append("SELL signal from negative market sentiment")
        else:
            signals.append("HOLD signal from neutral market sentiment")
        
        # Momentum signals
        if sentiment_data.get('trend_analysis'):
            momentum = sentiment_data['trend_analysis'].get('momentum', '')
            if 'RAPIDLY INCREASING' in momentum:
                signals.append("Momentum signal: Rapidly increasing interest may indicate breakout")
            elif 'DECREASING' in momentum:
                signals.append("Caution: Decreasing interest may indicate waning momentum")
        
        # Divergence signals
        if sentiment_data.get('news_sentiment') and sentiment_data.get('reddit_sentiment'):
            news_score = sentiment_data['news_sentiment'].get('aggregate_scores', {}).get('vader_average', 0)
            social_score = sentiment_data['reddit_sentiment'].get('average_sentiment', 0)
            
            if abs(news_score - social_score) > 0.5:
                signals.append("Divergence detected between news and social sentiment")
        
        return signals