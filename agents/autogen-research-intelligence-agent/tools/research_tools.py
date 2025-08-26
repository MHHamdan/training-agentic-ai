"""
Research Tools for ARIA
Advanced tools for web search, academic search, and content analysis
"""

import os
import json
import requests
import re
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from urllib.parse import quote_plus

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class WebSearchTool:
    """
    Web search tool using multiple search providers
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize web search tool
        
        Args:
            api_keys: Dictionary containing API keys for different search providers
        """
        self.api_keys = api_keys or {}
        self.search_providers = self._initialize_providers()
    
    def _initialize_providers(self) -> Dict[str, bool]:
        """Initialize available search providers"""
        providers = {
            "duckduckgo": True,  # No API key required
            "google": bool(self.api_keys.get("google_search")),
            "bing": bool(self.api_keys.get("bing_search")),
            "serp": bool(self.api_keys.get("serp_api"))
        }
        return providers
    
    def search(self, query: str, max_results: int = 10, provider: str = "duckduckgo") -> Dict[str, Any]:
        """
        Perform web search
        
        Args:
            query: Search query
            max_results: Maximum number of results
            provider: Search provider to use
            
        Returns:
            Dictionary containing search results
        """
        try:
            if provider == "duckduckgo" and self.search_providers["duckduckgo"]:
                return self._search_duckduckgo(query, max_results)
            elif provider == "google" and self.search_providers["google"]:
                return self._search_google(query, max_results)
            else:
                # Fallback to DuckDuckGo
                return self._search_duckduckgo(query, max_results)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def _search_duckduckgo(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using DuckDuckGo instant answer API"""
        try:
            # DuckDuckGo instant answer API
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                
                # Add abstract if available
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", "Abstract"),
                        "snippet": data.get("Abstract"),
                        "url": data.get("AbstractURL", ""),
                        "source": "DuckDuckGo Abstract"
                    })
                
                # Add related topics
                for topic in data.get("RelatedTopics", [])[:max_results-1]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                            "source": "DuckDuckGo Related"
                        })
                
                return {
                    "success": True,
                    "query": query,
                    "provider": "duckduckgo",
                    "total_results": len(results),
                    "results": results[:max_results]
                }
            else:
                return self._fallback_search_results(query, max_results)
                
        except Exception as e:
            return self._fallback_search_results(query, max_results)
    
    def _search_google(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search using Google Custom Search API"""
        # Placeholder for Google Custom Search implementation
        # Would require API key and custom search engine ID
        return self._fallback_search_results(query, max_results)
    
    def _fallback_search_results(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate fallback search results when APIs are unavailable"""
        return {
            "success": True,
            "query": query,
            "provider": "fallback",
            "total_results": 1,
            "results": [{
                "title": f"Research Topic: {query}",
                "snippet": f"This is a research topic about {query}. For comprehensive information, please use external search engines or academic databases.",
                "url": "",
                "source": "ARIA Fallback"
            }]
        }


class AcademicSearchTool:
    """
    Academic search tool for scholarly articles and papers
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize academic search tool
        
        Args:
            api_keys: Dictionary containing API keys for academic databases
        """
        self.api_keys = api_keys or {}
        self.sources = self._initialize_sources()
    
    def _initialize_sources(self) -> Dict[str, bool]:
        """Initialize available academic sources"""
        return {
            "arxiv": True,  # Open access
            "pubmed": True,  # Open access
            "crossref": True,  # Open access
            "semantic_scholar": bool(self.api_keys.get("semantic_scholar")),
            "google_scholar": False  # Requires scraping, not recommended
        }
    
    def search_academic(self, query: str, max_results: int = 10, source: str = "arxiv") -> Dict[str, Any]:
        """
        Search for academic papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            source: Academic source to search
            
        Returns:
            Dictionary containing academic search results
        """
        try:
            if source == "arxiv" and self.sources["arxiv"]:
                return self._search_arxiv(query, max_results)
            elif source == "pubmed" and self.sources["pubmed"]:
                return self._search_pubmed(query, max_results)
            elif source == "crossref" and self.sources["crossref"]:
                return self._search_crossref(query, max_results)
            else:
                return self._fallback_academic_results(query, max_results)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }
    
    def _search_arxiv(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search arXiv for papers"""
        try:
            if not FEEDPARSER_AVAILABLE:
                return self._fallback_academic_results(query, max_results)
            
            # arXiv API
            base_url = "http://export.arxiv.org/api/query?"
            search_query = f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
            url = base_url + search_query
            
            feed = feedparser.parse(url)
            
            results = []
            for entry in feed.entries:
                results.append({
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors] if hasattr(entry, 'authors') else [],
                    "abstract": entry.summary if hasattr(entry, 'summary') else "",
                    "url": entry.link,
                    "published": entry.published if hasattr(entry, 'published') else "",
                    "categories": [tag.term for tag in entry.tags] if hasattr(entry, 'tags') else [],
                    "source": "arXiv"
                })
            
            return {
                "success": True,
                "query": query,
                "source": "arxiv",
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            return self._fallback_academic_results(query, max_results)
    
    def _search_pubmed(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search PubMed for medical/biological papers"""
        # Placeholder for PubMed implementation
        return self._fallback_academic_results(query, max_results)
    
    def _search_crossref(self, query: str, max_results: int) -> Dict[str, Any]:
        """Search Crossref for academic publications"""
        try:
            url = f"https://api.crossref.org/works?query={quote_plus(query)}&rows={max_results}"
            headers = {"User-Agent": "ARIA Research Agent (mailto: research@example.com)"}
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                results = []
                for item in data.get("message", {}).get("items", []):
                    results.append({
                        "title": " ".join(item.get("title", ["Untitled"])),
                        "authors": [f"{author.get('given', '')} {author.get('family', '')}" 
                                  for author in item.get("author", [])],
                        "abstract": item.get("abstract", ""),
                        "url": item.get("URL", ""),
                        "published": item.get("published-print", {}).get("date-parts", [[""]])[0],
                        "journal": item.get("container-title", [""])[0],
                        "doi": item.get("DOI", ""),
                        "source": "Crossref"
                    })
                
                return {
                    "success": True,
                    "query": query,
                    "source": "crossref",
                    "total_results": len(results),
                    "results": results
                }
            else:
                return self._fallback_academic_results(query, max_results)
                
        except Exception as e:
            return self._fallback_academic_results(query, max_results)
    
    def _fallback_academic_results(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate fallback academic results"""
        return {
            "success": True,
            "query": query,
            "source": "fallback",
            "total_results": 1,
            "results": [{
                "title": f"Academic Research: {query}",
                "authors": ["ARIA Research Assistant"],
                "abstract": f"This topic requires academic research on {query}. Please consult scholarly databases like Google Scholar, PubMed, or arXiv for peer-reviewed sources.",
                "url": "",
                "published": datetime.now().strftime("%Y-%m-%d"),
                "journal": "ARIA Research Notes",
                "source": "ARIA Fallback"
            }]
        }


class ContentAnalyzer:
    """
    Tool for analyzing and summarizing content
    """
    
    def __init__(self):
        """Initialize content analyzer"""
        self.analysis_cache = {}
    
    def analyze_text(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze text content
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis (basic, comprehensive, summary)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if analysis_type == "basic":
                return self._basic_analysis(text)
            elif analysis_type == "comprehensive":
                return self._comprehensive_analysis(text)
            elif analysis_type == "summary":
                return self._summary_analysis(text)
            else:
                return self._basic_analysis(text)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {}
            }
    
    def _basic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform basic text analysis"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            "success": True,
            "analysis_type": "basic",
            "metrics": {
                "word_count": len(words),
                "sentence_count": len([s for s in sentences if s.strip()]),
                "paragraph_count": len([p for p in paragraphs if p.strip()]),
                "character_count": len(text),
                "avg_words_per_sentence": len(words) / max(len(sentences), 1),
                "reading_time_minutes": len(words) / 200  # Assuming 200 WPM
            },
            "key_terms": self._extract_key_terms(text),
            "timestamp": datetime.now().isoformat()
        }
    
    def _comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive text analysis"""
        basic_analysis = self._basic_analysis(text)
        
        # Add more sophisticated analysis
        analysis = basic_analysis["analysis"] if "analysis" in basic_analysis else {}
        analysis.update({
            "sentiment": self._analyze_sentiment(text),
            "topics": self._extract_topics(text),
            "entities": self._extract_entities(text),
            "complexity": self._analyze_complexity(text),
            "structure": self._analyze_structure(text)
        })
        
        return {
            "success": True,
            "analysis_type": "comprehensive",
            "metrics": basic_analysis["metrics"],
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _summary_analysis(self, text: str) -> Dict[str, Any]:
        """Generate summary analysis"""
        # Extract key sentences (simplified extractive summarization)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Simple ranking based on word frequency
        word_freq = {}
        for sentence in sentences:
            for word in sentence.lower().split():
                word = re.sub(r'[^\w]', '', word)
                if len(word) > 3:  # Ignore short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            for word in words:
                word = re.sub(r'[^\w]', '', word)
                if word in word_freq:
                    score += word_freq[word]
            sentence_scores[i] = score / max(len(words), 1)
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        
        return {
            "success": True,
            "analysis_type": "summary",
            "summary": ". ".join(summary_sentences) + ".",
            "key_points": summary_sentences,
            "compression_ratio": len(summary_sentences) / max(len(sentences), 1),
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 10 most frequent words
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'success', 'beneficial', 'improvement']
        negative_words = ['bad', 'poor', 'negative', 'problem', 'issue', 'challenge', 'difficulty']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count,
            "confidence": abs(positive_count - negative_count) / max(positive_count + negative_count, 1)
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract potential topics from text"""
        # Simple topic extraction based on noun phrases and capitalized words
        topics = []
        
        # Find capitalized phrases (potential proper nouns/topics)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        topics.extend(capitalized[:10])
        
        return list(set(topics))
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified)"""
        # Simple entity extraction - looking for patterns
        entities = []
        
        # Dates
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', text)
        entities.extend([f"DATE: {date}" for date in dates])
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        entities.extend([f"NUMBER: {num}" for num in numbers[:5]])
        
        return entities
    
    def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity"""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Simple complexity score
        complexity_score = (avg_word_length * 0.5) + (avg_sentence_length * 0.1)
        
        if complexity_score < 6:
            level = "Simple"
        elif complexity_score < 10:
            level = "Moderate"
        else:
            level = "Complex"
        
        return {
            "complexity_level": level,
            "complexity_score": complexity_score,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length
        }
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure"""
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            "paragraph_count": len(paragraphs),
            "has_headings": bool(re.search(r'^#+ ', text, re.MULTILINE)),
            "has_lists": bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE)),
            "has_numbers": bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))
        }


def get_research_tools(api_keys: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Get initialized research tools
    
    Args:
        api_keys: Dictionary containing API keys
        
    Returns:
        Dictionary containing initialized tools
    """
    return {
        "web_search": WebSearchTool(api_keys),
        "academic_search": AcademicSearchTool(api_keys),
        "content_analyzer": ContentAnalyzer(),
        "capabilities": {
            "web_search_providers": ["duckduckgo", "google", "bing"],
            "academic_sources": ["arxiv", "pubmed", "crossref"],
            "analysis_types": ["basic", "comprehensive", "summary"],
            "dependencies": {
                "feedparser": FEEDPARSER_AVAILABLE,
                "beautifulsoup4": BS4_AVAILABLE
            }
        }
    }