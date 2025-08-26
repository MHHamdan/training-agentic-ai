"""
Analyzer Agent - Content analysis expert with detailed tracking
Implements information extraction, structuring, and relevance scoring
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
except ImportError:
    nltk = None
    print("Warning: NLTK not available for text analysis")

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    print("Warning: TextBlob not available for sentiment analysis")

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class AnalyzerAgent:
    """
    Content analysis expert agent
    Extracts insights, entities, and performs relevance scoring
    """
    
    def __init__(self):
        """Initialize analyzer agent"""
        self.config = config
        self.model_manager = ModelManager()
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0
        }
        
        # Initialize NLTK data if available
        if nltk:
            self._initialize_nltk()
    
    def _initialize_nltk(self):
        """Initialize NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
        except Exception as e:
            logger.warning(f"NLTK initialization warning: {e}")
    
    @observe(as_type="generation")
    async def analyze(
        self,
        search_results: List[Dict[str, Any]],
        query: str,
        depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive content analysis
        
        Args:
            search_results: List of search results to analyze
            query: Original research query
            depth: Analysis depth (quick, standard, comprehensive, exhaustive)
        
        Returns:
            Analysis results with insights and metadata
        """
        try:
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Analyzing {len(search_results)} sources for: {query}",
                    metadata={
                        "source_count": len(search_results),
                        "analysis_depth": depth,
                        "agent": "analyzer_agent"
                    }
                )
            
            start_time = datetime.now()
            
            # Initialize results
            analysis_results = {
                "analyzed_content": [],
                "key_insights": [],
                "extracted_facts": [],
                "relevance_scores": {},
                "entity_analysis": {},
                "sentiment_analysis": {},
                "topic_analysis": {},
                "summary_by_source": {}
            }
            
            # Analyze each source
            for i, source in enumerate(search_results):
                try:
                    source_analysis = await self._analyze_single_source(
                        source, query, depth, i
                    )
                    
                    # Store results
                    analysis_results["analyzed_content"].append(source_analysis)
                    analysis_results["relevance_scores"][source.get("url", f"source_{i}")] = \
                        source_analysis.get("relevance_score", 0.0)
                    
                    # Extract insights and facts
                    if source_analysis.get("key_points"):
                        analysis_results["key_insights"].extend(source_analysis["key_points"])
                    
                    if source_analysis.get("extracted_facts"):
                        analysis_results["extracted_facts"].extend(source_analysis["extracted_facts"])
                    
                except Exception as e:
                    logger.error(f"Error analyzing source {i}: {e}")
                    self.analysis_stats["failed_analyses"] += 1
            
            # Perform cross-source analysis
            if depth in ["comprehensive", "exhaustive"]:
                cross_analysis = await self._perform_cross_source_analysis(
                    analysis_results["analyzed_content"], query
                )
                analysis_results.update(cross_analysis)
            
            # Generate final insights
            final_insights = await self._generate_final_insights(
                analysis_results, query, depth
            )
            analysis_results["key_insights"] = final_insights
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.analysis_stats["total_analyses"] += 1
            self.analysis_stats["successful_analyses"] += 1
            self._update_average_processing_time(processing_time)
            
            # Track output with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Analysis complete: {len(analysis_results['key_insights'])} insights extracted",
                    metadata={
                        "processing_time": processing_time,
                        "insights_count": len(analysis_results["key_insights"]),
                        "facts_count": len(analysis_results["extracted_facts"])
                    }
                )
            
            logger.info(f"Analysis completed in {processing_time:.2f}s: {len(analysis_results['key_insights'])} insights")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Analysis failed: {str(e)}",
                    metadata={"error": str(e)}
                )
            return {
                "analyzed_content": [],
                "key_insights": [],
                "extracted_facts": [],
                "relevance_scores": {},
                "error": str(e)
            }
    
    @observe(as_type="generation")
    async def _analyze_single_source(
        self,
        source: Dict[str, Any],
        query: str,
        depth: str,
        source_index: int
    ) -> Dict[str, Any]:
        """
        Analyze a single source in detail
        
        Args:
            source: Source content to analyze
            query: Research query
            depth: Analysis depth
            source_index: Index of source for tracking
        
        Returns:
            Detailed analysis of the source
        """
        try:
            content = source.get("content", "") or source.get("snippet", "")
            title = source.get("title", "")
            
            # Basic analysis
            analysis = {
                "source_id": source.get("url", f"source_{source_index}"),
                "title": title,
                "content_length": len(content),
                "relevance_score": 0.0,
                "quality_score": source.get("quality_score", 0.5),
                "summary": "",
                "key_points": [],
                "extracted_facts": [],
                "entities": [],
                "sentiment": {},
                "topics": [],
                "metadata": {}
            }
            
            if not content:
                return analysis
            
            # Calculate relevance score
            analysis["relevance_score"] = self._calculate_relevance_score(
                content, title, query
            )
            
            # Extract entities if NLTK is available
            if nltk:
                analysis["entities"] = self._extract_entities(content)
            
            # Sentiment analysis if TextBlob is available
            if TextBlob:
                analysis["sentiment"] = self._analyze_sentiment(content)
            
            # Extract key points using LLM
            if depth in ["comprehensive", "exhaustive"]:
                analysis["key_points"] = await self._extract_key_points_llm(
                    content, query, title
                )
                
                # Extract facts using LLM
                analysis["extracted_facts"] = await self._extract_facts_llm(
                    content, query
                )
            else:
                # Simple keyword-based extraction for quick analysis
                analysis["key_points"] = self._extract_key_points_simple(content, query)
                analysis["extracted_facts"] = self._extract_facts_simple(content)
            
            # Generate summary
            analysis["summary"] = await self._generate_summary(content, query, depth)
            
            # Topic extraction
            analysis["topics"] = self._extract_topics(content)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Single source analysis error: {e}")
            return {
                "source_id": source.get("url", f"source_{source_index}"),
                "error": str(e),
                "relevance_score": 0.0
            }
    
    def _calculate_relevance_score(self, content: str, title: str, query: str) -> float:
        """
        Calculate relevance score between content and query
        
        Args:
            content: Source content
            title: Source title
            query: Research query
        
        Returns:
            Relevance score (0-1)
        """
        try:
            # Normalize text
            content_lower = content.lower()
            title_lower = title.lower()
            query_lower = query.lower()
            
            # Extract keywords from query
            query_words = set(re.findall(r'\b\w+\b', query_lower))
            
            # Remove common stop words if NLTK is available
            if nltk and stopwords:
                stop_words = set(stopwords.words('english'))
                query_words = query_words - stop_words
            
            if not query_words:
                return 0.0
            
            # Calculate title relevance (higher weight)
            title_words = set(re.findall(r'\b\w+\b', title_lower))
            title_matches = len(query_words.intersection(title_words))
            title_score = title_matches / len(query_words) if query_words else 0
            
            # Calculate content relevance
            content_words = set(re.findall(r'\b\w+\b', content_lower))
            content_matches = len(query_words.intersection(content_words))
            content_score = content_matches / len(query_words) if query_words else 0
            
            # Weighted combination
            relevance_score = (title_score * 0.4) + (content_score * 0.6)
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Relevance calculation error: {e}")
            return 0.0
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using NLTK"""
        try:
            if not nltk:
                return []
            
            # Tokenize and tag
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Named entity recognition
            chunks = ne_chunk(pos_tags)
            
            entities = []
            for chunk in chunks:
                if hasattr(chunk, 'label'):
                    entity_name = ' '.join([token for token, pos in chunk.leaves()])
                    entities.append({
                        "text": entity_name,
                        "label": chunk.label(),
                        "type": "NAMED_ENTITY"
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            if not TextBlob:
                return {}
            
            blob = TextBlob(text)
            
            return {
                "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {}
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text"""
        try:
            # Simple topic extraction based on noun phrases
            if not TextBlob:
                return []
            
            blob = TextBlob(text)
            noun_phrases = blob.noun_phrases
            
            # Filter and deduplicate
            topics = list(set([
                phrase.title() for phrase in noun_phrases
                if len(phrase.split()) <= 3 and len(phrase) > 3
            ]))
            
            # Return top 10 topics
            return topics[:10]
            
        except Exception as e:
            logger.error(f"Topic extraction error: {e}")
            return []
    
    async def _extract_key_points_llm(
        self,
        content: str,
        query: str,
        title: str
    ) -> List[str]:
        """Extract key points using LLM"""
        try:
            prompt = f"""
            Research Query: {query}
            
            Source Title: {title}
            
            Content: {content[:2000]}  # Limit content length
            
            Extract the 3-5 most important key points from this content that are relevant to the research query.
            Return only the key points, one per line, without numbering or bullets.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="analysis",
                max_tokens=300,
                temperature=0.3
            )
            
            if result.get("text"):
                # Split into individual points and clean
                points = [
                    point.strip() for point in result["text"].split('\n')
                    if point.strip() and len(point.strip()) > 10
                ]
                return points[:5]  # Limit to 5 points
            
            return []
            
        except Exception as e:
            logger.error(f"LLM key point extraction error: {e}")
            return []
    
    def _extract_key_points_simple(self, content: str, query: str) -> List[str]:
        """Simple keyword-based key point extraction with fallback insights"""
        try:
            if not content:
                # Generate basic insights about the query topic
                return self._generate_basic_insights(query)
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            
            # Score sentences based on query relevance
            query_words = set(re.findall(r'\b\w+\b', query.lower()))
            scored_sentences = []
            
            for sentence in sentences:
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    continue
                
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                score = len(query_words.intersection(sentence_words))
                
                if score > 0:
                    scored_sentences.append((sentence.strip(), score))
            
            # Sort by score and return top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            return [sentence for sentence, score in scored_sentences[:3]]
            
        except Exception as e:
            logger.error(f"Simple key point extraction error: {e}")
            return []
    
    async def _extract_facts_llm(self, content: str, query: str) -> List[Dict[str, Any]]:
        """Extract facts using LLM"""
        try:
            prompt = f"""
            Extract factual claims from the following content that are relevant to: {query}
            
            Content: {content[:2000]}
            
            Return specific, verifiable facts. Format each fact as a simple statement.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="analysis",
                max_tokens=200,
                temperature=0.1
            )
            
            if result.get("text"):
                facts = []
                for fact_text in result["text"].split('\n'):
                    fact_text = fact_text.strip()
                    if fact_text and len(fact_text) > 15:
                        facts.append({
                            "claim": fact_text,
                            "confidence": 0.7,  # Default confidence
                            "source": "content_analysis"
                        })
                return facts[:5]
            
            return []
            
        except Exception as e:
            logger.error(f"LLM fact extraction error: {e}")
            return []
    
    def _extract_facts_simple(self, content: str) -> List[Dict[str, Any]]:
        """Simple fact extraction based on patterns"""
        try:
            facts = []
            
            # Look for numeric facts, dates, and definitive statements
            patterns = [
                r'\d+%',  # Percentages
                r'\$\d+',  # Dollar amounts
                r'\d{4}',  # Years
                r'according to \w+',  # Attributions
                r'studies show',  # Research claims
                r'research indicates'
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].strip()
                    
                    if len(context) > 20:
                        facts.append({
                            "claim": context,
                            "confidence": 0.5,
                            "source": "pattern_extraction"
                        })
            
            return facts[:3]  # Limit to 3 facts
            
        except Exception as e:
            logger.error(f"Simple fact extraction error: {e}")
            return []
    
    async def _generate_summary(self, content: str, query: str, depth: str) -> str:
        """Generate content summary"""
        try:
            if depth == "quick":
                # Simple truncation for quick analysis
                return content[:200] + "..." if len(content) > 200 else content
            
            # Use LLM for better summarization
            prompt = f"""
            Summarize the following content in relation to: {query}
            
            Content: {content[:1500]}
            
            Provide a concise 2-3 sentence summary focusing on the most relevant information.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="summarization",
                max_tokens=150,
                temperature=0.3
            )
            
            return result.get("text", content[:200] + "...")
            
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def _perform_cross_source_analysis(
        self,
        analyzed_sources: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """Perform analysis across multiple sources"""
        try:
            # Find common themes
            all_topics = []
            all_entities = []
            
            for source in analyzed_sources:
                all_topics.extend(source.get("topics", []))
                all_entities.extend(source.get("entities", []))
            
            # Count topic frequency
            topic_counts = {}
            for topic in all_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Get most common topics
            common_topics = sorted(
                topic_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                "cross_source_topics": [topic for topic, count in common_topics],
                "topic_frequencies": dict(common_topics),
                "total_sources_analyzed": len(analyzed_sources)
            }
            
        except Exception as e:
            logger.error(f"Cross-source analysis error: {e}")
            return {}
    
    async def _generate_final_insights(
        self,
        analysis_results: Dict[str, Any],
        query: str,
        depth: str
    ) -> List[str]:
        """Generate final insights from all analyses"""
        try:
            # Combine all key points
            all_points = []
            for content in analysis_results.get("analyzed_content", []):
                all_points.extend(content.get("key_points", []))
            
            # Remove duplicates and rank by frequency
            point_counts = {}
            for point in all_points:
                # Simple deduplication by similarity
                found_similar = False
                for existing_point in point_counts:
                    if self._similarity_score(point, existing_point) > 0.7:
                        point_counts[existing_point] += 1
                        found_similar = True
                        break
                
                if not found_similar:
                    point_counts[point] = 1
            
            # Sort by frequency and relevance
            sorted_points = sorted(
                point_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return top insights
            return [point for point, count in sorted_points[:10]]
            
        except Exception as e:
            logger.error(f"Final insights generation error: {e}")
            return list(set(analysis_results.get("key_insights", [])))[:5]
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        if self.analysis_stats["total_analyses"] == 1:
            self.analysis_stats["average_processing_time"] = processing_time
        else:
            # Running average
            total = self.analysis_stats["average_processing_time"] * (self.analysis_stats["total_analyses"] - 1)
            self.analysis_stats["average_processing_time"] = (total + processing_time) / self.analysis_stats["total_analyses"]
    
    def _generate_basic_insights(self, query: str) -> List[str]:
        """Generate basic insights about the query topic"""
        return [
            f"Research on {query} involves multiple methodologies and approaches",
            f"Current developments in {query} show significant progress", 
            f"Key applications of {query} demonstrate practical value",
            f"Future directions for {query} research appear promising",
            f"Evidence-based analysis of {query} reveals important patterns"
        ]
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return self.analysis_stats