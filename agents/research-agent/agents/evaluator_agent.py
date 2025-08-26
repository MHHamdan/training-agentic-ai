"""
Evaluator Agent - Quality assessment and validation
Comprehensive research quality evaluation
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.model_manager import ModelManager
from graph.state import ResearchState

logger = logging.getLogger(__name__)

class EvaluatorAgent:
    """
    Quality assessment and validation agent
    Evaluates research completeness, accuracy, and bias
    """
    
    def __init__(self):
        """Initialize evaluator agent"""
        self.config = config
        self.model_manager = ModelManager()
        self.evaluation_stats = {
            "total_evaluations": 0,
            "average_quality_score": 0.0,
            "evaluations_by_depth": {}
        }
    
    @observe(as_type="evaluation")
    async def evaluate(
        self,
        state: ResearchState,
        evaluation_depth: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive research evaluation
        
        Args:
            state: Complete research state
            evaluation_depth: Depth of evaluation
        
        Returns:
            Evaluation results with scores and recommendations
        """
        try:
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Evaluating research quality",
                    metadata={
                        "query": state.get("query", ""),
                        "evaluation_depth": evaluation_depth,
                        "agent": "evaluator_agent"
                    }
                )
            
            start_time = datetime.now()
            
            # Perform individual evaluations
            accuracy_score = await self._evaluate_accuracy(state)
            completeness_score = await self._evaluate_completeness(state)
            relevance_score = await self._evaluate_relevance(state)
            bias_score = await self._evaluate_bias(state)
            source_reliability = await self._evaluate_source_reliability(state)
            citation_quality = await self._evaluate_citation_quality(state)
            academic_compliance = await self._evaluate_academic_compliance(state)
            fact_check_score = await self._evaluate_fact_checking(state)
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality({
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "relevance": relevance_score,
                "bias": bias_score,
                "source_reliability": source_reliability,
                "citation_quality": citation_quality,
                "academic_compliance": academic_compliance,
                "fact_check": fact_check_score
            })
            
            # Generate evaluation insights
            insights = await self._generate_evaluation_insights(state, {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "relevance": relevance_score,
                "bias": bias_score,
                "source_reliability": source_reliability
            })
            
            # Identify issues and recommendations
            issues = self._identify_issues(state, {
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "bias": bias_score,
                "source_reliability": source_reliability
            })
            
            recommendations = self._generate_recommendations(issues, overall_quality)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare evaluation results
            evaluation_results = {
                "overall_quality": overall_quality,
                "accuracy": accuracy_score,
                "completeness": completeness_score,
                "relevance": relevance_score,
                "bias_detection": bias_score,
                "source_reliability": source_reliability,
                "citation_quality": citation_quality,
                "academic_compliance": academic_compliance,
                "fact_check_score": fact_check_score,
                "insights": insights,
                "issues_found": issues,
                "recommendations": recommendations,
                "confidence": self._calculate_confidence(state),
                "evaluation_metadata": {
                    "evaluation_depth": evaluation_depth,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat(),
                    "evaluator_version": "2.0.0"
                }
            }
            
            # Update statistics
            self._update_evaluation_stats(evaluation_results)
            
            # Track output with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Evaluation complete. Overall quality: {overall_quality:.2%}",
                    metadata={
                        "overall_quality": overall_quality,
                        "processing_time": processing_time,
                        "issues_count": len(issues)
                    }
                )
            
            logger.info(f"Evaluation completed: Overall quality {overall_quality:.2%}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Evaluation failed: {str(e)}",
                    metadata={"error": str(e)}
                )
            
            return {
                "overall_quality": 0.0,
                "error": str(e),
                "evaluation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "evaluation_failed": True
                }
            }
    
    async def _evaluate_accuracy(self, state: ResearchState) -> float:
        """Evaluate research accuracy"""
        try:
            # Check fact verification results
            fact_check_results = state.get("fact_check_results", [])
            verified_claims = state.get("verified_claims", [])
            disputed_claims = state.get("disputed_claims", [])
            
            if not fact_check_results:
                return 0.5  # Neutral score if no fact checking
            
            total_claims = len(verified_claims) + len(disputed_claims)
            if total_claims == 0:
                return 0.5
            
            # Calculate accuracy based on verification ratio
            accuracy = len(verified_claims) / total_claims
            
            # Bonus for high-quality sources
            source_quality_scores = state.get("source_quality_scores", {})
            if source_quality_scores:
                avg_source_quality = sum(source_quality_scores.values()) / len(source_quality_scores)
                accuracy = (accuracy * 0.7) + (avg_source_quality * 0.3)
            
            return min(accuracy, 1.0)
            
        except Exception as e:
            logger.error(f"Accuracy evaluation error: {e}")
            return 0.0
    
    async def _evaluate_completeness(self, state: ResearchState) -> float:
        """Evaluate research completeness"""
        try:
            query = state.get("query", "")
            synthesis = state.get("synthesis", "")
            key_insights = state.get("key_insights", [])
            search_results = state.get("search_results", [])
            
            completeness_factors = []
            
            # Check if synthesis addresses the query
            if synthesis and query:
                query_words = set(query.lower().split())
                synthesis_words = set(synthesis.lower().split())
                coverage = len(query_words.intersection(synthesis_words)) / len(query_words)
                completeness_factors.append(coverage)
            
            # Check insight density
            if key_insights:
                insight_density = min(len(key_insights) / 10, 1.0)  # Max 10 insights
                completeness_factors.append(insight_density)
            else:
                completeness_factors.append(0.0)
            
            # Check source diversity
            if search_results:
                sources = set(result.get("source", "") for result in search_results)
                source_diversity = min(len(sources) / 3, 1.0)  # Expect at least 3 different sources
                completeness_factors.append(source_diversity)
            else:
                completeness_factors.append(0.0)
            
            # Check synthesis length appropriateness
            if synthesis:
                synthesis_length_score = min(len(synthesis) / 1000, 1.0)  # Expect at least 1000 chars
                completeness_factors.append(synthesis_length_score)
            else:
                completeness_factors.append(0.0)
            
            return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0
            
        except Exception as e:
            logger.error(f"Completeness evaluation error: {e}")
            return 0.0
    
    async def _evaluate_relevance(self, state: ResearchState) -> float:
        """Evaluate research relevance to query"""
        try:
            query = state.get("query", "")
            relevance_scores = state.get("relevance_scores", {})
            
            if not relevance_scores:
                return 0.5  # Neutral if no relevance scores
            
            # Calculate average relevance
            avg_relevance = sum(relevance_scores.values()) / len(relevance_scores)
            
            # Check if synthesis is relevant to query using LLM
            synthesis = state.get("synthesis", "")
            if synthesis and query:
                relevance_check = await self._llm_relevance_check(query, synthesis)
                avg_relevance = (avg_relevance * 0.6) + (relevance_check * 0.4)
            
            return min(avg_relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Relevance evaluation error: {e}")
            return 0.0
    
    async def _evaluate_bias(self, state: ResearchState) -> float:
        """Evaluate potential bias in research"""
        try:
            synthesis = state.get("synthesis", "")
            search_results = state.get("search_results", [])
            
            bias_indicators = []
            
            # Check source diversity
            sources = [result.get("source", "") for result in search_results]
            unique_sources = len(set(sources))
            source_diversity_score = min(unique_sources / 5, 1.0)  # Expect 5 different sources
            bias_indicators.append(source_diversity_score)
            
            # Use LLM to detect bias in synthesis
            if synthesis:
                bias_check = await self._llm_bias_check(synthesis)
                bias_indicators.append(1.0 - bias_check)  # Lower bias = higher score
            
            # Check for balanced language
            if synthesis:
                balance_score = self._check_balanced_language(synthesis)
                bias_indicators.append(balance_score)
            
            return sum(bias_indicators) / len(bias_indicators) if bias_indicators else 0.5
            
        except Exception as e:
            logger.error(f"Bias evaluation error: {e}")
            return 0.5
    
    async def _evaluate_source_reliability(self, state: ResearchState) -> float:
        """Evaluate reliability of sources used"""
        try:
            search_results = state.get("search_results", [])
            
            if not search_results:
                return 0.0
            
            reliability_scores = []
            
            for result in search_results:
                source = result.get("source", "").lower()
                quality_score = result.get("quality_score", 0.5)
                
                # Boost scores for known reliable sources
                if any(reliable in source for reliable in ["arxiv", "wikipedia", "gov", "edu", "nature", "science"]):
                    quality_score = min(quality_score + 0.2, 1.0)
                
                # Penalize low-quality sources
                if any(unreliable in source for unreliable in ["blog", "forum", "social"]):
                    quality_score = max(quality_score - 0.2, 0.0)
                
                reliability_scores.append(quality_score)
            
            return sum(reliability_scores) / len(reliability_scores)
            
        except Exception as e:
            logger.error(f"Source reliability evaluation error: {e}")
            return 0.0
    
    async def _evaluate_citation_quality(self, state: ResearchState) -> float:
        """Evaluate citation quality"""
        try:
            citations = state.get("citations", [])
            bibliography = state.get("bibliography", [])
            search_results = state.get("search_results", [])
            
            if not citations:
                return 0.0
            
            # Check citation completeness
            expected_citations = len(search_results)
            actual_citations = len(citations)
            citation_completeness = min(actual_citations / expected_citations, 1.0) if expected_citations > 0 else 0.0
            
            # Check citation format consistency
            citation_format_score = 1.0  # Assume good format for now
            
            # Check bibliography quality
            bibliography_score = 0.5
            if bibliography:
                # Check for required fields
                required_fields = ["title", "source", "url"]
                complete_entries = sum(1 for entry in bibliography if all(field in entry for field in required_fields))
                bibliography_score = complete_entries / len(bibliography) if bibliography else 0.0
            
            return (citation_completeness * 0.4 + citation_format_score * 0.3 + bibliography_score * 0.3)
            
        except Exception as e:
            logger.error(f"Citation quality evaluation error: {e}")
            return 0.0
    
    async def _evaluate_academic_compliance(self, state: ResearchState) -> float:
        """Evaluate academic standards compliance"""
        try:
            synthesis = state.get("synthesis", "")
            citations = state.get("citations", [])
            
            compliance_factors = []
            
            # Check for proper citations
            if citations:
                compliance_factors.append(1.0)
            else:
                compliance_factors.append(0.0)
            
            # Check synthesis structure and tone
            if synthesis:
                structure_score = self._check_academic_structure(synthesis)
                compliance_factors.append(structure_score)
            
            # Check for plagiarism indicators (basic check)
            plagiarism_score = 1.0  # Assume no plagiarism for now
            compliance_factors.append(plagiarism_score)
            
            return sum(compliance_factors) / len(compliance_factors) if compliance_factors else 0.0
            
        except Exception as e:
            logger.error(f"Academic compliance evaluation error: {e}")
            return 0.0
    
    async def _evaluate_fact_checking(self, state: ResearchState) -> float:
        """Evaluate fact checking quality"""
        try:
            fact_check_results = state.get("fact_check_results", [])
            verified_claims = state.get("verified_claims", [])
            disputed_claims = state.get("disputed_claims", [])
            
            if not fact_check_results:
                return 0.0
            
            total_checks = len(fact_check_results)
            successful_verifications = len(verified_claims)
            
            if total_checks == 0:
                return 0.0
            
            # Calculate fact checking effectiveness
            verification_rate = successful_verifications / total_checks
            
            # Bonus for thorough fact checking
            thorough_bonus = min(total_checks / 10, 0.2)  # Up to 20% bonus for 10+ fact checks
            
            return min(verification_rate + thorough_bonus, 1.0)
            
        except Exception as e:
            logger.error(f"Fact checking evaluation error: {e}")
            return 0.0
    
    def _calculate_overall_quality(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "accuracy": 0.25,
            "completeness": 0.20,
            "relevance": 0.20,
            "bias": 0.15,
            "source_reliability": 0.10,
            "citation_quality": 0.05,
            "academic_compliance": 0.03,
            "fact_check": 0.02
        }
        
        weighted_score = sum(scores.get(metric, 0) * weight for metric, weight in weights.items())
        return min(weighted_score, 1.0)
    
    async def _llm_relevance_check(self, query: str, synthesis: str) -> float:
        """Use LLM to check relevance"""
        try:
            prompt = f"""
            Rate how well this synthesis addresses the research query on a scale of 0.0 to 1.0:
            
            Query: {query}
            
            Synthesis: {synthesis[:1000]}
            
            Consider:
            - Does it directly answer the query?
            - Are the key aspects of the query covered?
            - Is the information relevant and on-topic?
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="analysis",
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract score from response
            score_text = result.get("text", "0.5").strip()
            try:
                score = float(score_text)
                return max(0.0, min(score, 1.0))
            except ValueError:
                return 0.5
                
        except Exception as e:
            logger.error(f"LLM relevance check error: {e}")
            return 0.5
    
    async def _llm_bias_check(self, synthesis: str) -> float:
        """Use LLM to detect bias"""
        try:
            prompt = f"""
            Analyze this text for bias and rate the bias level from 0.0 (no bias) to 1.0 (heavily biased):
            
            Text: {synthesis[:1000]}
            
            Look for:
            - One-sided presentation
            - Loaded language
            - Missing perspectives
            - Unsubstantiated claims
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="analysis",
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract score from response
            score_text = result.get("text", "0.3").strip()
            try:
                score = float(score_text)
                return max(0.0, min(score, 1.0))
            except ValueError:
                return 0.3
                
        except Exception as e:
            logger.error(f"LLM bias check error: {e}")
            return 0.3
    
    def _check_balanced_language(self, text: str) -> float:
        """Check for balanced language indicators"""
        # Simple check for balanced language markers
        balance_indicators = [
            "however", "although", "while", "despite", "on the other hand",
            "nevertheless", "conversely", "alternatively", "in contrast"
        ]
        
        text_lower = text.lower()
        balance_count = sum(1 for indicator in balance_indicators if indicator in text_lower)
        
        # Normalize score
        return min(balance_count / 3, 1.0)  # Expect at least 3 balance indicators
    
    def _check_academic_structure(self, synthesis: str) -> float:
        """Check for academic structure"""
        structure_indicators = [
            "introduction", "conclusion", "findings", "results",
            "analysis", "discussion", "methodology", "evidence"
        ]
        
        text_lower = synthesis.lower()
        structure_count = sum(1 for indicator in structure_indicators if indicator in text_lower)
        
        # Normalize score
        return min(structure_count / 4, 1.0)  # Expect at least 4 structure indicators
    
    def _calculate_confidence(self, state: ResearchState) -> float:
        """Calculate confidence in the research results"""
        factors = []
        
        # Source count confidence
        source_count = len(state.get("search_results", []))
        source_confidence = min(source_count / 10, 1.0)
        factors.append(source_confidence)
        
        # Processing completeness confidence
        required_fields = ["synthesis", "key_insights", "citations"]
        completed_fields = sum(1 for field in required_fields if state.get(field))
        completion_confidence = completed_fields / len(required_fields)
        factors.append(completion_confidence)
        
        # Error rate confidence
        errors = state.get("errors", [])
        error_confidence = max(0, 1.0 - (len(errors) / 5))  # Penalty for errors
        factors.append(error_confidence)
        
        return sum(factors) / len(factors) if factors else 0.0
    
    async def _generate_evaluation_insights(
        self,
        state: ResearchState,
        scores: Dict[str, float]
    ) -> List[str]:
        """Generate insights from evaluation"""
        insights = []
        
        # Quality insights
        if scores["accuracy"] > 0.8:
            insights.append("High accuracy research with well-verified claims")
        elif scores["accuracy"] < 0.5:
            insights.append("Research accuracy could be improved with better fact-checking")
        
        if scores["completeness"] > 0.8:
            insights.append("Comprehensive research covering multiple aspects")
        elif scores["completeness"] < 0.5:
            insights.append("Research appears incomplete - consider additional sources")
        
        if scores["bias"] > 0.8:
            insights.append("Well-balanced research with multiple perspectives")
        elif scores["bias"] < 0.5:
            insights.append("Potential bias detected - consider alternative viewpoints")
        
        # Source insights
        source_count = len(state.get("search_results", []))
        if source_count < 3:
            insights.append("Limited number of sources - consider expanding research")
        elif source_count > 15:
            insights.append("Excellent source diversity enhances research credibility")
        
        return insights
    
    def _identify_issues(self, state: ResearchState, scores: Dict[str, float]) -> List[str]:
        """Identify specific issues with the research"""
        issues = []
        
        # Quality issues
        if scores["accuracy"] < 0.5:
            issues.append("Low accuracy score indicates potential factual errors")
        
        if scores["completeness"] < 0.5:
            issues.append("Incomplete research coverage of the topic")
        
        if scores["bias"] < 0.5:
            issues.append("Potential bias detected in source selection or analysis")
        
        if scores["source_reliability"] < 0.5:
            issues.append("Low reliability of sources used")
        
        # Missing components
        if not state.get("citations"):
            issues.append("No citations provided - academic integrity concern")
        
        if not state.get("fact_check_results"):
            issues.append("No fact-checking performed")
        
        if len(state.get("key_insights", [])) < 3:
            issues.append("Insufficient key insights extracted")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str], overall_quality: float) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if overall_quality < 0.5:
            recommendations.append("Consider conducting additional research with more diverse sources")
        
        if "Low accuracy" in str(issues):
            recommendations.append("Implement more rigorous fact-checking procedures")
        
        if "bias detected" in str(issues):
            recommendations.append("Seek sources with different perspectives and viewpoints")
        
        if "No citations" in str(issues):
            recommendations.append("Add proper citations for all sources used")
        
        if "Incomplete research" in str(issues):
            recommendations.append("Expand research scope to cover more aspects of the topic")
        
        if not recommendations:
            recommendations.append("Research meets quality standards")
        
        return recommendations
    
    def _update_evaluation_stats(self, results: Dict[str, Any]):
        """Update evaluation statistics"""
        self.evaluation_stats["total_evaluations"] += 1
        
        # Update average quality score
        overall_quality = results.get("overall_quality", 0.0)
        if self.evaluation_stats["total_evaluations"] == 1:
            self.evaluation_stats["average_quality_score"] = overall_quality
        else:
            # Running average
            total = self.evaluation_stats["average_quality_score"] * (self.evaluation_stats["total_evaluations"] - 1)
            self.evaluation_stats["average_quality_score"] = (total + overall_quality) / self.evaluation_stats["total_evaluations"]
        
        # Update depth statistics
        depth = results.get("evaluation_metadata", {}).get("evaluation_depth", "unknown")
        if depth not in self.evaluation_stats["evaluations_by_depth"]:
            self.evaluation_stats["evaluations_by_depth"][depth] = 0
        self.evaluation_stats["evaluations_by_depth"][depth] += 1
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return self.evaluation_stats