"""
Fact Checker Tool - Verify claims against reliable sources
Advanced fact verification with confidence scoring
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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class FactChecker:
    """
    Advanced fact checking tool with multiple verification methods
    """
    
    def __init__(self):
        """Initialize fact checker"""
        self.config = config
        self.model_manager = ModelManager()
        self.fact_check_stats = {
            "total_checks": 0,
            "verified_facts": 0,
            "disputed_facts": 0,
            "inconclusive_checks": 0
        }
    
    @observe(as_type="generation")
    async def verify_facts(
        self,
        claims: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Verify factual claims against sources
        
        Args:
            claims: List of claims to verify
            sources: List of sources to check against
            threshold: Confidence threshold for verification
        
        Returns:
            Fact checking results with confidence scores
        """
        try:
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Fact checking {len(claims)} claims against {len(sources)} sources",
                    metadata={
                        "claims_count": len(claims),
                        "sources_count": len(sources),
                        "threshold": threshold,
                        "tool": "fact_checker"
                    }
                )
            
            start_time = datetime.now()
            
            # Process claims in parallel
            verification_tasks = [
                self._verify_single_claim(claim, sources, threshold)
                for claim in claims
            ]
            
            verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
            
            # Process results
            verified_claims = []
            disputed_claims = []
            inconclusive_claims = []
            detailed_results = []
            
            for i, result in enumerate(verification_results):
                if isinstance(result, Exception):
                    logger.error(f"Fact check error for claim {i}: {result}")
                    inconclusive_claims.append(claims[i])
                    detailed_results.append({
                        "claim": claims[i],
                        "status": "error",
                        "confidence": 0.0,
                        "error": str(result)
                    })
                else:
                    detailed_results.append(result)
                    
                    if result["status"] == "verified":
                        verified_claims.append(result["claim"]["claim"])
                    elif result["status"] == "disputed":
                        disputed_claims.append(result["claim"]["claim"])
                    else:
                        inconclusive_claims.append(result["claim"])
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._update_fact_check_stats(len(verified_claims), len(disputed_claims), len(inconclusive_claims))
            
            # Prepare results
            results = {
                "results": detailed_results,
                "verified": verified_claims,
                "disputed": disputed_claims,
                "inconclusive": [claim.get("claim", str(claim)) for claim in inconclusive_claims],
                "summary": {
                    "total_claims": len(claims),
                    "verified_count": len(verified_claims),
                    "disputed_count": len(disputed_claims),
                    "inconclusive_count": len(inconclusive_claims),
                    "verification_rate": len(verified_claims) / len(claims) if claims else 0,
                    "processing_time": processing_time
                },
                "metadata": {
                    "threshold": threshold,
                    "timestamp": datetime.now().isoformat(),
                    "fact_checker_version": "2.0.0"
                }
            }
            
            # Track output with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Fact checking complete: {len(verified_claims)} verified, {len(disputed_claims)} disputed",
                    metadata={
                        "verified_count": len(verified_claims),
                        "disputed_count": len(disputed_claims),
                        "processing_time": processing_time
                    }
                )
            
            logger.info(f"Fact checking completed: {len(verified_claims)} verified, {len(disputed_claims)} disputed")
            
            return results
            
        except Exception as e:
            logger.error(f"Fact checking error: {e}")
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Fact checking failed: {str(e)}",
                    metadata={"error": str(e)}
                )
            
            return {
                "results": [],
                "verified": [],
                "disputed": [],
                "inconclusive": [],
                "error": str(e)
            }
    
    @observe(as_type="generation")
    async def _verify_single_claim(
        self,
        claim: Dict[str, Any],
        sources: List[Dict[str, Any]],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Verify a single claim against sources
        
        Args:
            claim: Claim to verify
            sources: Sources to check against
            threshold: Confidence threshold
        
        Returns:
            Verification result for the claim
        """
        try:
            claim_text = claim.get("claim", str(claim))
            
            # Find relevant sources for this claim
            relevant_sources = self._find_relevant_sources(claim_text, sources)
            
            if not relevant_sources:
                return {
                    "claim": claim,
                    "status": "inconclusive",
                    "confidence": 0.0,
                    "reason": "No relevant sources found",
                    "supporting_sources": [],
                    "contradicting_sources": []
                }
            
            # Perform different types of verification
            exact_match_score = await self._check_exact_match(claim_text, relevant_sources)
            semantic_match_score = await self._check_semantic_match(claim_text, relevant_sources)
            llm_verification_score = await self._llm_fact_verification(claim_text, relevant_sources)
            
            # Calculate overall confidence
            confidence_scores = [exact_match_score, semantic_match_score, llm_verification_score]
            overall_confidence = sum(score for score in confidence_scores if score is not None) / len([s for s in confidence_scores if s is not None])
            
            # Determine status based on confidence
            if overall_confidence >= threshold:
                status = "verified"
            elif overall_confidence <= (1 - threshold):
                status = "disputed"
            else:
                status = "inconclusive"
            
            # Find supporting and contradicting sources
            supporting_sources = self._find_supporting_sources(claim_text, relevant_sources, 0.7)
            contradicting_sources = self._find_contradicting_sources(claim_text, relevant_sources, 0.7)
            
            return {
                "claim": claim,
                "status": status,
                "confidence": overall_confidence,
                "verification_details": {
                    "exact_match_score": exact_match_score,
                    "semantic_match_score": semantic_match_score,
                    "llm_verification_score": llm_verification_score
                },
                "supporting_sources": supporting_sources,
                "contradicting_sources": contradicting_sources,
                "relevant_sources_count": len(relevant_sources)
            }
            
        except Exception as e:
            logger.error(f"Single claim verification error: {e}")
            return {
                "claim": claim,
                "status": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _find_relevant_sources(self, claim: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find sources relevant to the claim"""
        relevant_sources = []
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        
        for source in sources:
            content = (source.get("content", "") + " " + source.get("snippet", "")).lower()
            content_words = set(re.findall(r'\b\w+\b', content))
            
            # Calculate overlap
            overlap = len(claim_words.intersection(content_words))
            relevance_score = overlap / len(claim_words) if claim_words else 0
            
            if relevance_score > 0.2:  # At least 20% word overlap
                source_copy = source.copy()
                source_copy["relevance_score"] = relevance_score
                relevant_sources.append(source_copy)
        
        # Sort by relevance and return top 5
        relevant_sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_sources[:5]
    
    async def _check_exact_match(self, claim: str, sources: List[Dict[str, Any]]) -> Optional[float]:
        """Check for exact or near-exact matches"""
        try:
            claim_clean = re.sub(r'[^\w\s]', '', claim.lower()).strip()
            
            for source in sources:
                content = source.get("content", "") + " " + source.get("snippet", "")
                content_clean = re.sub(r'[^\w\s]', '', content.lower())
                
                # Check for exact substring match
                if claim_clean in content_clean:
                    return 1.0
                
                # Check for high similarity (simple approach)
                claim_words = claim_clean.split()
                content_words = content_clean.split()
                
                # Check if most claim words appear in order
                matches = 0
                content_idx = 0
                for word in claim_words:
                    while content_idx < len(content_words):
                        if content_words[content_idx] == word:
                            matches += 1
                            content_idx += 1
                            break
                        content_idx += 1
                
                if matches / len(claim_words) > 0.8:  # 80% of words found in order
                    return 0.9
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Exact match check error: {e}")
            return None
    
    async def _check_semantic_match(self, claim: str, sources: List[Dict[str, Any]]) -> Optional[float]:
        """Check for semantic similarity"""
        try:
            # Simple semantic matching based on key concepts
            # This could be enhanced with embedding models
            
            claim_concepts = self._extract_key_concepts(claim)
            
            best_match = 0.0
            for source in sources:
                content = source.get("content", "") + " " + source.get("snippet", "")
                content_concepts = self._extract_key_concepts(content)
                
                # Calculate concept overlap
                overlap = len(claim_concepts.intersection(content_concepts))
                similarity = overlap / len(claim_concepts) if claim_concepts else 0
                
                best_match = max(best_match, similarity)
            
            return best_match
            
        except Exception as e:
            logger.error(f"Semantic match check error: {e}")
            return None
    
    def _extract_key_concepts(self, text: str) -> set:
        """Extract key concepts from text"""
        # Simple concept extraction - could be enhanced with NLP
        words = re.findall(r'\b\w{4,}\b', text.lower())  # Words with 4+ characters
        
        # Filter out common words
        stop_words = {
            "that", "this", "with", "from", "they", "been", "have", "were", "said",
            "each", "which", "their", "time", "will", "about", "would", "there",
            "could", "other", "after", "first", "well", "many", "some", "what"
        }
        
        concepts = set(word for word in words if word not in stop_words)
        return concepts
    
    async def _llm_fact_verification(self, claim: str, sources: List[Dict[str, Any]]) -> Optional[float]:
        """Use LLM for fact verification"""
        try:
            # Prepare context from sources
            context_parts = []
            for i, source in enumerate(sources[:3], 1):  # Limit to 3 sources
                content = source.get("content", "") or source.get("snippet", "")
                context_parts.append(f"Source {i}: {content[:500]}")  # Limit content length
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""
            Verify the following claim against the provided sources. Rate your confidence that the claim is true on a scale of 0.0 to 1.0.
            
            Claim: {claim}
            
            Sources:
            {context}
            
            Instructions:
            - 1.0 = Completely supported by sources
            - 0.8-0.9 = Strongly supported with minor details
            - 0.6-0.7 = Generally supported but some uncertainty
            - 0.4-0.5 = Mixed evidence or unclear
            - 0.2-0.3 = Weakly contradicted
            - 0.0-0.1 = Strongly contradicted or false
            
            Respond with only a number between 0.0 and 1.0.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="fact_checking",
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract confidence score
            response_text = result.get("text", "0.5").strip()
            try:
                confidence = float(response_text)
                return max(0.0, min(confidence, 1.0))
            except ValueError:
                logger.warning(f"Invalid LLM response for fact checking: {response_text}")
                return 0.5
                
        except Exception as e:
            logger.error(f"LLM fact verification error: {e}")
            return None
    
    def _find_supporting_sources(
        self,
        claim: str,
        sources: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, str]]:
        """Find sources that support the claim"""
        supporting = []
        
        for source in sources:
            relevance = source.get("relevance_score", 0)
            if relevance >= threshold:
                supporting.append({
                    "title": source.get("title", "Unknown Title"),
                    "url": source.get("url", ""),
                    "source": source.get("source", "Unknown Source"),
                    "relevance_score": relevance
                })
        
        return supporting[:3]  # Return top 3 supporting sources
    
    def _find_contradicting_sources(
        self,
        claim: str,
        sources: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, str]]:
        """Find sources that contradict the claim"""
        # Simple implementation - look for negation words near claim concepts
        contradicting = []
        
        claim_concepts = self._extract_key_concepts(claim)
        negation_words = {"not", "no", "false", "incorrect", "wrong", "contradicts", "disputes"}
        
        for source in sources:
            content = (source.get("content", "") + " " + source.get("snippet", "")).lower()
            
            # Check for negation near claim concepts
            contradiction_score = 0
            for concept in claim_concepts:
                if concept in content:
                    # Look for negation words nearby
                    concept_pos = content.find(concept)
                    nearby_text = content[max(0, concept_pos-100):concept_pos+100]
                    
                    if any(neg_word in nearby_text for neg_word in negation_words):
                        contradiction_score += 1
            
            if contradiction_score > 0:
                contradicting.append({
                    "title": source.get("title", "Unknown Title"),
                    "url": source.get("url", ""),
                    "source": source.get("source", "Unknown Source"),
                    "contradiction_score": contradiction_score
                })
        
        return contradicting[:2]  # Return top 2 contradicting sources
    
    def _update_fact_check_stats(self, verified: int, disputed: int, inconclusive: int):
        """Update fact checking statistics"""
        self.fact_check_stats["total_checks"] += verified + disputed + inconclusive
        self.fact_check_stats["verified_facts"] += verified
        self.fact_check_stats["disputed_facts"] += disputed
        self.fact_check_stats["inconclusive_checks"] += inconclusive
    
    def get_fact_check_statistics(self) -> Dict[str, Any]:
        """Get fact checking statistics"""
        stats = self.fact_check_stats.copy()
        
        if stats["total_checks"] > 0:
            stats["verification_rate"] = stats["verified_facts"] / stats["total_checks"]
            stats["dispute_rate"] = stats["disputed_facts"] / stats["total_checks"]
            stats["inconclusive_rate"] = stats["inconclusive_checks"] / stats["total_checks"]
        else:
            stats["verification_rate"] = 0.0
            stats["dispute_rate"] = 0.0
            stats["inconclusive_rate"] = 0.0
        
        return stats
    
    @observe(as_type="generation")
    async def quick_fact_check(self, claim: str, context: str = "") -> Dict[str, Any]:
        """
        Quick fact check for a single claim with optional context
        
        Args:
            claim: Claim to verify
            context: Optional context to check against
        
        Returns:
            Quick verification result
        """
        try:
            if not context:
                return {
                    "claim": claim,
                    "status": "inconclusive",
                    "confidence": 0.0,
                    "reason": "No context provided for verification"
                }
            
            # Use LLM for quick verification
            prompt = f"""
            Fact-check this claim against the provided context:
            
            Claim: {claim}
            
            Context: {context[:1000]}
            
            Determine if the claim is:
            - VERIFIED: Supported by the context
            - DISPUTED: Contradicted by the context  
            - INCONCLUSIVE: Not enough information
            
            Respond with only: VERIFIED, DISPUTED, or INCONCLUSIVE
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="fact_checking",
                max_tokens=10,
                temperature=0.1
            )
            
            response = result.get("text", "INCONCLUSIVE").strip().upper()
            
            if response == "VERIFIED":
                status = "verified"
                confidence = 0.8
            elif response == "DISPUTED":
                status = "disputed"
                confidence = 0.2
            else:
                status = "inconclusive"
                confidence = 0.5
            
            return {
                "claim": claim,
                "status": status,
                "confidence": confidence,
                "method": "quick_llm_check"
            }
            
        except Exception as e:
            logger.error(f"Quick fact check error: {e}")
            return {
                "claim": claim,
                "status": "error",
                "confidence": 0.0,
                "error": str(e)
            }