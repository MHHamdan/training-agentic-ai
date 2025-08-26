"""
Synthesizer Agent - Information synthesis specialist
Generates comprehensive reports with citations
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
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

logger = logging.getLogger(__name__)

class SynthesizerAgent:
    """
    Information synthesis specialist
    Creates comprehensive reports from analyzed content
    """
    
    def __init__(self):
        """Initialize synthesizer agent"""
        self.config = config
        self.model_manager = ModelManager()
        self.synthesis_stats = {
            "total_syntheses": 0,
            "successful_syntheses": 0,
            "failed_syntheses": 0,
            "average_processing_time": 0
        }
    
    @observe(as_type="generation")
    async def synthesize(
        self,
        analyzed_content: List[Dict[str, Any]],
        key_insights: List[str],
        query: str,
        max_length: int = 5000
    ) -> Dict[str, Any]:
        """
        Synthesize information into comprehensive report
        
        Args:
            analyzed_content: List of analyzed sources
            key_insights: Extracted key insights
            query: Original research query
            max_length: Maximum synthesis length
        
        Returns:
            Synthesis results with citations
        """
        try:
            # Track with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Synthesizing research for: {query}",
                    metadata={
                        "source_count": len(analyzed_content),
                        "insights_count": len(key_insights),
                        "max_length": max_length,
                        "agent": "synthesizer_agent"
                    }
                )
            
            start_time = datetime.now()
            
            # Generate comprehensive synthesis
            synthesis_text = await self._generate_synthesis(
                analyzed_content, key_insights, query, max_length
            )
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(
                synthesis_text, query
            )
            
            # Generate detailed findings
            detailed_findings = await self._generate_detailed_findings(
                analyzed_content, key_insights
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                key_insights, query
            )
            
            # Generate citations
            citations = self._generate_citations(analyzed_content)
            
            # Generate bibliography
            bibliography = self._generate_bibliography(analyzed_content)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.synthesis_stats["total_syntheses"] += 1
            self.synthesis_stats["successful_syntheses"] += 1
            self._update_average_processing_time(processing_time)
            
            # Prepare results
            results = {
                "synthesis": synthesis_text,
                "executive_summary": executive_summary,
                "detailed_findings": detailed_findings,
                "recommendations": recommendations,
                "citations": citations,
                "bibliography": bibliography,
                "metadata": {
                    "query": query,
                    "sources_used": len(analyzed_content),
                    "insights_count": len(key_insights),
                    "synthesis_length": len(synthesis_text),
                    "processing_time": processing_time,
                    "citation_format": self.config.research.citation_format
                }
            }
            
            # Track output with Langfuse
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Synthesis complete: {len(synthesis_text)} characters",
                    metadata={
                        "processing_time": processing_time,
                        "synthesis_length": len(synthesis_text),
                        "citations_count": len(citations)
                    }
                )
            
            logger.info(f"Synthesis completed in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            self.synthesis_stats["failed_syntheses"] += 1
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Synthesis failed: {str(e)}",
                    metadata={"error": str(e)}
                )
            
            return {
                "synthesis": "",
                "executive_summary": "",
                "detailed_findings": {},
                "recommendations": [],
                "citations": [],
                "bibliography": [],
                "error": str(e)
            }
    
    async def _generate_synthesis(
        self,
        analyzed_content: List[Dict[str, Any]],
        key_insights: List[str],
        query: str,
        max_length: int
    ) -> str:
        """Generate comprehensive synthesis"""
        try:
            # Prepare content summary for LLM
            content_summaries = []
            for i, content in enumerate(analyzed_content, 1):
                summary = f"Source {i}: {content.get('summary', '')}"
                if content.get('key_points'):
                    summary += f"\nKey points: {'; '.join(content['key_points'][:3])}"
                content_summaries.append(summary)
            
            # Create synthesis prompt
            prompt = f"""
            Research Query: {query}
            
            Source Summaries:
            {chr(10).join(content_summaries[:10])}  # Limit to 10 sources
            
            Key Insights:
            {chr(10).join(f"- {insight}" for insight in key_insights[:15])}  # Limit to 15 insights
            
            Create a comprehensive research synthesis that:
            1. Directly addresses the research query
            2. Integrates findings from multiple sources
            3. Identifies patterns and themes
            4. Presents balanced analysis
            5. Maintains academic rigor
            
            The synthesis should be approximately {max_length // 4} words and follow this structure:
            - Introduction with query context
            - Main findings organized by themes
            - Analysis and interpretation
            - Conclusion with key takeaways
            
            Write in a clear, professional academic style.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="synthesis",
                max_tokens=max_length // 2,  # Rough token estimate
                temperature=0.4
            )
            
            return result.get("text", "Synthesis generation failed")
            
        except Exception as e:
            logger.error(f"Synthesis generation error: {e}")
            return f"Error generating synthesis: {str(e)}"
    
    async def _generate_executive_summary(self, synthesis: str, query: str) -> str:
        """Generate executive summary"""
        try:
            prompt = f"""
            Based on the following research synthesis, create a concise executive summary 
            that captures the most important findings for: {query}
            
            Research Synthesis:
            {synthesis[:2000]}  # Limit input length
            
            The executive summary should:
            - Be 3-4 sentences maximum
            - Highlight the most critical findings
            - Be accessible to non-experts
            - Focus on actionable insights
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="summarization",
                max_tokens=200,
                temperature=0.3
            )
            
            return result.get("text", "Executive summary generation failed")
            
        except Exception as e:
            logger.error(f"Executive summary generation error: {e}")
            return f"Error generating executive summary: {str(e)}"
    
    async def _generate_detailed_findings(
        self,
        analyzed_content: List[Dict[str, Any]],
        key_insights: List[str]
    ) -> Dict[str, str]:
        """Generate detailed findings by category"""
        try:
            # Categorize insights
            categories = {
                "Key Findings": [],
                "Trends and Patterns": [],
                "Implications": [],
                "Contradictions": []
            }
            
            # Simple categorization based on keywords
            for insight in key_insights:
                insight_lower = insight.lower()
                if any(word in insight_lower for word in ["trend", "increase", "decrease", "pattern"]):
                    categories["Trends and Patterns"].append(insight)
                elif any(word in insight_lower for word in ["implication", "consequence", "result"]):
                    categories["Implications"].append(insight)
                elif any(word in insight_lower for word in ["however", "but", "contradiction", "despite"]):
                    categories["Contradictions"].append(insight)
                else:
                    categories["Key Findings"].append(insight)
            
            # Generate detailed text for each category
            detailed_findings = {}
            for category, insights in categories.items():
                if insights:
                    detailed_findings[category] = "\n".join(f"• {insight}" for insight in insights[:5])
                else:
                    detailed_findings[category] = "No specific findings in this category."
            
            return detailed_findings
            
        except Exception as e:
            logger.error(f"Detailed findings generation error: {e}")
            return {"Error": f"Error generating detailed findings: {str(e)}"}
    
    async def _generate_recommendations(
        self,
        key_insights: List[str],
        query: str
    ) -> List[str]:
        """Generate actionable recommendations"""
        try:
            insights_text = "\n".join(f"- {insight}" for insight in key_insights[:10])
            
            prompt = f"""
            Based on these research insights for the query "{query}":
            
            {insights_text}
            
            Generate 3-5 specific, actionable recommendations. Each recommendation should:
            - Be practical and implementable
            - Be based on the research findings
            - Address the original query
            - Be clearly stated
            
            Format as a simple list, one recommendation per line.
            """
            
            result = await self.model_manager.generate_text(
                prompt=prompt,
                task_type="general_research",
                max_tokens=300,
                temperature=0.5
            )
            
            if result.get("text"):
                # Parse recommendations
                recommendations = [
                    rec.strip().lstrip("•-123456789. ")
                    for rec in result["text"].split('\n')
                    if rec.strip() and len(rec.strip()) > 10
                ]
                return recommendations[:5]
            
            return ["Unable to generate specific recommendations from the available data."]
            
        except Exception as e:
            logger.error(f"Recommendations generation error: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    def _generate_citations(self, analyzed_content: List[Dict[str, Any]]) -> List[str]:
        """Generate citations for sources"""
        citations = []
        
        for i, content in enumerate(analyzed_content, 1):
            try:
                title = content.get("title", "Unknown Title")
                source_id = content.get("source_id", "")
                
                # Extract domain from URL if available
                if source_id.startswith("http"):
                    from urllib.parse import urlparse
                    domain = urlparse(source_id).netloc
                    source_name = domain
                else:
                    source_name = content.get("source", "Unknown Source")
                
                # Simple citation format (APA-style)
                if self.config.research.citation_format == "APA":
                    citation = f"{title}. Retrieved from {source_name}"
                elif self.config.research.citation_format == "MLA":
                    citation = f'"{title}." {source_name}, Web.'
                else:
                    citation = f"{title} - {source_name}"
                
                citations.append(citation)
                
            except Exception as e:
                logger.error(f"Citation generation error for source {i}: {e}")
                citations.append(f"Source {i}: Citation generation failed")
        
        return citations
    
    def _generate_bibliography(self, analyzed_content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate structured bibliography"""
        bibliography = []
        
        for content in analyzed_content:
            try:
                entry = {
                    "title": content.get("title", "Unknown Title"),
                    "source": content.get("source", "Unknown Source"),
                    "url": content.get("source_id", ""),
                    "relevance_score": content.get("relevance_score", 0.0),
                    "quality_score": content.get("quality_score", 0.0),
                    "access_date": datetime.now().strftime("%Y-%m-%d")
                }
                
                # Add author if available
                if "author" in content.get("metadata", {}):
                    entry["author"] = content["metadata"]["author"]
                
                bibliography.append(entry)
                
            except Exception as e:
                logger.error(f"Bibliography entry generation error: {e}")
        
        return bibliography
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        if self.synthesis_stats["total_syntheses"] == 1:
            self.synthesis_stats["average_processing_time"] = processing_time
        else:
            # Running average
            total = self.synthesis_stats["average_processing_time"] * (self.synthesis_stats["total_syntheses"] - 1)
            self.synthesis_stats["average_processing_time"] = (total + processing_time) / self.synthesis_stats["total_syntheses"]
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics"""
        return self.synthesis_stats