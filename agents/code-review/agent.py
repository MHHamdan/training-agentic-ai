"""
Main Code Review Agent Orchestrator
Enterprise-grade AI code review with multi-provider support
Author: Mohammed Hamdan
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.api_key_manager import APIKeyManager
from models.hf_models import HuggingFaceCodeModels
from analyzers.security_analyzer import SecurityAnalyzer, SecurityAnalysisResult
from analyzers.performance_analyzer import PerformanceAnalyzer, PerformanceAnalysisResult
from analyzers.style_analyzer import StyleAnalyzer, StyleAnalysisResult

logger = logging.getLogger(__name__)

@dataclass
class CodeReviewRequest:
    """Code review request structure"""
    code: str
    context: str = ""
    providers: List[str] = None
    analysis_types: List[str] = None
    include_ai_analysis: bool = True
    user_api_keys: Dict[str, str] = None

@dataclass
class CodeReviewResult:
    """Complete code review result"""
    request_id: str
    timestamp: str
    overall_score: float
    total_issues: int
    security_result: Optional[SecurityAnalysisResult] = None
    performance_result: Optional[PerformanceAnalysisResult] = None
    style_result: Optional[StyleAnalysisResult] = None
    providers_used: List[str] = None
    analysis_time: float = 0.0
    recommendations: List[str] = None
    code_metrics: Dict[str, Any] = None

class CodeReviewAgent:
    """
    Production-Ready Code Review Agent with Multi-Provider Support
    
    Features:
    - Multi-provider API key management (user keys + HuggingFace fallback)
    - Comprehensive analysis (security, performance, style)
    - Full observability with LangSmith/AgentOps
    - Enterprise-grade error handling and logging
    """
    
    def __init__(self):
        """Initialize the code review agent"""
        self.config = config
        self.api_key_manager = APIKeyManager()
        self.hf_models = HuggingFaceCodeModels()
        
        # Initialize analyzers
        self.security_analyzer = SecurityAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.style_analyzer = StyleAnalyzer()
        
        # Session tracking
        self.session_stats = {
            'total_reviews': 0,
            'total_issues_found': 0,
            'avg_analysis_time': 0.0,
            'provider_usage': {},
            'analysis_type_usage': {
                'security': 0,
                'performance': 0,
                'style': 0
            }
        }
        
        logger.info(f"CodeReviewAgent V{self.config.agent_version} initialized")
        logger.info(f"Available providers: {self.api_key_manager.get_available_providers()}")
    
    @observe(as_type="code_review_session")
    async def review_code(
        self,
        request: Union[CodeReviewRequest, str],
        context: str = "",
        providers: List[str] = None,
        analysis_types: List[str] = None,
        include_ai_analysis: bool = True,
        user_api_keys: Dict[str, str] = None
    ) -> CodeReviewResult:
        """
        Main code review endpoint with multi-provider support
        
        Args:
            request: CodeReviewRequest object or code string
            context: Additional context for analysis
            providers: List of providers to use
            analysis_types: Types of analysis to perform
            include_ai_analysis: Whether to include AI analysis
            user_api_keys: User-provided API keys
            
        Returns:
            CodeReviewResult with comprehensive analysis
        """
        start_time = datetime.now()
        request_id = f"review_{int(start_time.timestamp())}"
        
        try:
            # Handle different input types
            if isinstance(request, str):
                request = CodeReviewRequest(
                    code=request,
                    context=context,
                    providers=providers,
                    analysis_types=analysis_types,
                    include_ai_analysis=include_ai_analysis,
                    user_api_keys=user_api_keys
                )
            
            # Set up observability
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Code review request - {len(request.code)} characters",
                    metadata={
                        "request_id": request_id,
                        "organization": "code-review-org",
                        "project": "code-review-agent-v2",
                        "code_length": len(request.code),
                        "providers_requested": request.providers or [],
                        "analysis_types": request.analysis_types or [],
                        "ai_analysis_enabled": request.include_ai_analysis
                    }
                )
            
            # Set user API keys if provided
            if request.user_api_keys:
                self.api_key_manager.set_user_api_keys(request.user_api_keys)
            
            # Determine providers to use
            if not request.providers:
                available_providers = self.api_key_manager.get_available_providers()
                request.providers = [
                    provider_id for provider_id, info in available_providers.items()
                    if info["status"] == "available"
                ]
                
                # Prioritize HuggingFace
                if "huggingface" in request.providers:
                    request.providers = ["huggingface"] + [p for p in request.providers if p != "huggingface"]
            
            # Determine analysis types
            if not request.analysis_types:
                request.analysis_types = ["security", "performance", "style"]
            
            # Validate input
            validation_result = self._validate_request(request)
            if not validation_result["valid"]:
                raise ValueError(f"Invalid request: {validation_result['error']}")
            
            # Run comprehensive analysis
            analysis_results = await self._run_comprehensive_analysis(request)
            
            # Compile final result
            result = self._compile_review_result(
                request_id, request, analysis_results, start_time
            )
            
            # Update session statistics
            self._update_session_stats(result)
            
            # Log completion
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Code review complete - Score: {result.overall_score:.1f}/10",
                    metadata={
                        "overall_score": result.overall_score,
                        "total_issues": result.total_issues,
                        "analysis_time": result.analysis_time,
                        "providers_used": result.providers_used,
                        "security_score": result.security_result.security_score if result.security_result else None,
                        "performance_score": result.performance_result.performance_score if result.performance_result else None,
                        "style_score": result.style_result.style_score if result.style_result else None
                    }
                )
            
            logger.info(f"Code review completed: {request_id}, Score: {result.overall_score:.1f}/10")
            return result
            
        except Exception as e:
            logger.error(f"Code review error: {e}")
            
            # Return error result
            error_result = CodeReviewResult(
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                overall_score=0.0,
                total_issues=1,
                analysis_time=(datetime.now() - start_time).total_seconds(),
                recommendations=[f"Analysis failed: {str(e)}"],
                providers_used=[]
            )
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Code review failed: {str(e)}",
                    metadata={"error": str(e)}
                )
            
            return error_result
    
    @observe(as_type="provider_comparison")
    async def compare_providers(
        self,
        code: str,
        providers: List[str],
        analysis_type: str = "security"
    ) -> Dict[str, Any]:
        """
        Compare analysis results across different providers
        
        Args:
            code: Code to analyze
            providers: List of providers to compare
            analysis_type: Type of analysis to compare
            
        Returns:
            Comparison results with provider performance metrics
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Provider comparison - {len(providers)} providers",
                    metadata={
                        "providers": providers,
                        "analysis_type": analysis_type,
                        "code_length": len(code)
                    }
                )
            
            comparison_results = {
                "analysis_type": analysis_type,
                "providers_tested": providers,
                "results": {},
                "performance_metrics": {},
                "recommendations": {}
            }
            
            # Run analysis with each provider
            for provider in providers:
                try:
                    start_time = datetime.now()
                    
                    if analysis_type == "security":
                        result = await self.security_analyzer.analyze_security(
                            code, "", include_ai_analysis=True
                        )
                        score = result.security_score
                        issues = result.total_issues
                        
                    elif analysis_type == "performance":
                        result = await self.performance_analyzer.analyze_performance(
                            code, "", include_ai_analysis=True
                        )
                        score = result.performance_score
                        issues = result.total_issues
                        
                    elif analysis_type == "style":
                        result = await self.style_analyzer.analyze_style(
                            code, "", include_ai_analysis=True
                        )
                        score = result.style_score
                        issues = result.total_issues
                    
                    else:
                        continue
                    
                    processing_time = (datetime.now() - start_time).total_seconds()
                    
                    comparison_results["results"][provider] = {
                        "score": score,
                        "issues_found": issues,
                        "processing_time": processing_time,
                        "success": True
                    }
                    
                    comparison_results["performance_metrics"][provider] = {
                        "response_time": processing_time,
                        "accuracy_estimate": score / 10.0,
                        "reliability": 1.0  # Assume success means reliable
                    }
                    
                except Exception as e:
                    logger.warning(f"Provider {provider} failed: {e}")
                    comparison_results["results"][provider] = {
                        "error": str(e),
                        "success": False
                    }
            
            # Generate recommendations
            successful_results = {
                k: v for k, v in comparison_results["results"].items()
                if v.get("success", False)
            }
            
            if successful_results:
                best_provider = max(
                    successful_results.keys(),
                    key=lambda p: successful_results[p]["score"]
                )
                fastest_provider = min(
                    successful_results.keys(),
                    key=lambda p: successful_results[p]["processing_time"]
                )
                
                comparison_results["recommendations"] = {
                    "best_accuracy": best_provider,
                    "fastest_response": fastest_provider,
                    "overall_recommendation": best_provider
                }
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Provider comparison complete",
                    metadata={
                        "successful_providers": len(successful_results),
                        "best_provider": comparison_results["recommendations"].get("best_accuracy"),
                        "avg_processing_time": sum(
                            r["processing_time"] for r in successful_results.values()
                        ) / len(successful_results) if successful_results else 0
                    }
                )
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Provider comparison error: {e}")
            return {
                "error": str(e),
                "analysis_type": analysis_type,
                "providers_tested": providers,
                "results": {}
            }
    
    @observe(as_type="security_scan")
    async def security_scan(
        self,
        code: str,
        provider: str = "auto",
        include_ai_analysis: bool = True
    ) -> SecurityAnalysisResult:
        """
        Dedicated security vulnerability scanning
        
        Args:
            code: Code to scan
            provider: Provider to use ("auto" for best available)
            include_ai_analysis: Whether to include AI analysis
            
        Returns:
            SecurityAnalysisResult with vulnerability findings
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Security scan - {len(code)} characters",
                    metadata={
                        "scan_type": "security_only",
                        "provider": provider,
                        "ai_analysis": include_ai_analysis
                    }
                )
            
            # Auto-select provider
            if provider == "auto":
                provider = self.api_key_manager.get_recommended_provider("security_analysis")
            
            # Run security analysis
            result = await self.security_analyzer.analyze_security(
                code, "", include_ai_analysis
            )
            
            # Track provider usage
            if provider not in self.session_stats['provider_usage']:
                self.session_stats['provider_usage'][provider] = 0
            self.session_stats['provider_usage'][provider] += 1
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Security scan complete - {result.total_issues} issues found",
                    metadata={
                        "security_score": result.security_score,
                        "vulnerabilities_found": result.total_issues,
                        "critical_issues": result.critical_issues,
                        "provider_used": provider
                    }
                )
            
            logger.info(f"Security scan completed: {result.total_issues} vulnerabilities found")
            return result
            
        except Exception as e:
            logger.error(f"Security scan error: {e}")
            raise
    
    @observe(as_type="performance_audit")
    async def performance_audit(
        self,
        code: str,
        provider: str = "auto",
        include_ai_analysis: bool = True
    ) -> PerformanceAnalysisResult:
        """
        Performance optimization analysis
        
        Args:
            code: Code to analyze
            provider: Provider to use ("auto" for best available)
            include_ai_analysis: Whether to include AI analysis
            
        Returns:
            PerformanceAnalysisResult with optimization suggestions
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Performance audit - {len(code)} characters",
                    metadata={
                        "audit_type": "performance_only",
                        "provider": provider,
                        "ai_analysis": include_ai_analysis
                    }
                )
            
            # Auto-select provider
            if provider == "auto":
                provider = self.api_key_manager.get_recommended_provider("performance_analysis")
            
            # Run performance analysis
            result = await self.performance_analyzer.analyze_performance(
                code, "", include_ai_analysis
            )
            
            # Track provider usage
            if provider not in self.session_stats['provider_usage']:
                self.session_stats['provider_usage'][provider] = 0
            self.session_stats['provider_usage'][provider] += 1
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Performance audit complete - {result.total_issues} issues found",
                    metadata={
                        "performance_score": result.performance_score,
                        "issues_found": result.total_issues,
                        "critical_issues": result.critical_issues,
                        "provider_used": provider
                    }
                )
            
            logger.info(f"Performance audit completed: {result.total_issues} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Performance audit error: {e}")
            raise
    
    def _validate_request(self, request: CodeReviewRequest) -> Dict[str, Any]:
        """Validate code review request"""
        if not request.code or not request.code.strip():
            return {"valid": False, "error": "Code cannot be empty"}
        
        if len(request.code) > self.config.analysis.max_code_length:
            return {
                "valid": False,
                "error": f"Code exceeds maximum length of {self.config.analysis.max_code_length} characters"
            }
        
        if request.providers:
            available = self.api_key_manager.get_available_providers()
            invalid_providers = [
                p for p in request.providers
                if p not in available or available[p]["status"] != "available"
            ]
            if invalid_providers:
                return {
                    "valid": False,
                    "error": f"Invalid or unavailable providers: {invalid_providers}"
                }
        
        if request.analysis_types:
            valid_types = ["security", "performance", "style"]
            invalid_types = [t for t in request.analysis_types if t not in valid_types]
            if invalid_types:
                return {
                    "valid": False,
                    "error": f"Invalid analysis types: {invalid_types}"
                }
        
        return {"valid": True}
    
    async def _run_comprehensive_analysis(self, request: CodeReviewRequest) -> Dict[str, Any]:
        """Run comprehensive analysis based on request"""
        results = {}
        
        # Run requested analyses
        if "security" in request.analysis_types:
            results["security"] = await self.security_analyzer.analyze_security(
                request.code, request.context, request.include_ai_analysis
            )
            self.session_stats['analysis_type_usage']['security'] += 1
        
        if "performance" in request.analysis_types:
            results["performance"] = await self.performance_analyzer.analyze_performance(
                request.code, request.context, request.include_ai_analysis
            )
            self.session_stats['analysis_type_usage']['performance'] += 1
        
        if "style" in request.analysis_types:
            results["style"] = await self.style_analyzer.analyze_style(
                request.code, request.context, request.include_ai_analysis
            )
            self.session_stats['analysis_type_usage']['style'] += 1
        
        return results
    
    def _compile_review_result(
        self,
        request_id: str,
        request: CodeReviewRequest,
        analysis_results: Dict[str, Any],
        start_time: datetime
    ) -> CodeReviewResult:
        """Compile final code review result"""
        
        # Calculate overall score
        scores = []
        total_issues = 0
        
        security_result = analysis_results.get("security")
        performance_result = analysis_results.get("performance")
        style_result = analysis_results.get("style")
        
        if security_result:
            scores.append(security_result.security_score)
            total_issues += security_result.total_issues
        
        if performance_result:
            scores.append(performance_result.performance_score)
            total_issues += performance_result.total_issues
        
        if style_result:
            scores.append(style_result.style_score)
            total_issues += style_result.total_issues
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Generate recommendations
        recommendations = []
        if overall_score < 5:
            recommendations.append("🚨 Critical: Immediate attention required for code quality")
        elif overall_score < 7:
            recommendations.append("⚠️ Important: Several improvements needed")
        else:
            recommendations.append("✅ Good: Code meets quality standards")
        
        if security_result and security_result.critical_issues > 0:
            recommendations.append(f"🔒 Security: {security_result.critical_issues} critical vulnerabilities need immediate fixes")
        
        if performance_result and performance_result.critical_issues > 0:
            recommendations.append(f"⚡ Performance: {performance_result.critical_issues} critical bottlenecks identified")
        
        # Calculate code metrics
        code_metrics = {
            "lines_of_code": len([line for line in request.code.split('\n') if line.strip()]),
            "characters": len(request.code),
            "functions": request.code.count('def '),
            "classes": request.code.count('class '),
            "complexity_estimate": min(10, request.code.count('if ') + request.code.count('for ') + request.code.count('while '))
        }
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        return CodeReviewResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            total_issues=total_issues,
            security_result=security_result,
            performance_result=performance_result,
            style_result=style_result,
            providers_used=request.providers,
            analysis_time=analysis_time,
            recommendations=recommendations,
            code_metrics=code_metrics
        )
    
    def _update_session_stats(self, result: CodeReviewResult):
        """Update session statistics"""
        self.session_stats['total_reviews'] += 1
        self.session_stats['total_issues_found'] += result.total_issues
        
        # Update average analysis time
        current_avg = self.session_stats['avg_analysis_time']
        total_reviews = self.session_stats['total_reviews']
        
        if total_reviews == 1:
            self.session_stats['avg_analysis_time'] = result.analysis_time
        else:
            self.session_stats['avg_analysis_time'] = (
                (current_avg * (total_reviews - 1) + result.analysis_time) / total_reviews
            )
        
        # Update provider usage
        for provider in result.providers_used or []:
            if provider not in self.session_stats['provider_usage']:
                self.session_stats['provider_usage'][provider] = 0
            self.session_stats['provider_usage'][provider] += 1
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            **self.session_stats,
            "available_providers": len(self.api_key_manager.get_available_providers()),
            "config_valid": all(self.config.validate_configuration().values())
        }
    
    def get_supported_features(self) -> Dict[str, Any]:
        """Get list of supported features and capabilities"""
        return {
            "analysis_types": ["security", "performance", "style"],
            "supported_providers": list(self.config.SUPPORTED_PROVIDERS.keys()),
            "security_checks": list(self.config.ANALYSIS_CATEGORIES["security"]["checks"]),
            "performance_checks": list(self.config.ANALYSIS_CATEGORIES["performance"]["checks"]),
            "style_checks": list(self.config.ANALYSIS_CATEGORIES["style"]["checks"]),
            "observability": {
                "langsmith_enabled": self.config.observability.langsmith_enabled,
                "agentops_enabled": self.config.observability.agentops_enabled
            },
            "version": self.config.agent_version
        }