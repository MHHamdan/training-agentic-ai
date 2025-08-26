import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from config import config, ModelConfig
from models.hf_models import HuggingFaceModel
from utils.observability import get_observability_manager, track_agent_performance

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetric:
    """Model performance tracking"""
    model_name: str
    task_type: str
    accuracy_score: Optional[float] = None
    latency_ms: Optional[float] = None
    cost_per_token: Optional[float] = None
    error_rate: Optional[float] = None
    total_requests: int = 0
    successful_requests: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

class ModelManager:
    """Advanced multi-model management with performance tracking and comparison"""
    
    def __init__(self):
        self.config = config
        self.models_cache: Dict[str, HuggingFaceModel] = {}
        self.performance_metrics: Dict[str, ModelPerformanceMetric] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_models)
        self.observability = get_observability_manager()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available models"""
        logger.info("Initializing model manager with AgentOps tracking...")
        
        for category, models in self.config.available_models.items():
            logger.info(f"Category {category}: {len(models)} models available")
            
            # Initialize performance tracking for each model
            for model_config in models:
                if model_config.name not in self.performance_metrics:
                    self.performance_metrics[model_config.name] = ModelPerformanceMetric(
                        model_name=model_config.name,
                        task_type=category
                    )
    
    @track_agent_performance("ModelManager", "get_model")
    def get_model(self, model_name: str) -> HuggingFaceModel:
        """Get or create a model instance"""
        if model_name in self.models_cache:
            return self.models_cache[model_name]
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        # Create model instance
        model = HuggingFaceModel(
            model_name=model_config.name,
            api_key=self.config.huggingface_api_key,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature
        )
        
        self.models_cache[model_name] = model
        logger.info(f"Model {model_name} initialized and cached")
        
        return model
    
    async def generate_with_model(self, model_name: str, prompt: str, 
                                 task_type: str = "general") -> Tuple[str, float]:
        """Generate text with a specific model and track performance"""
        start_time = time.time()
        model = self.get_model(model_name)
        
        # Update metrics
        if model_name in self.performance_metrics:
            self.performance_metrics[model_name].total_requests += 1
        
        try:
            result = await model.generate(prompt)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Update success metrics
            if model_name in self.performance_metrics:
                metric = self.performance_metrics[model_name]
                metric.successful_requests += 1
                
                # Update latency (moving average)
                if metric.latency_ms is None:
                    metric.latency_ms = latency
                else:
                    metric.latency_ms = (metric.latency_ms * 0.7) + (latency * 0.3)
            
            # Log to AgentOps
            self.observability.log_tool_usage(
                tool_name=f"HuggingFace-{model_name}",
                input_data={"prompt": prompt[:200], "task_type": task_type},
                output_data={"result": result[:200]},
                duration=latency/1000,
                success=True
            )
            
            return result, latency
            
        except Exception as e:
            logger.error(f"Error generating with model {model_name}: {str(e)}")
            
            # Log error to AgentOps
            self.observability.log_tool_usage(
                tool_name=f"HuggingFace-{model_name}",
                input_data={"prompt": prompt[:200], "task_type": task_type},
                output_data={"error": str(e)},
                duration=(time.time() - start_time),
                success=False
            )
            
            raise
    
    @track_agent_performance("ModelManager", "compare_models")
    async def compare_models(self, prompt: str, models: Optional[List[str]] = None,
                           task_type: str = "financial_analysis") -> Dict[str, Any]:
        """Compare multiple models on the same task"""
        if not models:
            # Use financial optimized models by default
            models = self.config.financial_models[:self.config.max_concurrent_models]
        
        if not models:
            # Fallback to first few models from each category
            models = []
            for category, model_list in self.config.available_models.items():
                if model_list:
                    models.append(model_list[0].name)
                if len(models) >= self.config.max_concurrent_models:
                    break
        
        logger.info(f"Comparing {len(models)} models for task: {task_type}")
        
        # Run models concurrently
        tasks = []
        for model_name in models:
            task = self.generate_with_model(model_name, prompt, task_type)
            tasks.append(task)
        
        results = {}
        performance_results = {}
        
        try:
            # Execute all models concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (model_name, response) in enumerate(zip(models, responses)):
                if isinstance(response, Exception):
                    results[model_name] = {
                        "result": None,
                        "error": str(response),
                        "latency": None,
                        "success": False
                    }
                    performance_results[model_name] = {"score": 0, "latency": float('inf'), "success": False}
                else:
                    result, latency = response
                    results[model_name] = {
                        "result": result,
                        "error": None,
                        "latency": latency,
                        "success": True
                    }
                    
                    # Simple scoring based on result length and latency
                    result_quality_score = min(100, len(result.split()) * 2)  # Basic quality metric
                    latency_score = max(0, 100 - (latency / 100))  # Penalty for high latency
                    overall_score = (result_quality_score * 0.7) + (latency_score * 0.3)
                    
                    performance_results[model_name] = {
                        "score": overall_score,
                        "latency": latency,
                        "success": True,
                        "result_length": len(result)
                    }
        
        except Exception as e:
            logger.error(f"Error in model comparison: {str(e)}")
            raise
        
        # Determine winner
        winner = None
        best_score = 0
        
        for model_name, perf in performance_results.items():
            if perf["success"] and perf["score"] > best_score:
                best_score = perf["score"]
                winner = model_name
        
        comparison_result = {
            "task_type": task_type,
            "models_tested": models,
            "results": results,
            "performance": performance_results,
            "winner": winner,
            "best_score": best_score,
            "timestamp": time.time()
        }
        
        # Log to AgentOps
        self.observability.log_model_comparison(
            models_tested=models,
            performance_results=performance_results,
            winner=winner or "none",
            task_type=task_type
        )
        
        return comparison_result
    
    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the best performing model for a specific task type"""
        task_models = []
        
        # Get models suitable for the task type
        if task_type in ["sentiment", "news_analysis"]:
            category = "sentiment_analysis"
        elif task_type in ["technical_analysis", "math", "calculations"]:
            category = "technical_analysis"
        elif task_type in ["reasoning", "analysis", "general"]:
            category = "financial_reasoning"
        else:
            category = "text_generation"
        
        if category in self.config.available_models:
            task_models = [m.name for m in self.config.available_models[category]]
        
        if not task_models:
            return None
        
        # Find model with best performance
        best_model = None
        best_score = 0
        
        for model_name in task_models:
            if model_name in self.performance_metrics:
                metric = self.performance_metrics[model_name]
                # Combine success rate and latency for scoring
                score = metric.success_rate * 100
                if metric.latency_ms:
                    score -= (metric.latency_ms / 100)  # Penalty for high latency
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model or task_models[0]  # Fallback to first model
    
    async def analyze_financial_text(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Specialized financial text analysis using best models"""
        results = {}
        
        # Sentiment Analysis
        if analysis_type in ["comprehensive", "sentiment"]:
            sentiment_model = self.get_best_model_for_task("sentiment")
            if sentiment_model:
                sentiment_prompt = f"""Analyze the financial sentiment of this text. 
                Classify as POSITIVE, NEGATIVE, or NEUTRAL and provide confidence score.
                
                Text: {text[:1000]}
                
                Response format:
                Sentiment: [POSITIVE/NEGATIVE/NEUTRAL]
                Confidence: [0-100]
                Reasoning: [brief explanation]"""
                
                try:
                    sentiment_result, _ = await self.generate_with_model(sentiment_model, sentiment_prompt, "sentiment")
                    results["sentiment"] = sentiment_result
                except Exception as e:
                    logger.error(f"Sentiment analysis failed: {str(e)}")
                    results["sentiment"] = "Error in sentiment analysis"
        
        # Financial Reasoning
        if analysis_type in ["comprehensive", "reasoning"]:
            reasoning_model = self.get_best_model_for_task("reasoning")
            if reasoning_model:
                reasoning_prompt = f"""Provide financial analysis and insights for this text.
                Focus on market implications, risk factors, and investment considerations.
                
                Text: {text[:2000]}
                
                Provide structured analysis covering:
                1. Key Financial Points
                2. Market Implications  
                3. Risk Assessment
                4. Investment Outlook"""
                
                try:
                    reasoning_result, _ = await self.generate_with_model(reasoning_model, reasoning_prompt, "reasoning")
                    results["analysis"] = reasoning_result
                except Exception as e:
                    logger.error(f"Financial reasoning failed: {str(e)}")
                    results["analysis"] = "Error in financial analysis"
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        summary = {
            "total_models": len(self.performance_metrics),
            "models": {}
        }
        
        for model_name, metric in self.performance_metrics.items():
            summary["models"][model_name] = {
                "success_rate": metric.success_rate,
                "total_requests": metric.total_requests,
                "average_latency_ms": metric.latency_ms,
                "task_type": metric.task_type
            }
        
        return summary
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.models_cache.clear()
        logger.info("Model manager cleanup completed")