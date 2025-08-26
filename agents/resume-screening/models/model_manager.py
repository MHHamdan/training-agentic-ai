import logging
from typing import Dict, List, Optional, Any
from langsmith import traceable
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import config
from models.hf_models import HuggingFaceModel

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.config = config
        self.models_cache: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._initialize_models()
    
    def _initialize_models(self):
        logger.info("Initializing model manager...")
        for category, models in self.config.available_models.items():
            logger.info(f"Category {category}: {len(models)} models available")
    
    @traceable(name="get_model", metadata={"component": "model_manager"})
    def get_model(self, model_name: str) -> Any:
        if model_name in self.models_cache:
            return self.models_cache[model_name]
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        if model_config.provider == "huggingface":
            model = HuggingFaceModel(
                model_name=model_config.name,
                api_key=self.config.huggingface_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "openai":
            from .cloud_models import OpenAIModel
            model = OpenAIModel(
                model_name=model_config.name,
                api_key=self.config.openai_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "anthropic":
            from .cloud_models import AnthropicModel
            model = AnthropicModel(
                model_name=model_config.name,
                api_key=self.config.anthropic_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "google":
            from .cloud_models import GoogleModel
            model = GoogleModel(
                model_name=model_config.name,
                api_key=self.config.google_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "cohere":
            from .cloud_models import CohereModel
            model = CohereModel(
                model_name=model_config.name,
                api_key=self.config.cohere_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "groq":
            from .cloud_models import GroqModel
            model = GroqModel(
                model_name=model_config.name,
                api_key=self.config.groq_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "mistral":
            from .cloud_models import MistralModel
            model = MistralModel(
                model_name=model_config.name,
                api_key=self.config.mistral_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "deepseek":
            from .cloud_models import DeepSeekModel
            model = DeepSeekModel(
                model_name=model_config.name,
                api_key=self.config.deepseek_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "xai":
            from .cloud_models import XAIModel
            model = XAIModel(
                model_name=model_config.name,
                api_key=self.config.xai_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        elif model_config.provider == "cerebras":
            from .cloud_models import CerebrasModel
            model = CerebrasModel(
                model_name=model_config.name,
                api_key=self.config.cerebras_api_key,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature
            )
        else:
            raise ValueError(f"Provider {model_config.provider} not supported")
        
        self.models_cache[model_name] = model
        return model
    
    @traceable(name="get_available_models", metadata={"component": "model_manager"})
    def get_available_models(self) -> Dict[str, List[str]]:
        available = {}
        for category, models in self.config.available_models.items():
            available[category] = [m.name for m in models]
        return available
    
    @traceable(name="benchmark_models", metadata={"component": "model_manager"})
    async def benchmark_models(
        self,
        test_input: str,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not model_names:
            model_names = [self.config.default_model]
        
        benchmarks = {}
        
        for model_name in model_names:
            try:
                import time
                start_time = time.time()
                
                model = self.get_model(model_name)
                response = await model.generate(test_input)
                
                end_time = time.time()
                
                benchmarks[model_name] = {
                    "response_time": end_time - start_time,
                    "response_length": len(response) if response else 0,
                    "status": "success"
                }
            except Exception as e:
                benchmarks[model_name] = {
                    "error": str(e),
                    "status": "failed"
                }
        
        return benchmarks
    
    def cleanup(self):
        self.executor.shutdown(wait=True)
        self.models_cache.clear()