"""
Research Agent Configuration Management
Production-ready configuration with Langfuse integration
"""

import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LangfuseConfig(BaseModel):
    """Langfuse observability configuration"""
    secret_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    public_key: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    host: str = Field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
    project: str = Field(default_factory=lambda: os.getenv("LANGFUSE_PROJECT", "research-agent-v2"))
    organization: str = Field(default_factory=lambda: os.getenv("LANGFUSE_ORGANIZATION", "research-org"))
    environment: str = Field(default_factory=lambda: os.getenv("LANGFUSE_ENVIRONMENT", "production"))
    tags: List[str] = Field(default_factory=lambda: os.getenv("LANGFUSE_TAGS", "research,multi-agent,evaluation").split(","))
    enabled: bool = Field(default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "true").lower() == "true")

class ModelConfig(BaseModel):
    """Multi-model configuration for research tasks"""
    
    # Research Models
    general_research_models: List[str] = Field(default=[
        "microsoft/DialoGPT-large",
        "microsoft/phi-3-mini-4k-instruct", 
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.1"
    ])
    
    analysis_models: List[str] = Field(default=[
        "facebook/bart-large-cnn",
        "microsoft/deberta-v3-large", 
        "sentence-transformers/all-MiniLM-L6-v2"
    ])
    
    specialized_research_models: List[str] = Field(default=[
        "allenai/scibert_scivocab_uncased",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "google/flan-t5-large"
    ])
    
    text_generation_models: List[str] = Field(default=[
        "google/gemma-2b-it",
        "meta-llama/Llama-2-7b-chat-hf",
        "EleutherAI/gpt-j-6b"
    ])
    
    # Model Selection
    default_model: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "microsoft/phi-3-mini-4k-instruct"))
    enable_model_comparison: bool = Field(default_factory=lambda: os.getenv("ENABLE_MODEL_COMPARISON", "true").lower() == "true")
    max_concurrent_models: int = Field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_MODELS", "3")))
    research_model_preference: str = Field(default_factory=lambda: os.getenv("RESEARCH_MODEL_PREFERENCE", "scientific"))

class ResearchConfig(BaseModel):
    """Research-specific settings"""
    max_search_results: int = Field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "20")))
    analysis_depth: str = Field(default_factory=lambda: os.getenv("ANALYSIS_DEPTH", "comprehensive"))
    citation_format: str = Field(default_factory=lambda: os.getenv("CITATION_FORMAT", "APA"))
    fact_check_threshold: float = Field(default_factory=lambda: float(os.getenv("FACT_CHECK_THRESHOLD", "0.8")))
    quality_score_minimum: float = Field(default_factory=lambda: float(os.getenv("QUALITY_SCORE_MINIMUM", "0.7")))

class PerformanceConfig(BaseModel):
    """Performance and timing configuration"""
    max_research_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("MAX_RESEARCH_TIMEOUT_SECONDS", "900")))
    search_result_limit: int = Field(default_factory=lambda: int(os.getenv("SEARCH_RESULT_LIMIT", "50")))
    synthesis_max_length: int = Field(default_factory=lambda: int(os.getenv("SYNTHESIS_MAX_LENGTH", "5000")))
    evaluation_depth: str = Field(default_factory=lambda: os.getenv("EVALUATION_DEPTH", "detailed"))

class APIConfig(BaseModel):
    """API keys and external service configuration"""
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    huggingface_api_key: str = Field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    duckduckgo_api_key: str = Field(default_factory=lambda: os.getenv("DUCKDUCKGO_API_KEY", ""))
    newsapi_key: str = Field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    arxiv_api_key: str = Field(default_factory=lambda: os.getenv("ARXIV_API_KEY", ""))

class ResearchAgentConfig(BaseSettings):
    """Main configuration class for Research Agent"""
    
    # Core configurations
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig) 
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    apis: APIConfig = Field(default_factory=APIConfig)
    
    # Agent metadata
    agent_name: str = "Research Agent"
    agent_version: str = "2.0.0"
    agent_id: str = "research-agent-v2"
    port: int = 8514
    
    # Environment settings
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment

    def get_model_by_task(self, task_type: str) -> str:
        """Get recommended model for specific research task - prioritizes free HF models"""
        # Always prioritize free Hugging Face models (2025 updated)
        model_mapping = {
            "general_research": "meta-llama/Llama-3.1-8B-Instruct",  # Top performer 2025
            "analysis": "microsoft/Phi-3.5-mini-instruct",  # Best reasoning model
            "scientific": "Qwen/Qwen2.5-7B-Instruct",  # Excellent for research
            "text_generation": "meta-llama/Meta-Llama-3-8B",  # Rivals GPT models
            "summarization": "microsoft/Phi-3-mini-4k-instruct",  # Great for summaries 
            "fact_checking": "microsoft/Phi-3.5-mini-instruct",  # Strong logic capabilities
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2"
        }
        return model_mapping.get(task_type, "meta-llama/Llama-3.1-8B-Instruct")
    
    def get_langfuse_config(self) -> Dict[str, Any]:
        """Get Langfuse configuration dictionary"""
        return {
            "secret_key": self.langfuse.secret_key,
            "public_key": self.langfuse.public_key,
            "host": self.langfuse.host,
            "project": self.langfuse.project,
            "organization": self.langfuse.organization,
            "environment": self.langfuse.environment,
            "tags": self.langfuse.tags,
            "enabled": self.langfuse.enabled
        }
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration completeness"""
        validation_results = {
            "langfuse_configured": bool(self.langfuse.secret_key and self.langfuse.public_key),
            "huggingface_configured": bool(self.apis.huggingface_api_key),
            "models_available": len(self.models.general_research_models) > 0,
            "research_settings_valid": (
                self.research.max_search_results > 0 and
                self.research.quality_score_minimum > 0 and
                self.research.fact_check_threshold > 0
            ),
            "performance_settings_valid": (
                self.performance.max_research_timeout_seconds > 0 and
                self.performance.search_result_limit > 0
            )
        }
        return validation_results

# Global configuration instance
config = ResearchAgentConfig()

# Validate configuration on import
validation_results = config.validate_configuration()
if not all(validation_results.values()):
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Configuration validation issues: {validation_results}")

# Export configuration for easy import
__all__ = ["config", "ResearchAgentConfig", "LangfuseConfig", "ModelConfig", "ResearchConfig"]