import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    name: str
    provider: str
    endpoint: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    financial_optimized: bool = False
    
@dataclass 
class AgentOpsConfig:
    api_key: str
    project_name: str = "stock-analysis-agent-v2"
    environment: str = "production"
    tags: List[str] = None
    enable_logging: bool = True
    enable_compliance_tracking: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["financial", "compliance", "multi-agent", "production"]

@dataclass
class FinancialConfig:
    # API Keys for Financial Data
    alpha_vantage_api_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    
    # Analysis Configuration
    max_analysis_timeout: int = 600
    news_search_limit: int = 50
    technical_indicator_periods: List[int] = None
    risk_assessment_depth: str = "comprehensive"
    
    # Compliance Settings
    enable_audit_logging: bool = True
    regulatory_compliance_level: str = "SEC"
    risk_disclosure_required: bool = True
    
    def __post_init__(self):
        if self.technical_indicator_periods is None:
            self.technical_indicator_periods = [14, 30, 50, 200]

@dataclass
class StockAnalysisAgentConfig:
    # Agent Settings
    agent_id: str = "agent-13"
    agent_name: str = "Stock Analysis Agent"
    version: str = "2.0.0"
    port: int = 8513
    
    # Core API Keys
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    agentops_api_key: str = os.getenv("AGENTOPS_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model Configuration
    default_model: str = os.getenv("DEFAULT_MODEL", "microsoft/phi-3-mini-4k-instruct")
    enable_model_comparison: bool = os.getenv("ENABLE_MODEL_COMPARISON", "true").lower() == "true"
    max_concurrent_models: int = int(os.getenv("MAX_CONCURRENT_MODELS", "3"))
    financial_model_preference: str = os.getenv("FINANCIAL_MODEL_PREFERENCE", "finbert")
    
    # Performance Settings
    max_concurrent_analyses: int = 5
    memory_limit_gb: int = 3
    response_timeout: int = 60
    
    # AgentOps Configuration
    agentops: AgentOpsConfig = None
    
    # Financial Configuration  
    financial: FinancialConfig = None
    
    def __post_init__(self):
        if self.agentops is None:
            self.agentops = AgentOpsConfig(
                api_key=self.agentops_api_key,
                project_name="stock-analysis-agent-v2",
                environment=os.getenv("AGENTOPS_ENVIRONMENT", "production"),
                tags=os.getenv("AGENTOPS_TAGS", "financial,compliance,multi-agent").split(",")
            )
        
        if self.financial is None:
            self.financial = FinancialConfig()
    
    @property
    def available_models(self) -> Dict[str, List[ModelConfig]]:
        """Available financial models organized by category"""
        return {
            "financial_reasoning": [
                ModelConfig("microsoft/DialoGPT-medium-finance", "huggingface", financial_optimized=True),
                ModelConfig("EleutherAI/gpt-j-6b", "huggingface", max_tokens=2048),
                ModelConfig("microsoft/phi-3-mini-4k-instruct", "huggingface"),
                ModelConfig("Qwen/Qwen2.5-7B-Instruct", "huggingface"),
                ModelConfig("meta-llama/Llama-2-7b-chat-hf", "huggingface", max_tokens=2048),
            ],
            
            "sentiment_analysis": [
                ModelConfig("cardiffnlp/twitter-roberta-base-sentiment-latest", "huggingface", financial_optimized=True),
                ModelConfig("ProsusAI/finbert", "huggingface", financial_optimized=True),
                ModelConfig("nlptown/bert-base-multilingual-uncased-sentiment", "huggingface"),
                ModelConfig("yiyanghkust/finbert-tone", "huggingface", financial_optimized=True),
            ],
            
            "text_generation": [
                ModelConfig("mistralai/Mistral-7B-Instruct-v0.1", "huggingface"),
                ModelConfig("google/gemma-2b-it", "huggingface"),
                ModelConfig("microsoft/phi-3-mini-128k-instruct", "huggingface"),
                ModelConfig("Qwen/Qwen2.5-Coder-7B-Instruct", "huggingface"),
            ],
            
            "technical_analysis": [
                ModelConfig("microsoft/phi-3-mini-4k-instruct", "huggingface", financial_optimized=True),
                ModelConfig("Qwen/Qwen2.5-Math-7B-Instruct", "huggingface", financial_optimized=True),
                ModelConfig("deepseek-ai/deepseek-math-7b-instruct", "huggingface"),
            ]
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        for category, models in self.available_models.items():
            for model in models:
                if model.name == model_name:
                    return model
        return None
    
    @property
    def financial_models(self) -> List[str]:
        """Get list of financially optimized models"""
        models = []
        for category, model_list in self.available_models.items():
            for model in model_list:
                if model.financial_optimized:
                    models.append(model.name)
        return models
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        required_keys = ["huggingface_api_key", "agentops_api_key"]
        missing = [key for key in required_keys if not getattr(self, key)]
        
        if missing:
            raise ValueError(f"Missing required configuration: {missing}")
        
        return True

# Global configuration instance
config = StockAnalysisAgentConfig()