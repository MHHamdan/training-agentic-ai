"""
Configuration Management for Code Review Agent
Handles environment variables, API keys, and agent settings
"""

import os
from typing import Dict, Any, List, Optional, ClassVar
from pydantic import Field
from pydantic_settings import BaseSettings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProviderConfig:
    """Configuration for each provider"""
    name: str
    models: List[str]
    env_key: str
    required: bool = False
    timeout: int = 60
    max_retries: int = 3

class CodeReviewConfig(BaseSettings):
    """Main configuration class for Code Review Agent"""
    
    # Basic Configuration
    agent_name: str = "Code Review Agent V1"
    agent_version: str = "1.0.0"
    port: int = 8515
    log_level: str = "INFO"
    
    # Observability Configuration
    class ObservabilityConfig:
        # LangSmith Configuration
        langsmith_enabled: bool = Field(default=True)
        langsmith_api_key: str = Field(default_factory=lambda: os.getenv("LANGCHAIN_API_KEY", ""))
        langsmith_project: str = Field(default="code-review-agent-v2")
        langsmith_organization: str = Field(default="code-review-org")
        
        # AgentOps Configuration
        agentops_enabled: bool = Field(default=True)
        agentops_api_key: str = Field(default_factory=lambda: os.getenv("AGENTOPS_API_KEY", ""))
        agentops_project: str = Field(default="code-review-monitoring")
        
        # General Observability
        enable_tracing: bool = Field(default=True)
        log_code_snippets: bool = Field(default=False)
        mask_sensitive_data: bool = Field(default=True)
    
    # API Keys Configuration
    class APIKeys:
        # System API Keys (Pre-configured)
        huggingface_api_key: str = Field(default_factory=lambda: os.getenv("HUGGINGFACE_API_KEY", ""))
        
        # Optional System Provider Keys (Fallback)
        openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
        anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
        google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
        groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
        mistral_api_key: str = Field(default_factory=lambda: os.getenv("MISTRAL_API_KEY", ""))
    
    # Analysis Configuration
    class AnalysisConfig:
        max_code_length: int = Field(default=10000)
        analysis_timeout_seconds: int = Field(default=300)
        enable_security_scan: bool = Field(default=True)
        enable_performance_analysis: bool = Field(default=True)
        enable_style_check: bool = Field(default=True)
        enable_complexity_analysis: bool = Field(default=True)
        
        # Quality thresholds
        min_security_score: float = Field(default=7.0)
        min_performance_score: float = Field(default=6.0)
        min_style_score: float = Field(default=8.0)
    
    # Model Configuration  
    class ModelConfig:
        # Default to free HuggingFace models
        default_hf_model: str = Field(default="microsoft/CodeBERT-base")
        enable_model_comparison: bool = Field(default=True)
        max_concurrent_providers: int = Field(default=3)
        provider_timeout_seconds: int = Field(default=60)
        
        # Prioritize free models - HuggingFace first
        preferred_providers: List[str] = Field(default=["huggingface"])
        fallback_model: str = Field(default="microsoft/CodeBERT-base")
        
        # Free HuggingFace models for code analysis
        free_code_models: Dict[str, str] = Field(default={
            "code_analysis": "microsoft/CodeBERT-base",
            "code_generation": "Salesforce/codet5-base", 
            "security_analysis": "microsoft/GraphCodeBERT-base",
            "performance_analysis": "microsoft/unixcoder-base",
            "style_analysis": "huggingface/CodeBERTa-small-v1",
            "general_coding": "bigcode/starcoder2-3b",
            "code_review": "microsoft/DialoGPT-medium"
        })
    
    # Security Configuration
    class SecurityConfig:
        enable_code_sanitization: bool = Field(default=True)
        log_code_snippets: bool = Field(default=False)
        mask_sensitive_data: bool = Field(default=True)
        max_file_size_mb: int = Field(default=5)
        
        # Security scan settings
        detect_hardcoded_secrets: bool = Field(default=True)
        detect_sql_injection: bool = Field(default=True)
        detect_xss_vulnerabilities: bool = Field(default=True)
        detect_command_injection: bool = Field(default=True)
    
    # Initialize sub-configurations
    observability: ObservabilityConfig = ObservabilityConfig()
    api_keys: APIKeys = APIKeys()
    analysis: AnalysisConfig = AnalysisConfig()
    models: ModelConfig = ModelConfig()
    security: SecurityConfig = SecurityConfig()
    
    # Provider Configurations
    SUPPORTED_PROVIDERS: ClassVar[Dict[str, ProviderConfig]] = {
        'openai': ProviderConfig(
            name='OpenAI',
            models=['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o'],
            env_key='OPENAI_API_KEY',
            required=False,
            timeout=60,
            max_retries=3
        ),
        'anthropic': ProviderConfig(
            name='Anthropic Claude',
            models=['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            env_key='ANTHROPIC_API_KEY',
            required=False,
            timeout=90,
            max_retries=3
        ),
        'google': ProviderConfig(
            name='Google Gemini',
            models=['gemini-pro', 'gemini-pro-vision', 'gemini-1.5-pro'],
            env_key='GOOGLE_API_KEY',
            required=False,
            timeout=60,
            max_retries=3
        ),
        'groq': ProviderConfig(
            name='Groq',
            models=['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
            env_key='GROQ_API_KEY',
            required=False,
            timeout=30,
            max_retries=2
        ),
        'mistral': ProviderConfig(
            name='Mistral AI',
            models=['mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest'],
            env_key='MISTRAL_API_KEY',
            required=False,
            timeout=60,
            max_retries=3
        ),
        'huggingface': ProviderConfig(
            name='Hugging Face (Free Models)',
            models=[
                'microsoft/CodeBERT-base',
                'Salesforce/codet5-base', 
                'microsoft/GraphCodeBERT-base',
                'microsoft/unixcoder-base',
                'huggingface/CodeBERTa-small-v1',
                'bigcode/starcoder2-3b',
                'microsoft/DialoGPT-medium',
                'Salesforce/codet5-small',
                'microsoft/codebert-base-mlm'
            ],
            env_key='HUGGINGFACE_API_KEY',
            required=True,  # Always available as primary option
            timeout=90,
            max_retries=2
        )
    }
    
    # Code Analysis Categories
    ANALYSIS_CATEGORIES: ClassVar[Dict[str, Dict[str, Any]]] = {
        'security': {
            'name': 'Security Analysis',
            'description': 'Vulnerability detection and security best practices',
            'weight': 0.30,
            'checks': [
                'sql_injection',
                'xss_vulnerabilities', 
                'command_injection',
                'hardcoded_secrets',
                'insecure_crypto',
                'input_validation'
            ]
        },
        'performance': {
            'name': 'Performance Analysis',
            'description': 'Code efficiency and optimization opportunities',
            'weight': 0.25,
            'checks': [
                'time_complexity',
                'memory_usage',
                'loop_efficiency',
                'database_optimization',
                'algorithmic_improvements'
            ]
        },
        'style': {
            'name': 'Style Analysis',
            'description': 'Code style and formatting compliance',
            'weight': 0.20,
            'checks': [
                'pep8_compliance',
                'naming_conventions',
                'code_formatting',
                'import_organization',
                'line_length'
            ]
        },
        'maintainability': {
            'name': 'Maintainability',
            'description': 'Code readability and maintainability metrics',
            'weight': 0.15,
            'checks': [
                'code_complexity',
                'function_length',
                'class_design',
                'coupling_cohesion',
                'code_duplication'
            ]
        },
        'documentation': {
            'name': 'Documentation',
            'description': 'Code documentation and comments quality',
            'weight': 0.10,
            'checks': [
                'docstring_coverage',
                'comment_quality',
                'type_annotations',
                'readme_completeness',
                'api_documentation'
            ]
        }
    }
    
    model_config = {"env_file": ".env", "extra": "allow"}
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate configuration and return status"""
        validation_results = {}
        
        # Check observability configuration
        validation_results["langsmith_configured"] = bool(self.observability.langsmith_api_key)
        validation_results["agentops_configured"] = bool(self.observability.agentops_api_key)
        
        # Check API keys
        validation_results["huggingface_key_available"] = bool(self.api_keys.huggingface_api_key)
        validation_results["openai_key_available"] = bool(self.api_keys.openai_api_key)
        validation_results["anthropic_key_available"] = bool(self.api_keys.anthropic_api_key)
        
        # Check required settings
        validation_results["analysis_enabled"] = (
            self.analysis.enable_security_scan or 
            self.analysis.enable_performance_analysis or 
            self.analysis.enable_style_check
        )
        
        validation_results["security_configured"] = self.security.enable_code_sanitization
        
        return validation_results
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys"""
        available = []
        
        for provider_id, provider_config in self.SUPPORTED_PROVIDERS.items():
            api_key = getattr(self.api_keys, f"{provider_id}_api_key", "")
            if api_key or provider_id == "huggingface":  # HF always available
                available.append(provider_id)
        
        return available
    
    def get_provider_config(self, provider_id: str) -> Optional[ProviderConfig]:
        """Get configuration for specific provider"""
        return self.SUPPORTED_PROVIDERS.get(provider_id)
    
    def get_analysis_weights(self) -> Dict[str, float]:
        """Get analysis category weights for scoring"""
        return {
            category: config['weight'] 
            for category, config in self.ANALYSIS_CATEGORIES.items()
        }

# Global configuration instance
config = CodeReviewConfig()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info(f"Code Review Agent V{config.agent_version} configuration loaded")
logger.info(f"Available providers: {config.get_available_providers()}")