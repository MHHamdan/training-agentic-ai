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
    
@dataclass
class AgentConfig:
    agent_id: str = "agent-12"
    agent_name: str = "Resume Screening Agent"
    version: str = "1.0.0"
    
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    langchain_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    cerebras_api_key: str = os.getenv("CEREBRAS_API_KEY", "")
    
    langchain_project: str = os.getenv("LANGCHAIN_PROJECT", "resume-screening-agent-v2")
    langchain_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
    langchain_endpoint: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    
    default_model: str = os.getenv("DEFAULT_MODEL", "microsoft/phi-3-mini-4k-instruct")
    enable_model_comparison: bool = os.getenv("ENABLE_MODEL_COMPARISON", "true").lower() == "true"
    max_concurrent_models: int = int(os.getenv("MAX_CONCURRENT_MODELS", "3"))
    
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    processing_timeout_seconds: int = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "300"))
    vector_store_batch_size: int = int(os.getenv("VECTOR_STORE_BATCH_SIZE", "100"))
    
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Provider preference settings
    use_cloud_apis: bool = os.getenv("USE_CLOUD_APIS", "false").lower() == "true"
    prefer_free_models: bool = os.getenv("PREFER_FREE_MODELS", "true").lower() == "true"
    
    @property
    def available_models(self) -> Dict[str, List[ModelConfig]]:
        models = {}
        
        # Default: Free Hugging Face models (latest and most powerful available)
        if not self.use_cloud_apis or self.prefer_free_models:
            models = {
                "openai_alternatives": [
                    ModelConfig("microsoft/Phi-3.5-mini-instruct", "huggingface"),  # Latest Phi model
                    ModelConfig("Qwen/Qwen2.5-7B-Instruct", "huggingface"),  # Latest Qwen
                    ModelConfig("HuggingFaceH4/zephyr-7b-beta", "huggingface"),  # Strong alternative
                ],
                "anthropic_alternatives": [
                    ModelConfig("meta-llama/Llama-3.2-3B-Instruct", "huggingface"),  # Latest Llama
                    ModelConfig("mistralai/Mistral-7B-Instruct-v0.3", "huggingface"),  # Latest Mistral
                    ModelConfig("tiiuae/falcon-7b-instruct", "huggingface"),  # Falcon model
                ],
                "gemini_alternatives": [
                    ModelConfig("google/gemma-2-9b-it", "huggingface"),  # Latest Gemma 2
                    ModelConfig("google/gemma-2-2b-it", "huggingface"),  # Smaller Gemma 2
                    ModelConfig("google/flan-t5-xxl", "huggingface"),  # Google's T5
                ],
                "cohere_alternatives": [
                    ModelConfig("CohereForAI/c4ai-command-r-v01", "huggingface"),  # Cohere's Command R
                    ModelConfig("CohereForAI/aya-23-8B", "huggingface"),  # Cohere's Aya
                ],
                "grok_alternatives": [
                    ModelConfig("EleutherAI/gpt-neox-20b", "huggingface"),  # Large open model
                    ModelConfig("databricks/dolly-v2-12b", "huggingface"),  # Databricks model
                ],
                "mistral": [
                    ModelConfig("mistralai/Mistral-7B-Instruct-v0.3", "huggingface"),
                    ModelConfig("mistralai/Mixtral-8x7B-Instruct-v0.1", "huggingface"),  # MoE model
                    ModelConfig("mistralai/Mistral-7B-Instruct-v0.2", "huggingface"),
                ],
                "deepseek": [
                    ModelConfig("deepseek-ai/deepseek-coder-7b-instruct-v1.5", "huggingface"),
                    ModelConfig("deepseek-ai/deepseek-llm-7b-chat", "huggingface"),
                    ModelConfig("deepseek-ai/deepseek-math-7b-instruct", "huggingface"),
                ],
                "qwen": [
                    ModelConfig("Qwen/Qwen2.5-7B-Instruct", "huggingface"),  # Latest Qwen 2.5
                    ModelConfig("Qwen/Qwen2.5-Coder-7B-Instruct", "huggingface"),  # Code focused
                    ModelConfig("Qwen/Qwen2.5-Math-7B-Instruct", "huggingface"),  # Math focused
                ],
                "grok_xai": [
                    ModelConfig("stabilityai/stablelm-2-12b-chat", "huggingface"),  # Stability AI
                    ModelConfig("togethercomputer/Llama-2-7B-32K-Instruct", "huggingface"),  # Extended context
                ],
                "cerebras": [
                    ModelConfig("cerebras/Cerebras-GPT-13B", "huggingface"),  # Cerebras' largest
                    ModelConfig("cerebras/Cerebras-GPT-6.7B", "huggingface"),
                    ModelConfig("cerebras/Cerebras-GPT-2.7B", "huggingface"),
                ],
                "ollama_compatible": [
                    ModelConfig("TheBloke/Llama-2-7B-Chat-GGUF", "huggingface"),  # Ollama compatible
                    ModelConfig("TheBloke/CodeLlama-7B-Instruct-GGUF", "huggingface"),
                    ModelConfig("TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "huggingface"),
                ],
            }
        
        # Option 2: Use cloud provider APIs if keys are available (check both config and env vars)
        cloud_models = {}
        
        # Check for API keys in both config and environment (for dynamic updates)
        openai_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        anthropic_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        google_key = self.google_api_key or os.getenv("GOOGLE_API_KEY")
        cohere_key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        groq_key = self.groq_api_key or os.getenv("GROQ_API_KEY")
        mistral_key = self.mistral_api_key or os.getenv("MISTRAL_API_KEY")
        deepseek_key = self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        xai_key = self.xai_api_key or os.getenv("XAI_API_KEY")
        cerebras_key = self.cerebras_api_key or os.getenv("CEREBRAS_API_KEY")
        
        if openai_key:
            cloud_models["openai"] = [
                ModelConfig("gpt-4-turbo-preview", "openai"),
                ModelConfig("gpt-4", "openai"),
                ModelConfig("gpt-3.5-turbo", "openai"),
            ]
        
        if anthropic_key:
            cloud_models["anthropic"] = [
                ModelConfig("claude-3-opus-20240229", "anthropic"),
                ModelConfig("claude-3-sonnet-20240229", "anthropic"),
                ModelConfig("claude-3-haiku-20240307", "anthropic"),
            ]
        
        if google_key:
            cloud_models["gemini"] = [
                ModelConfig("gemini-1.5-pro", "google"),
                ModelConfig("gemini-1.5-flash", "google"),
                ModelConfig("gemini-pro", "google"),
            ]
        
        if cohere_key:
            cloud_models["cohere"] = [
                ModelConfig("command-r-plus", "cohere"),
                ModelConfig("command-r", "cohere"),
                ModelConfig("command", "cohere"),
            ]
        
        if groq_key:
            cloud_models["groq"] = [
                ModelConfig("llama3-70b-8192", "groq"),
                ModelConfig("mixtral-8x7b-32768", "groq"),
                ModelConfig("gemma-7b-it", "groq"),
            ]
        
        if mistral_key:
            cloud_models["mistral_cloud"] = [
                ModelConfig("mistral-large-latest", "mistral"),
                ModelConfig("mistral-medium-latest", "mistral"),
                ModelConfig("mistral-small-latest", "mistral"),
            ]
        
        if deepseek_key:
            cloud_models["deepseek_cloud"] = [
                ModelConfig("deepseek-chat", "deepseek"),
                ModelConfig("deepseek-coder", "deepseek"),
            ]
        
        if xai_key:
            cloud_models["xai"] = [
                ModelConfig("grok-1", "xai"),
                ModelConfig("grok-2", "xai"),
            ]
        
        if cerebras_key:
            cloud_models["cerebras_cloud"] = [
                ModelConfig("cerebras-llama3.1-8b", "cerebras"),
                ModelConfig("cerebras-llama3.1-70b", "cerebras"),
            ]
        
        # Merge cloud models with free models if both are available
        if cloud_models:
            models.update(cloud_models)
        
        return models if models else self._get_fallback_models()
    
    def _get_fallback_models(self) -> Dict[str, List[ModelConfig]]:
        """Fallback to basic free models if no configuration is available"""
        return {
            "free_models": [
                ModelConfig("microsoft/phi-3-mini-4k-instruct", "huggingface"),
                ModelConfig("Qwen/Qwen2.5-7B-Instruct", "huggingface"),
                ModelConfig("mistralai/Mistral-7B-Instruct-v0.3", "huggingface"),
            ]
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        for category, models in self.available_models.items():
            for model in models:
                if model.name == model_name:
                    return model
        return None
    
config = AgentConfig()