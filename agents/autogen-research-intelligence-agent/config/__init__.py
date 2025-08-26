"""
Configuration for ARIA
Autogen and LLM configuration management
"""

from .autogen_config import (
    get_autogen_config,
    get_research_assistant_config,
    get_user_proxy_config,
    create_model_config_file,
    get_research_templates,
    get_conversation_starters
)

from .llm_clients import (
    SimpleLLMClient,
    create_llm_client,
    get_available_providers
)

__all__ = [
    "get_autogen_config",
    "get_research_assistant_config",
    "get_user_proxy_config", 
    "create_model_config_file",
    "get_research_templates",
    "get_conversation_starters",
    "SimpleLLMClient",
    "create_llm_client",
    "get_available_providers"
]