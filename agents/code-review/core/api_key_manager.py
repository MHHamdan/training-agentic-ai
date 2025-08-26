"""
Multi-Provider API Key Management System
Prioritizes free HuggingFace models with optional user-provided paid API keys
Author: Mohammed Hamdan
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import asyncio
import aiohttp

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config, ProviderConfig

logger = logging.getLogger(__name__)

@dataclass
class APIKeyValidationResult:
    """Result of API key validation"""
    provider: str
    is_valid: bool
    error_message: str = ""
    models_available: List[str] = None
    rate_limit_info: Dict[str, Any] = None

class APIKeyManager:
    """
    Secure management of API keys with priority on free HuggingFace models
    - Always uses system HuggingFace API key from .env
    - Optionally accepts user-provided API keys for paid providers
    - Validates and manages all provider connections
    """
    
    def __init__(self):
        """Initialize API key manager"""
        self.config = config
        self.system_api_keys = self._load_system_keys()
        self.user_api_keys = {}
        self.validated_providers = {}
        self.provider_status = {}
        
        logger.info("API Key Manager initialized with HuggingFace as primary provider")
    
    def _load_system_keys(self) -> Dict[str, str]:
        """Load system API keys from environment"""
        system_keys = {}
        
        # Always load HuggingFace key from .env (required)
        hf_key = os.getenv("HUGGINGFACE_API_KEY", "")
        if hf_key:
            system_keys["huggingface"] = hf_key
            logger.info("✅ System HuggingFace API key loaded")
        else:
            logger.warning("⚠️ HuggingFace API key not found in .env file")
        
        # Load optional system keys as fallbacks
        optional_keys = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY", 
            "google": "GOOGLE_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY"
        }
        
        for provider, env_key in optional_keys.items():
            api_key = os.getenv(env_key, "")
            if api_key:
                system_keys[provider] = api_key
                logger.info(f"✅ System {provider} API key loaded as fallback")
        
        return system_keys
    
    @observe(as_type="api_key_validation")
    async def validate_api_key(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """
        Validate API key for specific provider
        
        Args:
            provider: Provider name (openai, anthropic, google, etc.)
            api_key: API key to validate
            
        Returns:
            APIKeyValidationResult with validation status
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Validating API key for provider: {provider}",
                    metadata={
                        "provider": provider,
                        "organization": "code-review-org",
                        "project": "code-review-agent-v2"
                    }
                )
            
            # Basic format validation first
            if not self._validate_key_format(provider, api_key):
                return APIKeyValidationResult(
                    provider=provider,
                    is_valid=False,
                    error_message=f"Invalid {provider} API key format"
                )
            
            # Provider-specific validation
            validation_result = await self._validate_provider_key(provider, api_key)
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Validation result: {validation_result.is_valid}",
                    metadata={
                        "validation_success": validation_result.is_valid,
                        "provider": provider,
                        "models_count": len(validation_result.models_available or [])
                    }
                )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"API key validation error for {provider}: {e}")
            return APIKeyValidationResult(
                provider=provider,
                is_valid=False,
                error_message=str(e)
            )
    
    def _validate_key_format(self, provider: str, api_key: str) -> bool:
        """Validate API key format for each provider"""
        format_patterns = {
            "openai": r"^sk-[a-zA-Z0-9]{32,}$",
            "anthropic": r"^sk-ant-[a-zA-Z0-9\-_]{95,}$",
            "google": r"^AIza[0-9A-Za-z\-_]{35}$",
            "groq": r"^gsk_[a-zA-Z0-9]{52}$",
            "mistral": r"^[a-zA-Z0-9]{32}$",
            "huggingface": r"^hf_[a-zA-Z0-9]{34,}$"
        }
        
        pattern = format_patterns.get(provider)
        if not pattern:
            return True  # Unknown provider, skip format validation
        
        return bool(re.match(pattern, api_key))
    
    async def _validate_provider_key(self, provider: str, api_key: str) -> APIKeyValidationResult:
        """Validate API key by making test API call"""
        try:
            if provider == "openai":
                return await self._validate_openai_key(api_key)
            elif provider == "anthropic":
                return await self._validate_anthropic_key(api_key)
            elif provider == "google":
                return await self._validate_google_key(api_key)
            elif provider == "groq":
                return await self._validate_groq_key(api_key)
            elif provider == "mistral":
                return await self._validate_mistral_key(api_key)
            elif provider == "huggingface":
                return await self._validate_huggingface_key(api_key)
            else:
                return APIKeyValidationResult(
                    provider=provider,
                    is_valid=False,
                    error_message=f"Unknown provider: {provider}"
                )
        except Exception as e:
            return APIKeyValidationResult(
                provider=provider,
                is_valid=False,
                error_message=f"Validation failed: {str(e)}"
            )
    
    async def _validate_openai_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate OpenAI API key"""
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['id'] for model in data.get('data', [])]
                    return APIKeyValidationResult(
                        provider="openai",
                        is_valid=True,
                        models_available=models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="openai",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    async def _validate_anthropic_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate Anthropic API key"""
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}]
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status in [200, 400]:  # 400 is fine for validation
                    return APIKeyValidationResult(
                        provider="anthropic",
                        is_valid=True,
                        models_available=config.SUPPORTED_PROVIDERS["anthropic"].models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="anthropic",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    async def _validate_huggingface_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate HuggingFace API key"""
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return APIKeyValidationResult(
                        provider="huggingface",
                        is_valid=True,
                        models_available=config.SUPPORTED_PROVIDERS["huggingface"].models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="huggingface",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    async def _validate_google_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate Google API key"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://generativelanguage.googleapis.com/v1/models?key={api_key}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'].split('/')[-1] for model in data.get('models', [])]
                    return APIKeyValidationResult(
                        provider="google",
                        is_valid=True,
                        models_available=models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="google",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    async def _validate_groq_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate Groq API key"""
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.groq.com/openai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['id'] for model in data.get('data', [])]
                    return APIKeyValidationResult(
                        provider="groq",
                        is_valid=True,
                        models_available=models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="groq",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    async def _validate_mistral_key(self, api_key: str) -> APIKeyValidationResult:
        """Validate Mistral API key"""
        headers = {"Authorization": f"Bearer {api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.mistral.ai/v1/models",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['id'] for model in data.get('data', [])]
                    return APIKeyValidationResult(
                        provider="mistral",
                        is_valid=True,
                        models_available=models
                    )
                else:
                    return APIKeyValidationResult(
                        provider="mistral",
                        is_valid=False,
                        error_message=f"API returned status {response.status}"
                    )
    
    def set_user_api_keys(self, user_keys: Dict[str, str]) -> Dict[str, bool]:
        """
        Set user-provided API keys and validate them
        
        Args:
            user_keys: Dictionary of provider -> API key
            
        Returns:
            Dictionary of provider -> validation success
        """
        validation_results = {}
        
        for provider, api_key in user_keys.items():
            if api_key and api_key.strip():
                self.user_api_keys[provider] = api_key.strip()
                validation_results[provider] = True
                logger.info(f"User API key set for {provider}")
            else:
                validation_results[provider] = False
        
        return validation_results
    
    def get_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available providers with their status
        Priority: User keys > System keys > HuggingFace fallback
        """
        available_providers = {}
        
        # Always include HuggingFace as primary free option (works even without API key)
        hf_key = self.system_api_keys.get("huggingface", "")
        available_providers["huggingface"] = {
            "name": "🤗 Hugging Face (Free Models)",
            "type": "free",
            "status": "available",  # Always available with fallback analysis
            "models": config.SUPPORTED_PROVIDERS["huggingface"].models,
            "source": "system",
            "priority": 1
        }
        
        # Add user-provided keys with high priority
        priority = 2
        for provider, api_key in self.user_api_keys.items():
            if api_key and provider in config.SUPPORTED_PROVIDERS:
                provider_config = config.SUPPORTED_PROVIDERS[provider]
                available_providers[provider] = {
                    "name": f"🔑 {provider_config.name} (Your API Key)",
                    "type": "paid",
                    "status": "available",
                    "models": provider_config.models,
                    "source": "user",
                    "priority": priority
                }
                priority += 1
        
        # Add system keys as fallbacks (lower priority)
        for provider, api_key in self.system_api_keys.items():
            if provider != "huggingface" and provider not in available_providers:
                if api_key and provider in config.SUPPORTED_PROVIDERS:
                    provider_config = config.SUPPORTED_PROVIDERS[provider]
                    available_providers[provider] = {
                        "name": f"🏢 {provider_config.name} (System)",
                        "type": "paid",
                        "status": "available",
                        "models": provider_config.models,
                        "source": "system",
                        "priority": priority
                    }
                    priority += 1
        
        return available_providers
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for provider with priority: User > System
        """
        # Check user-provided key first
        if provider in self.user_api_keys:
            return self.user_api_keys[provider]
        
        # Fallback to system key
        return self.system_api_keys.get(provider)
    
    def get_recommended_provider(self, task_type: str = "code_analysis") -> str:
        """
        Get recommended provider based on task and availability
        Always prioritizes HuggingFace for free usage
        """
        available = self.get_available_providers()
        
        # Always prefer HuggingFace for free usage
        if "huggingface" in available and available["huggingface"]["status"] == "available":
            return "huggingface"
        
        # If HuggingFace not available, try user keys
        user_providers = [p for p, info in available.items() if info["source"] == "user"]
        if user_providers:
            return user_providers[0]
        
        # Fallback to system keys
        system_providers = [p for p, info in available.items() if info["source"] == "system"]
        if system_providers:
            return system_providers[0]
        
        return "huggingface"  # Default fallback
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all providers"""
        available = self.get_available_providers()
        
        return {
            "total_providers": len(available),
            "free_providers": len([p for p, info in available.items() if info["type"] == "free"]),
            "paid_providers": len([p for p, info in available.items() if info["type"] == "paid"]),
            "user_providers": len([p for p, info in available.items() if info["source"] == "user"]),
            "system_providers": len([p for p, info in available.items() if info["source"] == "system"]),
            "recommended_provider": self.get_recommended_provider(),
            "providers": available
        }