"""
Generic API-based OCR System
Supports multiple AI providers for accurate text extraction
Author: Mohammed Hamdan
"""

import os
import logging
import base64
import time
import asyncio
from typing import Dict, Any, Optional, List
from PIL import Image
import io
import aiohttp
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for API providers"""
    provider: str
    api_key: str
    base_url: str
    model_name: str
    max_tokens: int = 4000
    temperature: float = 0.1

class GenericAPIProcessor:
    """
    Generic API processor that works with multiple AI providers
    Supports: OpenAI, Anthropic, Google, DeepSeek, Groq, and more
    """
    
    SUPPORTED_PROVIDERS = {
        "openai": {
            "base_url": "https://api.openai.com/v1/chat/completions",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"],
            "env_key": "OPENAI_API_KEY"
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1/messages",
            "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
            "env_key": "ANTHROPIC_API_KEY"
        },
        "google": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro-vision"],
            "env_key": "GOOGLE_API_KEY"
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1/chat/completions",
            "models": ["deepseek-chat", "deepseek-coder"],
            "env_key": "DEEPSEEK_API_KEY"
        },
        "groq": {
            "base_url": "https://api.groq.com/openai/v1/chat/completions",
            "models": ["llama-3.2-90b-vision-preview", "llama-3.2-11b-vision-preview"],
            "env_key": "GROQ_API_KEY"
        },
        "together": {
            "base_url": "https://api.together.xyz/v1/chat/completions",
            "models": ["meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"],
            "env_key": "TOGETHER_API_KEY"
        }
    }
    
    def __init__(self):
        """Initialize API processor with auto-detection of available providers"""
        self.available_configs = self._detect_available_providers()
        self.primary_config = self._select_primary_provider()
        logger.info(f"🚀 API OCR initialized with {len(self.available_configs)} providers")
        if self.primary_config:
            logger.info(f"🎯 Primary provider: {self.primary_config.provider} ({self.primary_config.model_name})")
    
    def _detect_available_providers(self) -> List[APIConfig]:
        """Detect available API providers from environment variables"""
        configs = []
        
        for provider, info in self.SUPPORTED_PROVIDERS.items():
            api_key = os.getenv(info["env_key"])
            if api_key and api_key.strip() and api_key != "your-api-key-here":
                # Select best model for each provider
                model = info["models"][0]  # Use first (usually best) model
                
                config = APIConfig(
                    provider=provider,
                    api_key=api_key,
                    base_url=info["base_url"],
                    model_name=model
                )
                configs.append(config)
                logger.info(f"✅ {provider.upper()} API available ({model})")
        
        return configs
    
    def _select_primary_provider(self) -> Optional[APIConfig]:
        """Select the best available provider based on capability ranking"""
        if not self.available_configs:
            return None
        
        # Ranking preference (best to worst for OCR tasks)
        preference_order = ["openai", "anthropic", "google", "groq", "together", "deepseek"]
        
        for preferred in preference_order:
            for config in self.available_configs:
                if config.provider == preferred:
                    return config
        
        # If no preferred provider found, use first available
        return self.available_configs[0]
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_string = base64.b64encode(image_bytes).decode('utf-8')
            return base64_string
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise
    
    async def extract_text_with_api(self, image: Image.Image, language_hint: str = "auto") -> Dict[str, Any]:
        """
        Extract text from image using the best available API provider
        """
        if not self.primary_config:
            raise ValueError("No API providers available. Please configure API keys in environment variables.")
        
        start_time = time.time()
        
        try:
            # Encode image
            base64_image = self._image_to_base64(image)
            
            # Create optimized prompt for OCR
            ocr_prompt = self._create_ocr_prompt(language_hint)
            
            # Try primary provider first
            try:
                result = await self._call_api(self.primary_config, base64_image, ocr_prompt)
                processing_time = time.time() - start_time
                
                return {
                    'text': result.get('text', ''),
                    'confidence': result.get('confidence', 0.9),
                    'method': f"api_{self.primary_config.provider}",
                    'model': self.primary_config.model_name,
                    'processing_time': processing_time,
                    'language_detected': result.get('language', language_hint),
                    'provider': self.primary_config.provider
                }
                
            except Exception as e:
                logger.warning(f"Primary provider {self.primary_config.provider} failed: {e}")
                
                # Try fallback providers
                for config in self.available_configs[1:]:
                    try:
                        logger.info(f"🔄 Trying fallback provider: {config.provider}")
                        result = await self._call_api(config, base64_image, ocr_prompt)
                        processing_time = time.time() - start_time
                        
                        return {
                            'text': result.get('text', ''),
                            'confidence': result.get('confidence', 0.85),
                            'method': f"api_{config.provider}_fallback",
                            'model': config.model_name,
                            'processing_time': processing_time,
                            'language_detected': result.get('language', language_hint),
                            'provider': config.provider
                        }
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback provider {config.provider} failed: {fallback_error}")
                        continue
                
                # All providers failed
                raise Exception("All API providers failed")
                
        except Exception as e:
            logger.error(f"API OCR failed: {e}")
            processing_time = time.time() - start_time
            return {
                'text': f'API OCR Error: {str(e)}',
                'confidence': 0.0,
                'method': 'api_error',
                'model': 'none',
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def _create_ocr_prompt(self, language_hint: str) -> str:
        """Create optimized prompt for OCR task"""
        base_prompt = '''You are an expert OCR system. Extract ALL text from this image with perfect accuracy.

INSTRUCTIONS:
1. Extract every single word, number, and symbol visible in the image
2. Maintain original formatting, spacing, and line breaks
3. Preserve the exact structure and layout of the text
4. Include punctuation marks exactly as shown
5. Do not add, interpret, or modify any content
6. If text is unclear, make your best guess but stay faithful to what you see
7. Return ONLY the extracted text, no explanations or comments'''

        if language_hint and language_hint != "auto":
            if language_hint == "arabic":
                base_prompt += "\n8. The document contains Arabic text - extract Arabic characters precisely"
            elif language_hint == "english":
                base_prompt += "\n8. The document contains English text - extract Latin characters precisely"
            elif language_hint == "mixed":
                base_prompt += "\n8. The document contains mixed languages - extract all scripts accurately"
        
        return base_prompt
    
    async def _call_api(self, config: APIConfig, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call specific API provider"""
        if config.provider == "openai":
            return await self._call_openai(config, base64_image, prompt)
        elif config.provider == "anthropic":
            return await self._call_anthropic(config, base64_image, prompt)
        elif config.provider == "google":
            return await self._call_google(config, base64_image, prompt)
        elif config.provider in ["deepseek", "groq", "together"]:
            return await self._call_openai_compatible(config, base64_image, prompt)
        else:
            raise ValueError(f"Provider {config.provider} not implemented")
    
    async def _call_openai(self, config: APIConfig, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['choices'][0]['message']['content'].strip()
                    return {'text': text, 'confidence': 0.9}
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error {response.status}: {error_text}")
    
    async def _call_anthropic(self, config: APIConfig, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['content'][0]['text'].strip()
                    return {'text': text, 'confidence': 0.95}
                else:
                    error_text = await response.text()
                    raise Exception(f"Anthropic API error {response.status}: {error_text}")
    
    async def _call_google(self, config: APIConfig, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call Google Gemini API"""
        url = config.base_url.format(model=config.model_name) + f"?key={config.api_key}"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "temperature": config.temperature
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                    return {'text': text, 'confidence': 0.9}
                else:
                    error_text = await response.text()
                    raise Exception(f"Google API error {response.status}: {error_text}")
    
    async def _call_openai_compatible(self, config: APIConfig, base64_image: str, prompt: str) -> Dict[str, Any]:
        """Call OpenAI-compatible APIs (DeepSeek, Groq, Together, etc.)"""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    text = data['choices'][0]['message']['content'].strip()
                    return {'text': text, 'confidence': 0.85}
                else:
                    error_text = await response.text()
                    raise Exception(f"{config.provider.upper()} API error {response.status}: {error_text}")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available API providers"""
        return [config.provider for config in self.available_configs]
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get detailed information about available providers"""
        return {
            "primary_provider": self.primary_config.provider if self.primary_config else None,
            "available_providers": [
                {
                    "provider": config.provider,
                    "model": config.model_name,
                    "status": "available"
                }
                for config in self.available_configs
            ],
            "total_providers": len(self.available_configs),
            "supported_providers": list(self.SUPPORTED_PROVIDERS.keys())
        }


# Global instance
api_ocr_processor = None

def get_api_ocr() -> GenericAPIProcessor:
    """Get or create API OCR processor singleton"""
    global api_ocr_processor
    if api_ocr_processor is None:
        api_ocr_processor = GenericAPIProcessor()
    return api_ocr_processor