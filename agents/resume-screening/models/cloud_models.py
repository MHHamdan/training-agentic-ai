import logging
import json
from typing import Optional, Dict, Any
import aiohttp
import asyncio
from langsmith import traceable

logger = logging.getLogger(__name__)

class CloudModelBase:
    """Base class for cloud provider models"""
    def __init__(
        self,
        model_name: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    async def generate(self, prompt: str) -> str:
        """Override in subclasses"""
        raise NotImplementedError
    
    async def analyze_resume(
        self,
        resume_text: str,
        job_requirements: str,
        prompt_template: str
    ) -> Dict[str, Any]:
        """Common resume analysis logic"""
        try:
            prompt = prompt_template.format(
                resume_text=resume_text[:3000],
                job_requirements=job_requirements[:1000]
            )
            
            response = await self.generate(prompt)
            
            # Parse response (same logic as HF models)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            return self._get_default_scores()
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured format"""
        try:
            # Try to parse as JSON first
            if response.strip().startswith("{"):
                return json.loads(response)
            
            # Fallback to pattern matching
            scores = {}
            insights = []
            
            lines = response.strip().split("\n")
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip().lower().replace(" ", "_")
                    value = parts[1].strip()
                    
                    if any(word in key for word in ["technical", "experience", "cultural", "growth", "risk", "overall"]):
                        try:
                            # Extract numeric value
                            import re
                            numeric_match = re.search(r'(\d+(?:\.\d+)?)', value)
                            if numeric_match:
                                scores[key] = float(numeric_match.group(1))
                        except:
                            pass
                    else:
                        insights.append(line)
            
            return {
                "scores": scores if scores else self._get_default_scores()["scores"],
                "insights": insights[:10] if insights else ["Analysis completed"],
                "raw_response": response[:500]
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Default scores when analysis fails"""
        return {
            "scores": {
                "technical_skills": 50,
                "experience_relevance": 50,
                "cultural_fit": 50,
                "growth_potential": 50,
                "risk_assessment": 50,
                "overall": 50
            },
            "insights": ["Analysis completed with default values"],
            "raw_response": "Default response"
        }

class OpenAIModel(CloudModelBase):
    @traceable(name="openai_generate", metadata={"provider": "openai"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return f"Error: {str(e)}"

class AnthropicModel(CloudModelBase):
    @traceable(name="anthropic_generate", metadata={"provider": "anthropic"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["content"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Anthropic API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Anthropic API error: {str(e)}")
                return f"Error: {str(e)}"

class GoogleModel(CloudModelBase):
    @traceable(name="google_generate", metadata={"provider": "google"})
    async def generate(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Google API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Google API error: {str(e)}")
                return f"Error: {str(e)}"

class CohereModel(CloudModelBase):
    @traceable(name="cohere_generate", metadata={"provider": "cohere"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.cohere.ai/v1/generate"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["generations"][0]["text"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Cohere API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Cohere API error: {str(e)}")
                return f"Error: {str(e)}"

class GroqModel(CloudModelBase):
    @traceable(name="groq_generate", metadata={"provider": "groq"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Groq API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Groq API error: {str(e)}")
                return f"Error: {str(e)}"

class MistralModel(CloudModelBase):
    @traceable(name="mistral_generate", metadata={"provider": "mistral"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Mistral API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Mistral API error: {str(e)}")
                return f"Error: {str(e)}"

class DeepSeekModel(CloudModelBase):
    @traceable(name="deepseek_generate", metadata={"provider": "deepseek"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"DeepSeek API error: {str(e)}")
                return f"Error: {str(e)}"

class XAIModel(CloudModelBase):
    @traceable(name="xai_generate", metadata={"provider": "xai"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"XAI API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"XAI API error: {str(e)}")
                return f"Error: {str(e)}"

class CerebrasModel(CloudModelBase):
    @traceable(name="cerebras_generate", metadata={"provider": "cerebras"})
    async def generate(self, prompt: str) -> str:
        url = "https://api.cerebras.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"Cerebras API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except Exception as e:
                logger.error(f"Cerebras API error: {str(e)}")
                return f"Error: {str(e)}"