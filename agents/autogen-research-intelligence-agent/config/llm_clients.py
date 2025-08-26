"""
LLM Client implementations for ARIA
Provides LLM integration for research tasks
"""

import os
from typing import Dict, List, Any, Optional
import json

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False


class SimpleLLMClient:
    """
    Simple LLM client for research tasks
    """
    
    def __init__(self, provider: str = "google", api_key: str = None, model: str = None):
        """
        Initialize LLM client
        
        Args:
            provider: LLM provider (google, openai, anthropic)
            api_key: API key for the provider
            model: Model name to use
        """
        self.provider = provider
        self.api_key = api_key or self._get_default_api_key(provider)
        self.model = model or self._get_default_model(provider)
        self.client = None
        
        if self.api_key:
            self._initialize_client()
    
    def _get_default_api_key(self, provider: str) -> Optional[str]:
        """Get default API key for provider"""
        key_mapping = {
            "google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"), 
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "huggingface": os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
        }
        return key_mapping.get(provider)
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider"""
        model_mapping = {
            "huggingface": "microsoft/DialoGPT-medium",  # Free, accessible conversational model
            "google": "gemini-1.5-flash",
            "openai": "gpt-3.5-turbo", 
            "anthropic": "claude-3-5-sonnet-20241022"
        }
        return model_mapping.get(provider, "microsoft/DialoGPT-medium")
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        try:
            if self.provider == "huggingface" and HUGGINGFACE_AVAILABLE:
                self.client = InferenceClient(token=self.api_key)
            elif self.provider == "google" and GOOGLE_AI_AVAILABLE:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                self.client = openai.OpenAI(api_key=self.api_key)
            # Add other providers as needed
        except Exception as e:
            print(f"Warning: Could not initialize {self.provider} client: {e}")
            self.client = None
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        if not self.client:
            return self._fallback_response(prompt)
        
        try:
            if self.provider == "huggingface" and HUGGINGFACE_AVAILABLE:
                
                try:
                    # Try using a simpler model that's guaranteed to work
                    simple_models = [
                        "gpt2",
                        "microsoft/DialoGPT-medium", 
                        "microsoft/DialoGPT-small"
                    ]
                    
                    for model in simple_models:
                        try:
                            response = self.client.text_generation(
                                prompt,
                                model=model,
                                max_new_tokens=kwargs.get("max_tokens", 300),
                                temperature=kwargs.get("temperature", 0.7),
                                return_full_text=False
                            )
                            if response:
                                return response
                        except Exception as model_error:
                            continue
                    
                    # If all models fail, return fallback
                    return self._fallback_response(prompt)
                    
                except Exception as hf_error:
                    return self._fallback_response(prompt)
            elif self.provider == "google" and GOOGLE_AI_AVAILABLE:
                response = self.client.generate_content(prompt)
                return response.text
            elif self.provider == "openai" and OPENAI_AVAILABLE:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=kwargs.get("max_tokens", 2000),
                    temperature=kwargs.get("temperature", 0.7)
                )
                return response.choices[0].message.content
            else:
                return self._fallback_response(prompt)
        except Exception as e:
            print(f"Error generating response: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM is not available"""
        
        # Enhanced topic extraction for better responses
        topic_summary = self._extract_topic_from_prompt(prompt)
        
        if "research" in prompt.lower():
            return f"""# Research Analysis for {topic_summary.title()}

## Overview
Based on your research request about **{topic_summary}**, I can provide a structured research framework:

## Key Research Areas
1. **Current State & Trends**
   - Market dynamics and recent developments
   - Key players and stakeholders
   - Statistical data and metrics

2. **Historical Context**
   - Evolution and timeline
   - Major milestones and breakthroughs
   - Lessons learned from past experiences

3. **Multiple Perspectives**
   - Industry viewpoints
   - Academic research findings
   - Consumer/user perspectives
   - Regulatory considerations

4. **Future Outlook**
   - Emerging trends and predictions
   - Potential challenges and opportunities
   - Innovation areas to watch

## Research Methodology
- **Primary Sources**: Direct interviews, surveys, official reports
- **Secondary Sources**: Academic papers, industry reports, news articles
- **Data Analysis**: Statistical trends, comparative studies
- **Expert Opinions**: Industry leaders, academic researchers

## Next Steps
1. Define specific research questions
2. Identify key information sources
3. Gather and analyze data systematically
4. Synthesize findings into actionable insights

*This analysis provides a foundation for comprehensive research on {topic_summary}. Use the research tools available in ARIA to gather specific data and insights.*"""
        
        return f"""I understand your request about **{topic_summary}**. 

Here's a structured approach to help you:

## Analysis Framework
- **Define scope and objectives**
- **Identify key information sources** 
- **Gather relevant data and insights**
- **Synthesize findings**

## Recommended Resources
- Academic databases and journals
- Industry reports and whitepapers  
- Government and regulatory sources
- Expert interviews and surveys

Please use the available research tools to gather specific information on your topic."""
    
    def _extract_topic_from_prompt(self, prompt: str) -> str:
        """Extract topic from research prompt"""
        prompt_lower = prompt.lower()
        
        # Look for "Topic: " pattern first (most specific)
        if "topic:" in prompt_lower:
            topic_start = prompt_lower.find("topic:") + len("topic:")
            topic_line = prompt[topic_start:].split('\n')[0].strip()
            if topic_line:
                return topic_line
        
        # Look for "research on" pattern
        if "research on" in prompt_lower:
            topic_part = prompt_lower.split("research on")[1].split("\n")[0].split(".")[0].strip()
            if topic_part and len(topic_part) < 200:
                return topic_part
        
        # Look for other patterns
        if "about" in prompt_lower:
            topic_part = prompt_lower.split("about")[1].split("\n")[0].split(".")[0].strip()
            if topic_part and len(topic_part) < 200:
                return topic_part
        
        # Fallback: look for topic-like content
        lines = prompt.split('\n')
        for line in lines:
            line = line.strip()
            # Look for lines that might contain the topic
            if (len(line) > 10 and len(line) < 200 and 
                not line.startswith('Please') and 
                not line.startswith('Structure') and
                not line.startswith('1.') and
                not line.startswith('2.')):
                return line
        
        return "the specified topic"
    
    def is_available(self) -> bool:
        """Check if the LLM client is available and functional"""
        return self.client is not None


def create_llm_client(config: Dict[str, Any] = None) -> SimpleLLMClient:
    """
    Create LLM client based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured SimpleLLMClient instance
    """
    if config is None:
        config = {}
    
    # Try providers in order of preference (prioritize working providers)
    providers = ["google", "openai", "anthropic", "huggingface"]
    
    for provider in providers:
        api_key = None
        if provider == "huggingface":
            api_key = os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN")
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if api_key:
            client = SimpleLLMClient(
                provider=provider,
                api_key=api_key,
                model=config.get("model")
            )
            if client.is_available():
                return client
    
    # Return fallback client
    return SimpleLLMClient(provider="fallback")


def get_available_providers() -> List[str]:
    """
    Get list of available LLM providers
    
    Returns:
        List of available provider names
    """
    providers = []
    
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append("google")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.getenv("HUGGING_FACE_API") or os.getenv("HF_TOKEN"):
        providers.append("huggingface")
    
    return providers