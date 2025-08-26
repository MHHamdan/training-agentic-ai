import logging
import json
from typing import Optional, Dict, Any
import aiohttp
import asyncio
from langsmith import traceable

logger = logging.getLogger(__name__)

class HuggingFaceModel:
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
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    @traceable(name="hf_generate", metadata={"component": "hf_model"})
    async def generate(self, prompt: str) -> str:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("generated_text", "")
                        return str(result)
                    else:
                        error_text = await response.text()
                        logger.error(f"HuggingFace API error: {error_text}")
                        return f"Error: {response.status} - {error_text}"
            except asyncio.TimeoutError:
                logger.error(f"Timeout calling HuggingFace API for {self.model_name}")
                return "Error: Request timeout"
            except Exception as e:
                logger.error(f"Error calling HuggingFace API: {str(e)}")
                return f"Error: {str(e)}"
    
    @traceable(name="hf_analyze_resume", metadata={"component": "hf_model"})
    async def analyze_resume(
        self,
        resume_text: str,
        job_requirements: str,
        prompt_template: str
    ) -> Dict[str, Any]:
        prompt = prompt_template.format(
            resume_text=resume_text[:3000],
            job_requirements=job_requirements[:1000]
        )
        
        response = await self.generate(prompt)
        
        # If HuggingFace API fails or returns error, raise exception to trigger fallback
        if isinstance(response, str) and ("Error:" in response or "timeout" in response.lower() or len(response.strip()) < 10):
            logger.warning(f"HuggingFace model returned error/short response: {response[:100]}")
            raise Exception(f"HuggingFace model failed: {response[:100]}")
        
        try:
            if response.startswith("{"):
                parsed = json.loads(response)
                # Validate that we actually got meaningful scores
                if "scores" in parsed and isinstance(parsed["scores"], dict):
                    return parsed
                else:
                    raise Exception("Invalid JSON response structure")
            else:
                lines = response.strip().split("\n")
                scores = {}
                insights = []
                
                for line in lines:
                    if ":" in line:
                        parts = line.split(":", 1)
                        key = parts[0].strip().lower().replace(" ", "_")
                        value = parts[1].strip()
                        
                        if any(word in key for word in ["technical", "experience", "cultural", "growth", "risk", "overall"]):
                            try:
                                scores[key] = float(value.replace("%", ""))
                            except:
                                pass
                        else:
                            insights.append(line)
                
                # If no scores were extracted, raise exception to trigger content-based analysis
                if not scores:
                    logger.warning("No scores extracted from HuggingFace response, triggering fallback")
                    raise Exception("No parseable scores in HuggingFace response")
                
                return {
                    "scores": scores,
                    "insights": insights or ["Analysis completed"],
                    "raw_response": response[:500]
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse HuggingFace JSON response: {str(e)}")
            raise Exception(f"Invalid JSON from HuggingFace: {str(e)}")
        except Exception as e:
            logger.warning(f"HuggingFace analysis failed: {str(e)}")
            raise e