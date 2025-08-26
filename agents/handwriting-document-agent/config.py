"""
Configuration for Handwriting & Historical Document Analysis Agent
Optimized for HuggingFace models with enterprise document processing
Author: Mohammed Hamdan
"""

import os
from typing import Dict, Any, List, Optional, ClassVar
from dataclasses import dataclass
from pydantic import Field
from pydantic_settings import BaseSettings

@dataclass
class DocumentModelConfig:
    """Configuration for document processing models"""
    name: str
    models: List[str]
    description: str
    max_tokens: int = 4096
    temperature: float = 0.1

# Supported document processing providers
SUPPORTED_PROVIDERS: ClassVar[Dict[str, DocumentModelConfig]] = {
    "huggingface": DocumentModelConfig(
        name="HuggingFace Document Models",
        models=[
            # Latest 2024-2025 High-Performance OCR Models
            "stepfun-ai/GOT-OCR2_0",  # 580M params, unified end-to-end OCR
            "stepfun-ai/GOT-OCR-2.0-hf",  # HF optimized version
            
            # TrOCR Models (Proven State-of-Art)
            "microsoft/trocr-large-handwritten",
            "microsoft/trocr-base-handwritten", 
            "microsoft/trocr-large-printed",
            "microsoft/trocr-base-printed",
            
            # Advanced Document Understanding
            "microsoft/layoutlmv3-base",
            "microsoft/layoutlmv2-base-uncased",
            "microsoft/DiT-base-finetuned-rvlcdip",
            
            # Latest OpenOCR (High accuracy, fast)
            "topdu/OpenOCR",
            
            # Specialized Handwriting & Historical
            "Saifullah/handwritten-text-recognition-model",
            "microsoft/trocr-large-stage1",  # Pre-trained stage
            
            # Document Structure & Tables
            "microsoft/table-transformer-structure-recognition",
            "microsoft/table-transformer-detection",
            "nielsr/layoutlm-finetuned-funsd",
            
            # Multi-modal Document Analysis
            "microsoft/kosmos-2-patch14-224",  # Vision-language model
            "naver-clova-ix/donut-base",  # End-to-end document understanding
            
            # NLP Models for Analysis
            "facebook/bart-large-cnn",  # Summarization
            "microsoft/DialoGPT-medium",  # Conversational AI
        ],
        description="Specialized document processing models for OCR, handwriting, and historical analysis",
        max_tokens=4096,
        temperature=0.1
    ),
    "openai": DocumentModelConfig(
        name="OpenAI GPT Models",
        models=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        description="Advanced language models for document analysis",
        max_tokens=4096,
        temperature=0.1
    ),
    "anthropic": DocumentModelConfig(
        name="Anthropic Claude Models", 
        models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        description="Reasoning models for complex document understanding",
        max_tokens=4096,
        temperature=0.1
    )
}

class DocumentAgentConfig(BaseSettings):
    """Configuration for Handwriting Document Analysis Agent"""
    
    # API Keys from environment
    huggingface_api_key: str = Field(default="", env="HUGGINGFACE_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # Document Processing Settings
    max_file_size_mb: int = Field(default=50, description="Maximum upload file size in MB")
    supported_formats: List[str] = Field(
        default=[
            # Images
            ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
            # Documents  
            ".pdf", ".txt", ".docx", ".doc",
            # Historical formats
            ".tif", ".jp2"
        ]
    )
    
    # OCR Settings
    ocr_confidence_threshold: float = Field(default=0.7, description="Minimum OCR confidence")
    handwriting_threshold: float = Field(default=0.6, description="Handwriting detection threshold")
    
    # Vector Database Settings
    vector_db_path: str = Field(default="./data/chroma_db", description="ChromaDB storage path")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = Field(default=1000, description="Text chunk size for embeddings")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    
    # Document Analysis Settings
    analysis_depth: str = Field(default="comprehensive", description="Analysis depth: quick|standard|comprehensive")
    include_metadata_extraction: bool = Field(default=True)
    include_handwriting_analysis: bool = Field(default=True)
    include_historical_context: bool = Field(default=True)
    
    # LangGraph Workflow Settings
    max_workflow_steps: int = Field(default=20, description="Maximum workflow steps")
    workflow_timeout: int = Field(default=600, description="Workflow timeout in seconds")
    enable_parallel_processing: bool = Field(default=True)
    
    # Output Settings
    output_formats: List[str] = Field(
        default=["json", "markdown", "html", "pdf"],
        description="Available output formats"
    )
    
    # Cache Settings
    enable_caching: bool = Field(default=True)
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow"
    }

# Global configuration instance
config = DocumentAgentConfig()

# Document processing templates
ANALYSIS_TEMPLATES = {
    "handwriting_analysis": """
    Analyze this handwritten text for:
    1. Readability and clarity
    2. Historical period indicators
    3. Writing style and characteristics
    4. Potential author demographics
    5. Content summary and key themes
    """,
    
    "historical_document": """
    Analyze this historical document for:
    1. Time period and dating clues
    2. Historical context and significance
    3. Document type and purpose
    4. Key historical figures or events mentioned
    5. Preservation recommendations
    """,
    
    "document_structure": """
    Analyze document structure and layout:
    1. Document organization and sections
    2. Visual elements (images, tables, diagrams)
    3. Typography and formatting
    4. Annotations and marginalia
    5. Document condition assessment
    """
}

# Supported document types with specialized processing
DOCUMENT_TYPES = {
    "manuscript": {
        "description": "Handwritten historical manuscripts",
        "models": ["microsoft/trocr-large-handwritten", "microsoft/layoutlmv3-base"],
        "analysis_template": "handwriting_analysis"
    },
    "printed_historical": {
        "description": "Historical printed documents",
        "models": ["microsoft/trocr-large-printed", "microsoft/DiT-base-finetuned-rvlcdip"],
        "analysis_template": "historical_document"
    },
    "mixed_document": {
        "description": "Documents with both handwritten and printed text",
        "models": ["microsoft/trocr-large-handwritten", "microsoft/trocr-large-printed"],
        "analysis_template": "document_structure"
    },
    "modern_handwriting": {
        "description": "Modern handwritten documents",
        "models": ["Saifullah/handwritten-text-recognition-model"],
        "analysis_template": "handwriting_analysis"
    }
}