"""
Handwriting & Historical Document Analysis Agent
Enterprise-grade document processing with OCR, AI analysis, and interactive chat
Author: Mohammed Hamdan
"""

from .config import config, DOCUMENT_TYPES, ANALYSIS_TEMPLATES
from .graph.document_workflow import DocumentWorkflowManager
from .processors.document_processor import DocumentProcessor, ProcessingOptions, DocumentPage
from .rag.document_rag import DocumentRAGSystem, DocumentChunk, ChatResponse
from .models.document_models import DocumentModelManager, DocumentProcessingResult

__version__ = "1.0.0"
__author__ = "Mohammed Hamdan"

__all__ = [
    "config",
    "DOCUMENT_TYPES", 
    "ANALYSIS_TEMPLATES",
    "DocumentWorkflowManager",
    "DocumentProcessor",
    "ProcessingOptions",
    "DocumentPage",
    "DocumentRAGSystem",
    "DocumentChunk", 
    "ChatResponse",
    "DocumentModelManager",
    "DocumentProcessingResult"
]