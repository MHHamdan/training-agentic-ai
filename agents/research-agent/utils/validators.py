"""
Input validation utilities for Research Agent
Ensures data quality and security
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def validate_search_query(query: str) -> str:
    """
    Validate and sanitize search query
    
    Args:
        query: Raw search query
    
    Returns:
        Cleaned and validated query
    
    Raises:
        ValueError: If query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    # Remove excessive whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    
    # Check minimum length
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long")
    
    # Check maximum length
    if len(query) > 500:
        raise ValueError("Query must be less than 500 characters")
    
    # Remove potentially dangerous characters
    query = re.sub(r'[<>"\'\`;]', '', query)
    
    return query

def validate_url(url: str) -> bool:
    """
    Validate URL format
    
    Args:
        url: URL to validate
    
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_email(email: str) -> bool:
    """
    Validate email format
    
    Args:
        email: Email to validate
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_api_key(api_key: str, provider: str = "unknown") -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
        provider: API provider name
    
    Returns:
        True if format looks valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Basic length check
    if len(api_key) < 10:
        return False
    
    # Provider-specific validation
    if provider.lower() == "openai":
        return api_key.startswith("sk-")
    elif provider.lower() == "anthropic":
        return api_key.startswith("sk-ant-")
    elif provider.lower() == "google":
        return api_key.startswith("AIza")
    elif provider.lower() == "langfuse":
        return api_key.startswith(("sk-lf-", "pk-lf-"))
    elif provider.lower() == "huggingface":
        return api_key.startswith("hf_")
    
    # Generic validation
    return len(api_key) >= 20 and len(api_key) <= 200

def validate_research_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate research parameters
    
    Args:
        params: Research parameters
    
    Returns:
        Validated parameters
    
    Raises:
        ValueError: If parameters are invalid
    """
    validated = {}
    
    # Query validation
    if "query" in params:
        validated["query"] = validate_search_query(params["query"])
    
    # Max results validation
    if "max_results" in params:
        max_results = params["max_results"]
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100:
            raise ValueError("max_results must be between 1 and 100")
        validated["max_results"] = max_results
    
    # Depth validation
    if "depth" in params:
        depth = params["depth"]
        if depth not in ["quick", "standard", "comprehensive", "exhaustive"]:
            raise ValueError("depth must be one of: quick, standard, comprehensive, exhaustive")
        validated["depth"] = depth
    
    # Citation format validation
    if "citation_format" in params:
        citation_format = params["citation_format"]
        if citation_format not in ["APA", "MLA", "Chicago", "IEEE", "Harvard"]:
            raise ValueError("citation_format must be one of: APA, MLA, Chicago, IEEE, Harvard")
        validated["citation_format"] = citation_format
    
    # Quality threshold validation
    if "quality_threshold" in params:
        threshold = params["quality_threshold"]
        if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            raise ValueError("quality_threshold must be between 0 and 1")
        validated["quality_threshold"] = float(threshold)
    
    # Sources validation
    if "sources" in params:
        sources = params["sources"]
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")
        
        valid_sources = ["duckduckgo", "arxiv", "wikipedia", "news", "pubmed"]
        for source in sources:
            if source.lower() not in valid_sources:
                raise ValueError(f"Invalid source: {source}. Must be one of: {valid_sources}")
        
        validated["sources"] = [s.lower() for s in sources]
    
    return validated

def sanitize_text(text: str) -> str:
    """
    Sanitize text content
    
    Args:
        text: Text to sanitize
    
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove potentially dangerous HTML/script content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    return text

def validate_model_name(model_name: str) -> bool:
    """
    Validate model name format
    
    Args:
        model_name: Model name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False
    
    # Basic format check
    if len(model_name) < 3 or len(model_name) > 100:
        return False
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9._/-]+$', model_name):
        return False
    
    return True

def validate_file_path(file_path: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate file path
    
    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions
    
    Returns:
        True if valid, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        return False
    
    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        return False
    
    # Check extension if specified
    if allowed_extensions:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in [ext.lower() for ext in allowed_extensions]:
            return False
    
    return True

def validate_json_structure(data: Any, required_keys: List[str] = None) -> bool:
    """
    Validate JSON structure
    
    Args:
        data: Data to validate
        required_keys: List of required keys
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys:
        for key in required_keys:
            if key not in data:
                return False
    
    return True

def validate_research_output(output: Dict[str, Any]) -> bool:
    """
    Validate research output structure
    
    Args:
        output: Research output to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        "query",
        "research_id", 
        "synthesis",
        "key_insights",
        "citations"
    ]
    
    return validate_json_structure(output, required_keys)

class ValidationError(Exception):
    """Custom validation error"""
    pass

def safe_int(value: Any, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    """
    Safely convert value to integer with bounds checking
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Safe integer value
    """
    try:
        result = int(value)
        
        if min_val is not None and result < min_val:
            result = min_val
        
        if max_val is not None and result > max_val:
            result = max_val
        
        return result
    except (ValueError, TypeError):
        return default

def safe_float(value: Any, default: float = 0.0, min_val: float = None, max_val: float = None) -> float:
    """
    Safely convert value to float with bounds checking
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        min_val: Minimum allowed value
        max_val: Maximum allowed value
    
    Returns:
        Safe float value
    """
    try:
        result = float(value)
        
        if min_val is not None and result < min_val:
            result = min_val
        
        if max_val is not None and result > max_val:
            result = max_val
        
        return result
    except (ValueError, TypeError):
        return default