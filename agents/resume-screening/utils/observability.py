import os
import logging
from typing import Any, Dict, Optional
from functools import wraps
import time
from datetime import datetime
from langsmith import Client, traceable

logger = logging.getLogger(__name__)

def setup_langsmith():
    try:
        if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "resume-screening-agent-v2")
            
            if os.getenv("LANGCHAIN_API_KEY"):
                client = Client()
                logger.info("LangSmith observability initialized successfully")
                return client
            else:
                logger.warning("LANGCHAIN_API_KEY not found, LangSmith disabled")
        else:
            logger.info("LangSmith tracing disabled")
    except Exception as e:
        logger.error(f"Error setting up LangSmith: {str(e)}")
    
    return None

def log_performance(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        error = None
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            log_entry = {
                "function": func.__name__,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "success": error is None,
                "error": error
            }
            
            if duration > 5:
                logger.warning(f"Slow operation: {func.__name__} took {duration:.2f}s")
            else:
                logger.debug(f"Operation {func.__name__} completed in {duration:.2f}s")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = None
        error = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            log_entry = {
                "function": func.__name__,
                "duration": duration,
                "timestamp": datetime.now().isoformat(),
                "success": error is None,
                "error": error
            }
            
            if duration > 5:
                logger.warning(f"Slow operation: {func.__name__} took {duration:.2f}s")
            else:
                logger.debug(f"Operation {func.__name__} completed in {duration:.2f}s")
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

class ObservabilityManager:
    def __init__(self):
        self.langsmith_client = setup_langsmith()
        self.metrics = {}
        self.start_time = datetime.now()
    
    @traceable(name="record_metric", metadata={"component": "observability"})
    def record_metric(self, metric_name: str, value: Any, metadata: Optional[Dict] = None):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        entry = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.metrics[metric_name].append(entry)
    
    @traceable(name="get_metrics_summary", metadata={"component": "observability"})
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics_count": len(self.metrics),
            "total_entries": sum(len(entries) for entries in self.metrics.values())
        }
        
        for metric_name, entries in self.metrics.items():
            if entries:
                values = [e["value"] for e in entries if isinstance(e["value"], (int, float))]
                if values:
                    summary[f"{metric_name}_avg"] = sum(values) / len(values)
                    summary[f"{metric_name}_min"] = min(values)
                    summary[f"{metric_name}_max"] = max(values)
        
        return summary
    
    def reset_metrics(self):
        self.metrics = {}
        self.start_time = datetime.now()

import asyncio