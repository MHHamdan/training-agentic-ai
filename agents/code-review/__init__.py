"""
Production-Ready Code Review Agent with Multi-Provider Support - Agent 15
Enterprise-grade AI code review assistant with full observability
"""

__version__ = "1.0.0"
__author__ = "Mohammed Hamdan"
__description__ = "Enterprise Code Review Agent with Multi-Provider Support"

from .agent import CodeReviewAgent
from .config import config

__all__ = ["CodeReviewAgent", "config"]