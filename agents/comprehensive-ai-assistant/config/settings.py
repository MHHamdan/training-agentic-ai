"""
Configuration settings for Comprehensive AI Assistant Agent
Handles API keys, endpoints, and service configurations
Author: Mohammed Hamdan
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for external APIs"""
    name: str
    base_url: str
    api_key: str
    rate_limit: int
    free_tier: bool = True
    endpoints: Dict[str, str] = None

# Core API Configurations
API_CONFIGS = {
    # News & Information
    "newsapi": APIConfig(
        name="NewsAPI",
        base_url="https://newsapi.org/v2",
        api_key=os.getenv("NEWSAPI_KEY", "mock-newsapi-key-12345"),
        rate_limit=1000,
        endpoints={
            "headlines": "/top-headlines",
            "everything": "/everything",
            "sources": "/sources"
        }
    ),
    
    "duckduckgo": APIConfig(
        name="DuckDuckGo News",
        base_url="https://api.duckduckgo.com",
        api_key="",  # No key required
        rate_limit=100,
        endpoints={
            "instant": "/?q={query}&format=json&no_html=1"
        }
    ),
    
    # Weather Services
    "openweather": APIConfig(
        name="OpenWeatherMap",
        base_url="https://api.openweathermap.org/data/2.5",
        api_key=os.getenv("OPENWEATHER_API_KEY", "mock-weather-key-67890"),
        rate_limit=1000,
        endpoints={
            "current": "/weather",
            "forecast": "/forecast",
            "alerts": "/onecall"
        }
    ),
    
    "weatherapi": APIConfig(
        name="WeatherAPI",
        base_url="https://api.weatherapi.com/v1",
        api_key=os.getenv("WEATHERAPI_KEY", "mock-weatherapi-key-11111"),
        rate_limit=1000000,
        endpoints={
            "current": "/current.json",
            "forecast": "/forecast.json",
            "alerts": "/alerts.json"
        }
    ),
    
    # Sports & Entertainment
    "sportsdata": APIConfig(
        name="SportsData.io",
        base_url="https://api.sportsdata.io/v3",
        api_key=os.getenv("SPORTSDATA_API_KEY", "mock-sports-key-22222"),
        rate_limit=1000,
        endpoints={
            "scores": "/scores/json/ScoresByDate",
            "teams": "/scores/json/Teams",
            "players": "/scores/json/Players"
        }
    ),
    
    "tmdb": APIConfig(
        name="The Movie Database",
        base_url="https://api.themoviedb.org/3",
        api_key=os.getenv("TMDB_API_KEY", "mock-tmdb-key-33333"),
        rate_limit=10000,
        endpoints={
            "movies": "/movie/popular",
            "tv": "/tv/popular",
            "search": "/search/multi"
        }
    ),
    
    # Financial Data
    "alphavantage": APIConfig(
        name="Alpha Vantage",
        base_url="https://www.alphavantage.co/query",
        api_key=os.getenv("ALPHAVANTAGE_API_KEY", "mock-alpha-key-44444"),
        rate_limit=5,
        endpoints={
            "stock": "?function=GLOBAL_QUOTE&symbol={symbol}",
            "forex": "?function=CURRENCY_EXCHANGE_RATE",
            "crypto": "?function=CRYPTOCURRENCY_RATING"
        }
    ),
    
    "coingecko": APIConfig(
        name="CoinGecko",
        base_url="https://api.coingecko.com/api/v3",
        api_key="",  # No key required for basic
        rate_limit=100,
        endpoints={
            "prices": "/simple/price",
            "coins": "/coins/markets",
            "trending": "/search/trending"
        }
    ),
    
    # Location & Places
    "foursquare": APIConfig(
        name="Foursquare Places",
        base_url="https://api.foursquare.com/v3/places",
        api_key=os.getenv("FOURSQUARE_API_KEY", "mock-foursquare-key-55555"),
        rate_limit=1000,
        endpoints={
            "search": "/search",
            "nearby": "/nearby",
            "details": "/{place_id}"
        }
    ),
    
    "yelp": APIConfig(
        name="Yelp Fusion",
        base_url="https://api.yelp.com/v3",
        api_key=os.getenv("YELP_API_KEY", "mock-yelp-key-66666"),
        rate_limit=5000,
        endpoints={
            "search": "/businesses/search",
            "details": "/businesses/{id}",
            "reviews": "/businesses/{id}/reviews"
        }
    ),
    
    # Shopping & Commerce
    "walmart": APIConfig(
        name="Walmart Open API",
        base_url="https://api.walmartlabs.com/v1",
        api_key=os.getenv("WALMART_API_KEY", "mock-walmart-key-77777"),
        rate_limit=1000,
        endpoints={
            "search": "/search",
            "product": "/items/{id}",
            "trending": "/trends"
        }
    ),
    
    # Health & Nutrition
    "edamam_nutrition": APIConfig(
        name="Edamam Nutrition",
        base_url="https://api.edamam.com/api/nutrition-data",
        api_key=os.getenv("EDAMAM_NUTRITION_KEY", "mock-edamam-key-88888"),
        rate_limit=1000,
        endpoints={
            "nutrition": "/v2/nutrients",
            "food": "/v2/parser"
        }
    ),
    
    "spoonacular": APIConfig(
        name="Spoonacular",
        base_url="https://api.spoonacular.com",
        api_key=os.getenv("SPOONACULAR_API_KEY", "mock-spoon-key-99999"),
        rate_limit=150,
        endpoints={
            "recipes": "/recipes/complexSearch",
            "ingredients": "/food/ingredients/search",
            "nutrition": "/recipes/{id}/nutritionWidget.json"
        }
    ),
    
    # Transportation
    "gas_buddy": APIConfig(
        name="GasBuddy",
        base_url="https://api.gasbuddy.com/v3",
        api_key=os.getenv("GASBUDDY_API_KEY", "mock-gas-key-00000"),
        rate_limit=1000,
        endpoints={
            "stations": "/stations/radius",
            "prices": "/stations/{id}/prices"
        }
    )
}

# LangSmith Configuration
LANGSMITH_CONFIG = {
    "api_key": os.getenv("LANGSMITH_API_KEY", "mock-langsmith-key-ls123"),
    "project_name": "comprehensive-ai-assistant",
    "endpoint": "https://api.smith.langchain.com",
    "tracing_enabled": True
}

# Workflow Configuration
WORKFLOW_CONFIG = {
    "max_concurrent_apis": 5,
    "api_timeout": 30,
    "retry_attempts": 3,
    "cache_duration": 300,  # 5 minutes
    "enable_observability": True,
    "log_level": "INFO"
}

# Service Categories
SERVICE_CATEGORIES = {
    "news": ["newsapi", "duckduckgo"],
    "weather": ["openweather", "weatherapi"],
    "sports": ["sportsdata"],
    "entertainment": ["tmdb"],
    "finance": ["alphavantage", "coingecko"],
    "places": ["foursquare", "yelp"],
    "shopping": ["walmart"],
    "health": ["edamam_nutrition", "spoonacular"],
    "transportation": ["gas_buddy"]
}

# User Personalization Settings
DEFAULT_USER_PREFERENCES = {
    "location": "San Francisco, CA",
    "interests": ["technology", "health", "finance"],
    "dietary_restrictions": [],
    "budget_range": "medium",
    "preferred_news_sources": ["technology", "business"],
    "weather_units": "imperial",
    "currency": "USD"
}

def get_api_config(service_name: str) -> Optional[APIConfig]:
    """Get configuration for a specific API service"""
    return API_CONFIGS.get(service_name)

def get_services_by_category(category: str) -> List[str]:
    """Get list of services for a specific category"""
    return SERVICE_CATEGORIES.get(category, [])

def validate_api_keys() -> Dict[str, bool]:
    """Validate which API keys are properly configured"""
    status = {}
    for name, config in API_CONFIGS.items():
        # Check if key exists and is not a mock key
        has_real_key = (
            config.api_key and 
            config.api_key != "" and 
            not config.api_key.startswith("mock-")
        )
        status[name] = has_real_key
    return status

def get_available_services() -> List[str]:
    """Get list of services that are properly configured"""
    validation = validate_api_keys()
    return [name for name, is_valid in validation.items() if is_valid or API_CONFIGS[name].api_key == ""]