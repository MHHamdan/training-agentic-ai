"""
Multi-API Service Integration
Handles concurrent API calls with rate limiting, caching, and error handling
Author: Mohammed Hamdan
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

from config.settings import API_CONFIGS, WORKFLOW_CONFIG

logger = logging.getLogger(__name__)

class APICache:
    """Simple in-memory cache for API responses"""
    
    def __init__(self, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, service: str, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key from service, endpoint, and parameters"""
        content = f"{service}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, service: str, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if valid"""
        key = self._generate_key(service, endpoint, params)
        
        if key in self.cache:
            cached_item = self.cache[key]
            if cached_item["expires_at"] > datetime.now():
                logger.debug(f"✅ Cache hit for {service}")
                return cached_item["data"]
            else:
                # Remove expired item
                del self.cache[key]
        
        return None
    
    def set(self, service: str, endpoint: str, params: Dict[str, Any], data: Dict[str, Any], ttl: int = None):
        """Cache response data"""
        key = self._generate_key(service, endpoint, params)
        expires_at = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        
        self.cache[key] = {
            "data": data,
            "expires_at": expires_at,
            "cached_at": datetime.now()
        }
        
        logger.debug(f"📦 Cached response for {service}")
    
    def clear(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("🗑️ Cache cleared")

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self):
        self.call_times: Dict[str, List[float]] = {}
    
    async def wait_if_needed(self, service: str, limit: int, window: int = 3600):
        """Wait if rate limit would be exceeded"""
        now = time.time()
        
        if service not in self.call_times:
            self.call_times[service] = []
        
        # Remove calls outside the window
        self.call_times[service] = [
            call_time for call_time in self.call_times[service]
            if now - call_time < window
        ]
        
        # Check if we need to wait
        if len(self.call_times[service]) >= limit:
            sleep_time = window - (now - self.call_times[service][0])
            if sleep_time > 0:
                logger.warning(f"⏱️ Rate limit reached for {service}, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this call
        self.call_times[service].append(now)

class MultiAPIService:
    """
    Service for making concurrent API calls to multiple providers
    """
    
    def __init__(self):
        self.cache = APICache(WORKFLOW_CONFIG["cache_duration"])
        self.rate_limiter = RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("🔌 Multi-API Service initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=WORKFLOW_CONFIG["api_timeout"])
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _make_api_call(self, service_name: str, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a single API call with error handling and caching"""
        try:
            config = API_CONFIGS.get(service_name)
            if not config:
                return {"success": False, "error": f"Unknown service: {service_name}"}
            
            # Check cache first
            params = params or {}
            cached_response = self.cache.get(service_name, endpoint, params)
            if cached_response:
                return {"success": True, "data": cached_response, "source": "cache"}
            
            # Rate limiting
            await self.rate_limiter.wait_if_needed(service_name, config.rate_limit)
            
            # Prepare request
            session = await self._get_session()
            url = config.base_url + endpoint
            
            headers = {}
            if config.api_key:
                if service_name == "newsapi":
                    headers["X-API-Key"] = config.api_key
                elif service_name in ["tmdb", "openweather"]:
                    params["api_key"] = config.api_key
                elif service_name == "yelp":
                    headers["Authorization"] = f"Bearer {config.api_key}"
                elif service_name == "foursquare":
                    headers["Authorization"] = config.api_key
                else:
                    headers["Authorization"] = f"Bearer {config.api_key}"
            
            # Make request
            logger.debug(f"🌐 Making API call to {service_name}: {url}")
            
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful response
                    self.cache.set(service_name, endpoint, params, data)
                    
                    return {"success": True, "data": data, "source": "api"}
                else:
                    error_text = await response.text()
                    logger.warning(f"⚠️ API call failed for {service_name}: {response.status} - {error_text}")
                    return {
                        "success": False, 
                        "error": f"HTTP {response.status}: {error_text}",
                        "status_code": response.status
                    }
        
        except asyncio.TimeoutError:
            error_msg = f"Timeout calling {service_name}"
            logger.error(f"⏰ {error_msg}")
            return {"success": False, "error": error_msg}
        
        except Exception as e:
            error_msg = f"Error calling {service_name}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
    
    async def fetch_news(self, query: str = "", location: str = "") -> Dict[str, Any]:
        """Fetch news from available news APIs"""
        try:
            # Try NewsAPI first
            if "newsapi" in API_CONFIGS:
                params = {
                    "q": query or "latest news",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10
                }
                result = await self._make_api_call("newsapi", "/top-headlines", params)
                if result["success"]:
                    return result
            
            # Fallback to DuckDuckGo
            if "duckduckgo" in API_CONFIGS:
                search_query = f"{query} news" if query else "latest news"
                result = await self._make_api_call("duckduckgo", f"/?q={search_query}&format=json&no_html=1")
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No news services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_weather(self, location: str) -> Dict[str, Any]:
        """Fetch weather information"""
        try:
            # Try OpenWeatherMap first
            if "openweather" in API_CONFIGS:
                params = {
                    "q": location,
                    "units": "imperial",
                    "appid": API_CONFIGS["openweather"].api_key
                }
                result = await self._make_api_call("openweather", "/weather", params)
                if result["success"]:
                    return result
            
            # Fallback to WeatherAPI
            if "weatherapi" in API_CONFIGS:
                params = {
                    "q": location,
                    "key": API_CONFIGS["weatherapi"].api_key
                }
                result = await self._make_api_call("weatherapi", "/current.json", params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No weather services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_places(self, query: str, location: str) -> Dict[str, Any]:
        """Fetch places and restaurants"""
        try:
            # Try Yelp first
            if "yelp" in API_CONFIGS:
                params = {
                    "term": query,
                    "location": location,
                    "limit": 10,
                    "sort_by": "rating"
                }
                result = await self._make_api_call("yelp", "/businesses/search", params)
                if result["success"]:
                    return result
            
            # Fallback to Foursquare
            if "foursquare" in API_CONFIGS:
                params = {
                    "query": query,
                    "near": location,
                    "limit": 10
                }
                result = await self._make_api_call("foursquare", "/search", params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No places services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_finance(self, symbol: str = "AAPL") -> Dict[str, Any]:
        """Fetch financial information"""
        try:
            # Try Alpha Vantage for stocks
            if "alphavantage" in API_CONFIGS:
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": API_CONFIGS["alphavantage"].api_key
                }
                result = await self._make_api_call("alphavantage", "", params)
                if result["success"]:
                    return result
            
            # Try CoinGecko for crypto
            if "coingecko" in API_CONFIGS and symbol.lower() in ["bitcoin", "btc", "ethereum", "eth"]:
                crypto_id = "bitcoin" if symbol.lower() in ["bitcoin", "btc"] else "ethereum"
                params = {
                    "ids": crypto_id,
                    "vs_currencies": "usd",
                    "include_24hr_change": "true"
                }
                result = await self._make_api_call("coingecko", "/simple/price", params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No finance services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_entertainment(self, query: str = "") -> Dict[str, Any]:
        """Fetch entertainment content"""
        try:
            if "tmdb" in API_CONFIGS:
                endpoint = "/search/multi" if query else "/movie/popular"
                params = {}
                if query:
                    params["query"] = query
                params["api_key"] = API_CONFIGS["tmdb"].api_key
                
                result = await self._make_api_call("tmdb", endpoint, params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No entertainment services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_sports(self, sport: str = "NBA") -> Dict[str, Any]:
        """Fetch sports information"""
        try:
            if "sportsdata" in API_CONFIGS:
                # Mock sports data since SportsData.io requires specific endpoints
                return {
                    "success": True,
                    "data": {
                        "games": [
                            {
                                "home_team": "Lakers",
                                "away_team": "Warriors",
                                "score": "110-105",
                                "status": "Final"
                            }
                        ]
                    },
                    "source": "mock"
                }
            
            return {"success": False, "error": "No sports services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_health(self, query: str) -> Dict[str, Any]:
        """Fetch health and nutrition information"""
        try:
            # Try Spoonacular for recipes/nutrition
            if "spoonacular" in API_CONFIGS:
                params = {
                    "query": query,
                    "number": 5,
                    "apiKey": API_CONFIGS["spoonacular"].api_key
                }
                result = await self._make_api_call("spoonacular", "/recipes/complexSearch", params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No health services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_shopping(self, query: str) -> Dict[str, Any]:
        """Fetch shopping information"""
        try:
            if "walmart" in API_CONFIGS:
                params = {
                    "query": query,
                    "format": "json"
                }
                result = await self._make_api_call("walmart", "/search", params)
                if result["success"]:
                    return result
            
            return {"success": False, "error": "No shopping services available"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fetch_from_multiple_services(self, services: List[str], query: str, location: str) -> Dict[str, Any]:
        """
        Fetch data from multiple services concurrently
        """
        logger.info(f"🔄 Fetching data from {len(services)} services: {services}")
        
        # Create tasks for concurrent execution
        tasks = {}
        
        for service in services:
            if service in ["newsapi", "duckduckgo"]:
                tasks[service] = self.fetch_news(query, location)
            elif service in ["openweather", "weatherapi"]:
                tasks[service] = self.fetch_weather(location)
            elif service in ["yelp", "foursquare"]:
                tasks[service] = self.fetch_places(query, location)
            elif service in ["alphavantage", "coingecko"]:
                # Extract stock symbol from query if present
                symbol = "AAPL"  # Default
                if any(stock in query.upper() for stock in ["AAPL", "GOOGL", "TSLA", "MSFT"]):
                    for stock in ["AAPL", "GOOGL", "TSLA", "MSFT"]:
                        if stock in query.upper():
                            symbol = stock
                            break
                tasks[service] = self.fetch_finance(symbol)
            elif service == "tmdb":
                tasks[service] = self.fetch_entertainment(query)
            elif service == "sportsdata":
                tasks[service] = self.fetch_sports()
            elif service in ["spoonacular", "edamam_nutrition"]:
                tasks[service] = self.fetch_health(query)
            elif service == "walmart":
                tasks[service] = self.fetch_shopping(query)
        
        # Execute all tasks concurrently
        if tasks:
            try:
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                # Map results back to service names
                service_results = {}
                for i, service in enumerate(tasks.keys()):
                    result = results[i]
                    if isinstance(result, Exception):
                        service_results[service] = {
                            "success": False,
                            "error": str(result)
                        }
                    else:
                        service_results[service] = result
                
                successful_count = sum(1 for result in service_results.values() if result.get("success", False))
                logger.info(f"✅ Completed {successful_count}/{len(services)} API calls successfully")
                
                return service_results
                
            except Exception as e:
                logger.error(f"❌ Error in concurrent API calls: {e}")
                return {service: {"success": False, "error": str(e)} for service in services}
        
        return {}
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("🔌 HTTP session closed")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all configured services"""
        status = {}
        for service_name, config in API_CONFIGS.items():
            has_real_key = (
                config.api_key and 
                config.api_key != "" and 
                not config.api_key.startswith("mock-")
            )
            status[service_name] = {
                "name": config.name,
                "configured": has_real_key or config.api_key == "",
                "rate_limit": config.rate_limit,
                "free_tier": config.free_tier
            }
        return status