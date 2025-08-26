"""
Response Formatter for Comprehensive AI Assistant
Formats multi-service data into coherent, user-friendly responses
Author: Mohammed Hamdan
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ResponseFormatter:
    """
    Formats aggregated data from multiple APIs into user-friendly responses
    """
    
    def __init__(self):
        logger.info("📝 Response Formatter initialized")
    
    async def format_comprehensive_response(
        self, 
        query: str, 
        data: Dict[str, Any], 
        intent: Dict[str, Any]
    ) -> str:
        """
        Format a comprehensive response based on collected data
        """
        try:
            response_sections = []
            
            # Add greeting and context
            response_sections.append(f"🤖 **Here's what I found for:** _{query}_\n")
            
            # Format weather information
            if data.get("weather") and any(data["weather"].values()):
                weather_section = self._format_weather_section(data["weather"])
                if weather_section:
                    response_sections.append(weather_section)
            
            # Format news information
            if data.get("news") and data["news"]:
                news_section = self._format_news_section(data["news"])
                if news_section:
                    response_sections.append(news_section)
            
            # Format places/restaurants
            if data.get("places") and data["places"]:
                places_section = self._format_places_section(data["places"])
                if places_section:
                    response_sections.append(places_section)
            
            # Format financial information
            if data.get("finance") and any(data["finance"].values()):
                finance_section = self._format_finance_section(data["finance"])
                if finance_section:
                    response_sections.append(finance_section)
            
            # Format entertainment
            if data.get("entertainment") and data["entertainment"]:
                entertainment_section = self._format_entertainment_section(data["entertainment"])
                if entertainment_section:
                    response_sections.append(entertainment_section)
            
            # Format sports
            if data.get("sports") and data["sports"]:
                sports_section = self._format_sports_section(data["sports"])
                if sports_section:
                    response_sections.append(sports_section)
            
            # Format health/nutrition
            if data.get("health") and data["health"]:
                health_section = self._format_health_section(data["health"])
                if health_section:
                    response_sections.append(health_section)
            
            # Format shopping
            if data.get("shopping") and data["shopping"]:
                shopping_section = self._format_shopping_section(data["shopping"])
                if shopping_section:
                    response_sections.append(shopping_section)
            
            # Add footer with timestamp
            current_time = datetime.now().strftime("%I:%M %p")
            response_sections.append(f"\n📅 _Updated at {current_time}_")
            
            # Join all sections
            final_response = "\n\n".join(response_sections)
            
            # If no data was found, provide a helpful fallback
            if len(response_sections) <= 2:  # Only greeting and timestamp
                final_response = self._format_fallback_response(query, intent)
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error formatting response: {e}")
            return f"I apologize, but I encountered an error while formatting the response for your query: '{query}'. Please try again."
    
    def _format_weather_section(self, weather_data: Dict[str, Any]) -> Optional[str]:
        """Format weather information"""
        try:
            if not weather_data:
                return None
            
            temp = weather_data.get("temperature")
            description = weather_data.get("description", "").title()
            location = weather_data.get("location", "your area")
            humidity = weather_data.get("humidity")
            
            section = "## 🌤️ Weather Information\n"
            
            if temp:
                section += f"**{location}:** {temp}°F"
                if description:
                    section += f" - {description}"
                section += "\n"
            
            if humidity:
                section += f"**Humidity:** {humidity}%\n"
            
            # Add weather advice
            if temp:
                if temp > 80:
                    section += "☀️ _Perfect weather for outdoor activities!_"
                elif temp < 40:
                    section += "🧥 _Bundle up - it's chilly out there!_"
                elif "rain" in description.lower():
                    section += "☂️ _Don't forget your umbrella!_"
                else:
                    section += "👌 _Nice weather for getting out and about!_"
            
            return section
            
        except Exception as e:
            logger.error(f"Error formatting weather: {e}")
            return None
    
    def _format_news_section(self, news_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format news information"""
        try:
            if not news_data:
                return None
            
            section = "## 📰 Latest News\n"
            
            for i, article in enumerate(news_data[:5], 1):
                title = article.get("title", "").strip()
                description = article.get("description", "").strip()
                source = article.get("source", {})
                
                if isinstance(source, dict):
                    source_name = source.get("name", "Unknown")
                else:
                    source_name = str(source) if source else "Unknown"
                
                if title:
                    section += f"**{i}. {title}**\n"
                    if description and len(description) > 0:
                        # Truncate long descriptions
                        if len(description) > 150:
                            description = description[:147] + "..."
                        section += f"   _{description}_ - *{source_name}*\n\n"
                    else:
                        section += f"   *Source: {source_name}*\n\n"
            
            return section.rstrip()
            
        except Exception as e:
            logger.error(f"Error formatting news: {e}")
            return None
    
    def _format_places_section(self, places_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format places/restaurants information"""
        try:
            if not places_data:
                return None
            
            section = "## 🍽️ Recommended Places\n"
            
            for i, place in enumerate(places_data[:5], 1):
                name = place.get("name", "").strip()
                rating = place.get("rating")
                price = place.get("price")
                categories = place.get("categories", [])
                location = place.get("location", {})
                
                if name:
                    section += f"**{i}. {name}**"
                    
                    # Add rating
                    if rating:
                        stars = "⭐" * min(int(float(rating)), 5)
                        section += f" {stars} ({rating})"
                    
                    # Add price level
                    if price:
                        section += f" • {price}"
                    
                    section += "\n"
                    
                    # Add categories
                    if categories:
                        if isinstance(categories, list) and categories:
                            cat_names = []
                            for cat in categories[:2]:  # Limit to 2 categories
                                if isinstance(cat, dict):
                                    cat_names.append(cat.get("title", cat.get("name", "")))
                                else:
                                    cat_names.append(str(cat))
                            if cat_names:
                                section += f"   *{', '.join(filter(None, cat_names))}*\n"
                    
                    # Add address
                    if isinstance(location, dict):
                        address = location.get("display_address", location.get("address1", ""))
                        if isinstance(address, list):
                            address = ", ".join(address)
                        if address:
                            section += f"   📍 {address}\n"
                    
                    section += "\n"
            
            return section.rstrip()
            
        except Exception as e:
            logger.error(f"Error formatting places: {e}")
            return None
    
    def _format_finance_section(self, finance_data: Dict[str, Any]) -> Optional[str]:
        """Format financial information"""
        try:
            if not finance_data:
                return None
            
            section = "## 💰 Financial Information\n"
            
            # Stock information
            stocks = finance_data.get("stocks", {})
            if stocks:
                symbol = stocks.get("01. symbol", "")
                price = stocks.get("05. price", "")
                change = stocks.get("09. change", "")
                change_percent = stocks.get("10. change percent", "")
                
                if symbol and price:
                    section += f"**{symbol}:** ${price}"
                    if change and change_percent:
                        change_float = float(change) if change else 0
                        emoji = "📈" if change_float >= 0 else "📉"
                        section += f" {emoji} {change} ({change_percent})"
                    section += "\n"
            
            # Crypto information
            crypto = finance_data.get("crypto", {})
            if crypto:
                for coin, data in crypto.items():
                    if isinstance(data, dict):
                        price = data.get("usd")
                        change = data.get("usd_24h_change")
                        if price:
                            section += f"**{coin.title()}:** ${price:,.2f}"
                            if change:
                                emoji = "📈" if change >= 0 else "📉"
                                section += f" {emoji} {change:+.2f}%"
                            section += "\n"
            
            return section.rstrip() if section.strip() != "## 💰 Financial Information" else None
            
        except Exception as e:
            logger.error(f"Error formatting finance: {e}")
            return None
    
    def _format_entertainment_section(self, entertainment_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format entertainment information"""
        try:
            if not entertainment_data:
                return None
            
            section = "## 🎬 Entertainment\n"
            
            for i, item in enumerate(entertainment_data[:5], 1):
                title = item.get("title", item.get("name", "")).strip()
                overview = item.get("overview", "").strip()
                rating = item.get("vote_average")
                release_date = item.get("release_date", item.get("first_air_date", ""))
                media_type = item.get("media_type", "movie")
                
                if title:
                    emoji = "🎬" if media_type == "movie" else "📺"
                    section += f"**{i}. {emoji} {title}**"
                    
                    if rating:
                        stars = "⭐" * min(int(float(rating) / 2), 5)
                        section += f" {stars} ({rating}/10)"
                    
                    if release_date:
                        year = release_date.split("-")[0] if "-" in release_date else release_date
                        section += f" ({year})"
                    
                    section += "\n"
                    
                    if overview and len(overview) > 0:
                        if len(overview) > 120:
                            overview = overview[:117] + "..."
                        section += f"   _{overview}_\n\n"
                    else:
                        section += "\n"
            
            return section.rstrip()
            
        except Exception as e:
            logger.error(f"Error formatting entertainment: {e}")
            return None
    
    def _format_sports_section(self, sports_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format sports information"""
        try:
            if not sports_data:
                return None
            
            section = "## ⚽ Sports Updates\n"
            
            games = sports_data.get("games", [])
            for i, game in enumerate(games[:5], 1):
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")
                score = game.get("score", "")
                status = game.get("status", "")
                
                if home_team and away_team:
                    section += f"**{i}. {away_team} vs {home_team}**"
                    if score:
                        section += f" - {score}"
                    if status:
                        section += f" ({status})"
                    section += "\n"
            
            return section.rstrip() if games else None
            
        except Exception as e:
            logger.error(f"Error formatting sports: {e}")
            return None
    
    def _format_health_section(self, health_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format health and nutrition information"""
        try:
            if not health_data:
                return None
            
            section = "## 🏥 Health & Nutrition\n"
            
            for i, item in enumerate(health_data[:3], 1):
                title = item.get("title", "").strip()
                ready_in_minutes = item.get("readyInMinutes")
                servings = item.get("servings")
                
                if title:
                    section += f"**{i}. {title}**\n"
                    
                    details = []
                    if ready_in_minutes:
                        details.append(f"⏱️ {ready_in_minutes} min")
                    if servings:
                        details.append(f"👥 {servings} servings")
                    
                    if details:
                        section += f"   {' • '.join(details)}\n"
                    section += "\n"
            
            return section.rstrip()
            
        except Exception as e:
            logger.error(f"Error formatting health: {e}")
            return None
    
    def _format_shopping_section(self, shopping_data: List[Dict[str, Any]]) -> Optional[str]:
        """Format shopping information"""
        try:
            if not shopping_data:
                return None
            
            section = "## 🛒 Shopping Results\n"
            
            items = shopping_data.get("items", shopping_data)
            for i, item in enumerate(items[:5], 1):
                name = item.get("name", item.get("title", "")).strip()
                price = item.get("salePrice", item.get("price"))
                
                if name:
                    section += f"**{i}. {name}**"
                    if price:
                        section += f" - ${price}"
                    section += "\n"
            
            return section.rstrip() if items else None
            
        except Exception as e:
            logger.error(f"Error formatting shopping: {e}")
            return None
    
    def _format_fallback_response(self, query: str, intent: Dict[str, Any]) -> str:
        """Format a fallback response when no data is available"""
        try:
            fallback = f"🤖 I received your request about: _{query}_\n\n"
            
            fallback += "I'm currently working on gathering information from multiple sources. "
            fallback += "While I couldn't retrieve specific data at this moment, I can help you with:\n\n"
            
            fallback += "• 📰 **Latest news and headlines**\n"
            fallback += "• 🌤️ **Weather forecasts and conditions**\n"
            fallback += "• 🍽️ **Restaurant and place recommendations**\n"
            fallback += "• 💰 **Stock prices and financial data**\n"
            fallback += "• 🎬 **Movie and TV recommendations**\n"
            fallback += "• ⚽ **Sports scores and updates**\n"
            fallback += "• 🏥 **Health and nutrition information**\n"
            fallback += "• 🛒 **Shopping and product searches**\n\n"
            
            fallback += "💡 **Try asking me:**\n"
            fallback += "- \"What's the weather like?\"\n"
            fallback += "- \"Show me the latest tech news\"\n"
            fallback += "- \"Find restaurants near me\"\n"
            fallback += "- \"What's the AAPL stock price?\"\n"
            fallback += "- \"Recommend a good movie\"\n\n"
            
            current_time = datetime.now().strftime("%I:%M %p")
            fallback += f"📅 _Response generated at {current_time}_"
            
            return fallback
            
        except Exception as e:
            logger.error(f"Error creating fallback response: {e}")
            return f"I apologize, but I'm having trouble processing your request: '{query}'. Please try again with a more specific question."
    
    def format_quick_summary(self, data: Dict[str, Any]) -> str:
        """Format a quick one-line summary"""
        try:
            summaries = []
            
            if data.get("weather") and data["weather"].get("temperature"):
                temp = data["weather"]["temperature"]
                desc = data["weather"].get("description", "")
                summaries.append(f"🌤️ {temp}°F {desc}")
            
            if data.get("news") and data["news"]:
                summaries.append(f"📰 {len(data['news'])} news updates")
            
            if data.get("places") and data["places"]:
                summaries.append(f"🍽️ {len(data['places'])} places found")
            
            if data.get("finance") and data["finance"]:
                summaries.append("💰 Market data available")
            
            return " • ".join(summaries) if summaries else "Information gathered from multiple sources"
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return "Data processed successfully"