"""Message processing utilities for Customer Support Agent"""

import re
import html
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ProcessedMessage:
    """Processed message with extracted information"""
    original_content: str
    cleaned_content: str
    entities: Dict[str, List[str]]
    sentiment: Optional[str]
    urgency_indicators: List[str]
    intent: Optional[str]
    language: str
    word_count: int
    character_count: int


class MessageProcessor:
    """Process and analyze customer messages"""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': [
                'thank', 'thanks', 'grateful', 'appreciate', 'excellent', 'great',
                'awesome', 'wonderful', 'perfect', 'amazing', 'love', 'happy'
            ],
            'negative': [
                'angry', 'frustrated', 'terrible', 'awful', 'horrible', 'hate',
                'worst', 'useless', 'broken', 'disappointed', 'annoyed', 'furious'
            ],
            'urgent': [
                'urgent', 'emergency', 'immediately', 'asap', 'critical', 'serious',
                'important', 'quickly', 'rush', 'priority'
            ]
        }
        
        self.intent_patterns = {
            'question': [r'\?', r'\bwhat\b', r'\bhow\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', r'\bwhich\b'],
            'complaint': [r'\bbug\b', r'\berror\b', r'\bproblem\b', r'\bissue\b', r'\bbroken\b', r'\bnot working\b'],
            'request': [r'\bcan you\b', r'\bcould you\b', r'\bplease\b', r'\bi need\b', r'\bi want\b', r'\bi would like\b'],
            'compliment': [r'\bgreat job\b', r'\bwell done\b', r'\bexcellent\b', r'\bthank you\b'],
            'greeting': [r'\bhi\b', r'\bhello\b', r'\bhey\b', r'\bgood morning\b', r'\bgood afternoon\b'],
            'goodbye': [r'\bbye\b', r'\bgoodbye\b', r'\bsee you\b', r'\btalk later\b', r'\bthanks again\b']
        }
        
        self.entity_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'currency': r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:dollars?|USD|euros?|EUR)',
            'date': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
            'time': r'\b\d{1,2}:\d{2}(?:\s*[AaPp][Mm])?\b',
            'order_number': r'\b(?:order|ticket|ref|reference)[\s#:]*([A-Z0-9]{6,})\b',
            'error_code': r'\b(?:error|err)[\s#:]*([A-Z0-9]{3,})\b'
        }
    
    def process_message(self, content: str) -> ProcessedMessage:
        """Process a message and extract relevant information"""
        # Clean the content
        cleaned_content = self.clean_message(content)
        
        # Extract entities
        entities = self.extract_entities(content)
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(cleaned_content)
        
        # Detect urgency indicators
        urgency_indicators = self.detect_urgency_indicators(cleaned_content)
        
        # Classify intent
        intent = self.classify_intent(cleaned_content)
        
        # Detect language (basic)
        language = self.detect_language(cleaned_content)
        
        # Calculate metrics
        word_count = len(cleaned_content.split())
        character_count = len(cleaned_content)
        
        return ProcessedMessage(
            original_content=content,
            cleaned_content=cleaned_content,
            entities=entities,
            sentiment=sentiment,
            urgency_indicators=urgency_indicators,
            intent=intent,
            language=language,
            word_count=word_count,
            character_count=character_count
        )
    
    def clean_message(self, content: str) -> str:
        """Clean and normalize message content"""
        if not content:
            return ""
        
        # Decode HTML entities
        cleaned = html.unescape(content)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()
        
        # Remove common email signatures
        signature_patterns = [
            r'\n\s*--\s*\n.*$',  # Standard email signature
            r'\n\s*Sent from my .*$',  # Mobile signatures
            r'\n\s*Best regards.*$',  # Formal closings
            r'\n\s*Thanks.*$'  # Thank you closings
        ]
        
        for pattern in signature_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        return cleaned
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from message content"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # Remove duplicates
        
        return entities
    
    def analyze_sentiment(self, content: str) -> Optional[str]:
        """Analyze sentiment of the message"""
        content_lower = content.lower()
        
        sentiment_scores = {'positive': 0, 'negative': 0, 'urgent': 0}
        
        for sentiment_type, keywords in self.sentiment_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    sentiment_scores[sentiment_type] += 1
        
        # Determine dominant sentiment
        max_score = max(sentiment_scores.values())
        if max_score == 0:
            return 'neutral'
        
        dominant_sentiments = [k for k, v in sentiment_scores.items() if v == max_score]
        
        # Priority order: urgent > negative > positive
        if 'urgent' in dominant_sentiments:
            return 'urgent'
        elif 'negative' in dominant_sentiments:
            return 'negative'
        elif 'positive' in dominant_sentiments:
            return 'positive'
        else:
            return 'neutral'
    
    def detect_urgency_indicators(self, content: str) -> List[str]:
        """Detect indicators of urgency in the message"""
        urgency_indicators = []
        content_lower = content.lower()
        
        # Check for urgent keywords
        for keyword in self.sentiment_keywords['urgent']:
            if keyword in content_lower:
                urgency_indicators.append(f"urgent_keyword:{keyword}")
        
        # Check for multiple exclamation marks
        if re.search(r'!{2,}', content):
            urgency_indicators.append("multiple_exclamation_marks")
        
        # Check for ALL CAPS words
        caps_words = re.findall(r'\b[A-Z]{3,}\b', content)
        if caps_words:
            urgency_indicators.append(f"caps_words:{','.join(caps_words[:3])}")  # Limit to 3
        
        # Check for time-sensitive phrases
        time_sensitive_patterns = [
            r'\btoday\b', r'\bright now\b', r'\bimmediately\b',
            r'\basap\b', r'\bby \w+day\b', r'\bdeadline\b'
        ]
        
        for pattern in time_sensitive_patterns:
            if re.search(pattern, content_lower):
                urgency_indicators.append(f"time_sensitive:{pattern}")
        
        return urgency_indicators
    
    def classify_intent(self, content: str) -> Optional[str]:
        """Classify the intent of the message"""
        content_lower = content.lower()
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, content_lower))
                score += matches
            
            if score > 0:
                intent_scores[intent_type] = score
        
        if not intent_scores:
            return None
        
        # Return intent with highest score
        return max(intent_scores, key=intent_scores.get)
    
    def detect_language(self, content: str) -> str:
        """Basic language detection"""
        # Simple heuristic-based language detection
        # In production, you'd use a proper language detection library
        
        # Common English words
        english_indicators = [
            'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with',
            'for', 'as', 'was', 'on', 'are', 'you', 'this', 'be', 'at', 'by'
        ]
        
        # Common Spanish words
        spanish_indicators = [
            'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no',
            'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al'
        ]
        
        # Common French words
        french_indicators = [
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
            'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se'
        ]
        
        content_lower = content.lower()
        words = content_lower.split()
        
        if len(words) < 3:
            return 'unknown'
        
        english_score = sum(1 for word in words if word in english_indicators)
        spanish_score = sum(1 for word in words if word in spanish_indicators)
        french_score = sum(1 for word in words if word in french_indicators)
        
        scores = {
            'english': english_score,
            'spanish': spanish_score,
            'french': french_score
        }
        
        max_score = max(scores.values())
        if max_score == 0:
            return 'unknown'
        
        detected_language = max(scores, key=scores.get)
        
        # Require at least 10% of words to match for confidence
        confidence_threshold = len(words) * 0.1
        if max_score >= confidence_threshold:
            return detected_language
        else:
            return 'unknown'
    
    def extract_questions(self, content: str) -> List[str]:
        """Extract questions from the message"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content)
        
        questions = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and ('?' in sentence or 
                           re.search(r'\b(what|how|why|when|where|which|who|can|could|would|will|is|are|do|does)\b', 
                                   sentence, re.IGNORECASE)):
                questions.append(sentence.rstrip('?') + '?')
        
        return questions
    
    def extract_keywords(self, content: str, min_length: int = 3, max_keywords: int = 10) -> List[str]:
        """Extract important keywords from the message"""
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', content.lower())
        words = cleaned.split()
        
        # Filter out stop words and short words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
        
        # Filter and count words
        filtered_words = [word for word in words 
                         if len(word) >= min_length and word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:max_keywords]]
    
    def calculate_readability_score(self, content: str) -> float:
        """Calculate a simple readability score (0-100, higher is more readable)"""
        if not content:
            return 0
        
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0
        
        # Simple metric based on average sentence length
        avg_sentence_length = words / sentences
        
        # Score inversely related to sentence length
        # Sentences with 15-20 words are considered optimal
        if 15 <= avg_sentence_length <= 20:
            return 100
        elif avg_sentence_length < 15:
            return 80 + (avg_sentence_length / 15) * 20
        else:
            return max(20, 100 - (avg_sentence_length - 20) * 2)
    
    def suggest_response_tone(self, processed_message: ProcessedMessage) -> str:
        """Suggest appropriate response tone based on message analysis"""
        sentiment = processed_message.sentiment
        urgency_indicators = processed_message.urgency_indicators
        intent = processed_message.intent
        
        # Determine tone based on analysis
        if sentiment == 'urgent' or urgency_indicators:
            return 'urgent_professional'
        elif sentiment == 'negative':
            return 'empathetic_helpful'
        elif sentiment == 'positive':
            return 'friendly_appreciative'
        elif intent == 'complaint':
            return 'understanding_solution_focused'
        elif intent == 'question':
            return 'informative_clear'
        elif intent == 'greeting':
            return 'warm_welcoming'
        else:
            return 'professional_helpful'


class MessageFormatter:
    """Format messages for different outputs"""
    
    @staticmethod
    def format_for_display(content: str, max_length: int = 100) -> str:
        """Format message for display in UI lists"""
        if len(content) <= max_length:
            return content
        
        # Truncate and add ellipsis
        return content[:max_length-3] + "..."
    
    @staticmethod
    def format_for_search(content: str) -> str:
        """Format message content for search indexing"""
        # Remove punctuation and extra spaces
        cleaned = re.sub(r'[^\w\s]', ' ', content)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.lower().strip()
    
    @staticmethod
    def highlight_entities(content: str, entities: Dict[str, List[str]]) -> str:
        """Highlight entities in content for display"""
        highlighted = content
        
        # Highlight emails
        if 'email' in entities:
            for email in entities['email']:
                highlighted = highlighted.replace(email, f"**{email}**")
        
        # Highlight phone numbers
        if 'phone' in entities:
            for phone in entities['phone']:
                highlighted = highlighted.replace(phone, f"**{phone}**")
        
        # Highlight URLs
        if 'url' in entities:
            for url in entities['url']:
                highlighted = highlighted.replace(url, f"[{url}]({url})")
        
        return highlighted
    
    @staticmethod
    def create_summary(processed_message: ProcessedMessage) -> str:
        """Create a summary of the processed message"""
        summary_parts = []
        
        # Basic info
        summary_parts.append(f"Length: {processed_message.word_count} words")
        
        # Sentiment
        if processed_message.sentiment and processed_message.sentiment != 'neutral':
            summary_parts.append(f"Sentiment: {processed_message.sentiment}")
        
        # Intent
        if processed_message.intent:
            summary_parts.append(f"Intent: {processed_message.intent}")
        
        # Urgency
        if processed_message.urgency_indicators:
            summary_parts.append(f"Urgency indicators: {len(processed_message.urgency_indicators)}")
        
        # Entities
        entity_count = sum(len(entities) for entities in processed_message.entities.values())
        if entity_count > 0:
            summary_parts.append(f"Entities found: {entity_count}")
        
        return " | ".join(summary_parts)
