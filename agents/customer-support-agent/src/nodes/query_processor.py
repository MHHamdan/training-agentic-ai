"""Query processing node for customer support agent"""

import re
from typing import Dict, List, Optional
from datetime import datetime
from ..agents.state import AgentState, StateConstants


class QueryProcessor:
    """Process and classify customer queries"""
    
    def __init__(self):
        self.category_keywords = {
            StateConstants.CATEGORY_TECHNICAL: [
                'bug', 'error', 'crash', 'not working', 'broken', 'issue', 
                'problem', 'failed', 'unable', 'cannot', 'won\'t load', 'timeout'
            ],
            StateConstants.CATEGORY_BILLING: [
                'payment', 'invoice', 'subscription', 'charge', 'billing',
                'refund', 'credit', 'cost', 'price', 'fee', 'receipt'
            ],
            StateConstants.CATEGORY_FEATURE: [
                'request', 'enhancement', 'improve', 'add', 'feature',
                'functionality', 'capability', 'suggest', 'recommend'
            ],
            StateConstants.CATEGORY_ACCOUNT: [
                'password', 'login', 'access', 'reset', 'account',
                'profile', 'settings', 'username', 'email', 'security'
            ]
        }
        
        self.urgency_keywords = {
            StateConstants.PRIORITY_URGENT: [
                'urgent', 'critical', 'emergency', 'immediately', 'asap',
                'production down', 'system down', 'can\'t access'
            ],
            StateConstants.PRIORITY_HIGH: [
                'important', 'priority', 'soon', 'quickly', 'blocking',
                'broken', 'not working'
            ]
        }
        
        self.greeting_patterns = [
            r'^(hi|hello|hey|good\s+(morning|afternoon|evening))',
            r'^(thanks?|thank\s+you)',
            r'^(bye|goodbye|see\s+you)'
        ]
    
    def process_query(self, state: AgentState) -> AgentState:
        """Main query processing function"""
        if not state.messages:
            return state
        
        # Get the latest user message
        latest_message = self._get_latest_user_message(state.messages)
        if not latest_message:
            return state
        
        query_text = latest_message.get('content', '')
        state.current_query = query_text
        
        # Start processing timer
        state.processing_start_time = datetime.now()
        
        # Classify the query
        state.query_category = self._classify_query(query_text)
        
        # Assess priority
        priority = self._assess_priority(query_text)
        if state.conversation_context:
            state.conversation_context.priority = priority
        
        # Check if it's a greeting or simple acknowledgment
        if self._is_greeting(query_text):
            state.metadata['is_greeting'] = True
        
        # Extract entities and intent
        entities = self._extract_entities(query_text)
        state.metadata['entities'] = entities
        
        # Assess query complexity
        complexity_score = self._assess_complexity(query_text, entities)
        state.metadata['complexity_score'] = complexity_score
        
        return state
    
    def _get_latest_user_message(self, messages: List[Dict]) -> Optional[Dict]:
        """Get the most recent message from the user"""
        for message in reversed(messages):
            if message.get('role') == StateConstants.ROLE_USER:
                return message
        return None
    
    def _classify_query(self, query_text: str) -> str:
        """Classify query into predefined categories"""
        query_lower = query_text.lower()
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score, or general if none found
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return StateConstants.CATEGORY_GENERAL
    
    def _assess_priority(self, query_text: str) -> str:
        """Assess the priority level of the query"""
        query_lower = query_text.lower()
        
        # Check for urgent keywords
        for keyword in self.urgency_keywords[StateConstants.PRIORITY_URGENT]:
            if keyword in query_lower:
                return StateConstants.PRIORITY_URGENT
        
        # Check for high priority keywords
        for keyword in self.urgency_keywords[StateConstants.PRIORITY_HIGH]:
            if keyword in query_lower:
                return StateConstants.PRIORITY_HIGH
        
        # Default to normal priority
        return StateConstants.PRIORITY_NORMAL
    
    def _is_greeting(self, query_text: str) -> bool:
        """Check if the query is just a greeting"""
        query_lower = query_text.lower().strip()
        
        # Check against greeting patterns
        for pattern in self.greeting_patterns:
            if re.match(pattern, query_lower):
                return True
        
        # Check for simple greetings
        simple_greetings = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'bye']
        return query_lower in simple_greetings
    
    def _extract_entities(self, query_text: str) -> Dict[str, List[str]]:
        """Extract entities from the query text"""
        entities = {
            'products': [],
            'error_codes': [],
            'email_addresses': [],
            'phone_numbers': [],
            'urls': [],
            'account_ids': []
        }
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['email_addresses'] = re.findall(email_pattern, query_text)
        
        # Extract error codes (assuming format like ERR-123, ERROR123, etc.)
        error_pattern = r'\b(?:ERR|ERROR)[-_]?\d+\b'
        entities['error_codes'] = re.findall(error_pattern, query_text, re.IGNORECASE)
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities['urls'] = re.findall(url_pattern, query_text)
        
        # Extract phone numbers (basic pattern)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities['phone_numbers'] = re.findall(phone_pattern, query_text)
        
        # Extract account IDs (assuming format like ACC-123456, ACCT123456, etc.)
        account_pattern = r'\b(?:ACC|ACCT|ACCOUNT)[-_]?\d+\b'
        entities['account_ids'] = re.findall(account_pattern, query_text, re.IGNORECASE)
        
        # Product names (basic detection for common product terms)
        product_keywords = ['dashboard', 'api', 'mobile app', 'web app', 'platform', 'service']
        entities['products'] = [keyword for keyword in product_keywords 
                               if keyword.lower() in query_text.lower()]
        
        return entities
    
    def _assess_complexity(self, query_text: str, entities: Dict) -> float:
        """Assess the complexity of the query (0.0 to 1.0)"""
        complexity_score = 0.0
        
        # Base complexity on text length
        word_count = len(query_text.split())
        if word_count > 50:
            complexity_score += 0.3
        elif word_count > 20:
            complexity_score += 0.2
        elif word_count > 10:
            complexity_score += 0.1
        
        # Increase complexity for multiple questions
        question_marks = query_text.count('?')
        if question_marks > 2:
            complexity_score += 0.2
        elif question_marks > 1:
            complexity_score += 0.1
        
        # Increase complexity for technical entities
        if entities.get('error_codes') or entities.get('urls'):
            complexity_score += 0.2
        
        # Increase complexity for multiple entities
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        if total_entities > 3:
            complexity_score += 0.3
        elif total_entities > 1:
            complexity_score += 0.1
        
        # Complex keywords that might indicate difficult queries
        complex_keywords = [
            'integration', 'api', 'database', 'configuration', 'permissions',
            'security', 'migration', 'deployment', 'architecture', 'performance'
        ]
        complex_keyword_count = sum(1 for keyword in complex_keywords 
                                   if keyword.lower() in query_text.lower())
        complexity_score += complex_keyword_count * 0.1
        
        # Negative sentiment words might indicate frustrated users (complex situation)
        negative_words = [
            'frustrated', 'angry', 'terrible', 'awful', 'horrible',
            'worst', 'hate', 'useless', 'broken', 'terrible'
        ]
        negative_count = sum(1 for word in negative_words 
                           if word.lower() in query_text.lower())
        complexity_score += negative_count * 0.1
        
        # Cap the complexity score at 1.0
        return min(complexity_score, 1.0)
    
    def get_suggested_responses(self, category: str, entities: Dict) -> List[str]:
        """Get suggested responses based on category and entities"""
        suggestions = {
            StateConstants.CATEGORY_TECHNICAL: [
                "I'd be happy to help you with this technical issue. Could you provide more details about when this started happening?",
                "Let me help you troubleshoot this issue. What steps have you already tried?",
                "I see you're experiencing a technical problem. Can you share any error messages you're seeing?"
            ],
            StateConstants.CATEGORY_BILLING: [
                "I can help you with your billing inquiry. Let me look up your account information.",
                "For billing questions, I'll need to verify some account details. Could you provide your account email?",
                "I understand you have a billing concern. Let me see how I can assist you with this."
            ],
            StateConstants.CATEGORY_ACCOUNT: [
                "I can help you with your account. For security purposes, I'll need to verify your identity first.",
                "Let me assist you with your account issue. Could you confirm the email address associated with your account?",
                "I see you need help with your account. What specific account-related issue are you experiencing?"
            ],
            StateConstants.CATEGORY_FEATURE: [
                "Thank you for your feature suggestion! I'd love to hear more about what you'd like to see improved.",
                "That's a great idea! Let me make sure I understand your feature request correctly.",
                "I appreciate your feedback! Feature requests help us improve our service."
            ]
        }
        
        return suggestions.get(category, [
            "I'm here to help! Could you provide a bit more detail about what you need assistance with?",
            "Thank you for contacting support. How can I best assist you today?",
            "I'd be happy to help you. Could you tell me more about your question or concern?"
        ])


# Function to use in LangGraph node
def process_query_node(state: AgentState) -> AgentState:
    """LangGraph node function for query processing"""
    processor = QueryProcessor()
    return processor.process_query(state)
