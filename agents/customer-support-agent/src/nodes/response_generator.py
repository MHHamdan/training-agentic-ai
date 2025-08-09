"""Response generation node for customer support agent"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from ..agents.state import AgentState, StateConstants


class ResponseGenerator:
    """Generate AI responses for customer queries"""
    
    def __init__(self, llm=None):
        self.llm = llm  # Will be injected from main agent
        self.response_templates = self._initialize_templates()
        self.common_responses = self._initialize_common_responses()
    
    def generate_response(self, state: AgentState) -> AgentState:
        """Main response generation function"""
        # Skip generation if escalated
        if state.requires_human:
            return state
        
        # Check for common patterns first
        response = self._check_common_responses(state)
        
        if not response:
            # Generate AI response if no common pattern matched
            response = self._generate_ai_response(state)
        
        # Add response to messages
        response_msg = {
            'role': StateConstants.ROLE_AGENT,
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'generated_by': 'ai',
                'confidence_score': state.metadata.get('response_confidence', 0.8),
                'category': state.query_category,
                'processing_time': self._calculate_processing_time(state)
            }
        }
        
        state.messages.append(response_msg)
        state.response_generated_at = datetime.now()
        
        return state
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize response templates for different scenarios"""
        return {
            'greeting': "Hello! I'm here to help you with any questions or issues you might have. How can I assist you today?",
            
            'clarification_needed': "I'd be happy to help you with that. Could you provide a bit more detail about {specific_aspect} so I can give you the most accurate assistance?",
            
            'technical_issue': "I understand you're experiencing a technical issue. Let me help you troubleshoot this step by step. {specific_guidance}",
            
            'billing_inquiry': "I can help you with your billing question. {specific_response} If you need more detailed information, I can connect you with our billing team.",
            
            'account_assistance': "I'll help you with your account issue. For security purposes, I need to verify some information first. {verification_request}",
            
            'feature_request': "Thank you for your suggestion! {acknowledgment} Feature requests like yours help us improve our service. {next_steps}",
            
            'escalation_notice': "I understand this is a complex issue. I'm connecting you with one of our human agents who can provide more specialized assistance. {timeline_info}",
            
            'follow_up': "Is there anything else I can help you with today? I'm here to ensure all your questions are answered.",
            
            'appreciation': "Thank you for contacting us! {specific_thanks} We appreciate your business and are always here to help."
        }
    
    def _initialize_common_responses(self) -> Dict[str, Dict[str, Any]]:
        """Initialize common responses for frequent queries"""
        return {
            'password_reset': {
                'keywords': ['password', 'reset', 'forgot', 'login'],
                'response': """To reset your password:
1. Go to the login page
2. Click on "Forgot Password?"
3. Enter your email address
4. Check your email for reset instructions
5. Follow the link to create a new password

If you don't receive the email within 5 minutes, please check your spam folder or contact us for further assistance.""",
                'confidence': 0.9
            },
            
            'billing_general': {
                'keywords': ['billing', 'invoice', 'payment', 'subscription'],
                'response': """For billing inquiries, you can:
• View your current subscription and billing history in Account Settings
• Download invoices from the Billing section
• Update payment methods in your account
• Contact billing support for disputes or questions

Is there a specific billing issue I can help you with?""",
                'confidence': 0.8
            },
            
            'technical_support': {
                'keywords': ['not working', 'error', 'bug', 'issue'],
                'response': """I'm sorry you're experiencing technical difficulties. To help me assist you better, could you please provide:
• What exactly isn't working?
• When did this issue start?
• What error messages are you seeing?
• What browser/device are you using?

This information will help me provide more targeted assistance.""",
                'confidence': 0.7
            },
            
            'account_access': {
                'keywords': ['access', 'login', 'account', 'username'],
                'response': """For account access issues:
• Verify you're using the correct email address
• Try resetting your password
• Clear your browser cache and cookies
• Disable browser extensions temporarily
• Try accessing from a different browser

If these steps don't resolve the issue, I can help you troubleshoot further.""",
                'confidence': 0.8
            },
            
            'feature_information': {
                'keywords': ['how to', 'feature', 'functionality', 'use'],
                'response': """I'd be happy to help you learn about our features! You can:
• Check our Help Center for detailed guides
• Watch tutorial videos in the Resources section
• Access step-by-step instructions in the app
• Contact support for personalized assistance

What specific feature would you like to learn more about?""",
                'confidence': 0.7
            }
        }
    
    def _check_common_responses(self, state: AgentState) -> Optional[str]:
        """Check if query matches common response patterns"""
        if not state.current_query:
            return None
        
        query_lower = state.current_query.lower()
        
        # Check against common response patterns
        for response_type, config in self.common_responses.items():
            keywords = config['keywords']
            
            # Count matching keywords
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            
            # If enough keywords match, use the common response
            if matches >= len(keywords) * 0.5:  # At least 50% of keywords match
                # Update confidence in metadata
                state.metadata['response_confidence'] = config['confidence']
                state.metadata['response_type'] = 'common_pattern'
                state.metadata['pattern_matched'] = response_type
                
                return config['response']
        
        return None
    
    def _generate_ai_response(self, state: AgentState) -> str:
        """Generate AI response using LLM"""
        if not self.llm:
            return self._generate_fallback_response(state)
        
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(state)
            
            # Create prompt
            prompt = self._create_response_prompt(state, context)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Update metadata
            state.metadata['response_confidence'] = 0.8  # Default confidence for LLM
            state.metadata['response_type'] = 'llm_generated'
            
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Error generating AI response: {e}")
            state.metadata['response_error'] = str(e)
            return self._generate_fallback_response(state)
    
    def _prepare_llm_context(self, state: AgentState) -> str:
        """Prepare context for LLM"""
        context_parts = []
        
        # User profile context
        if state.user_profile:
            context_parts.append(
                f"User: {state.user_profile.name} ({state.user_profile.account_type} account)"
            )
        
        # Query classification
        if state.query_category:
            context_parts.append(f"Query category: {state.query_category}")
        
        # Conversation history (last few messages)
        if state.messages:
            recent_messages = state.messages[-3:]  # Last 3 messages for context
            for msg in recent_messages:
                if msg['role'] in [StateConstants.ROLE_USER, StateConstants.ROLE_AGENT]:
                    context_parts.append(f"{msg['role']}: {msg['content']}")
        
        # User history insights
        if state.metadata.get('user_insights'):
            insights = state.metadata['user_insights']
            if insights.get('is_frequent_user'):
                context_parts.append("Note: This is a frequent user")
            if insights.get('user_type') == 'technical':
                context_parts.append("Note: User tends to ask technical questions")
        
        # Relevant historical context
        if state.metadata.get('history_summary'):
            context_parts.append(f"User history: {state.metadata['history_summary']}")
        
        return "\n".join(context_parts)
    
    def _create_response_prompt(self, state: AgentState, context: str) -> str:
        """Create prompt for LLM response generation"""
        prompt = f"""You are a helpful customer support agent for TechTrend Innovations. 
        
Context:
{context}

Current Query: {state.current_query}

Instructions:
- Provide a helpful, professional, and empathetic response
- Keep responses concise but complete
- If you need more information, ask specific clarifying questions
- Offer concrete next steps when possible
- Be proactive in suggesting solutions
- Maintain a friendly but professional tone
- If the issue seems complex, suggest escalation to human support

Response:"""
        
        return prompt
    
    def _generate_fallback_response(self, state: AgentState) -> str:
        """Generate fallback response when LLM is unavailable"""
        category = state.query_category or 'general'
        
        fallback_responses = {
            'technical': "I understand you're experiencing a technical issue. While I work on getting you the best assistance, could you provide more details about the specific problem you're encountering?",
            
            'billing': "I see you have a billing question. I want to make sure you get accurate information. Could you specify what billing aspect you need help with?",
            
            'account': "I'm here to help with your account. For security reasons, I may need to verify some information. What specific account issue can I assist you with?",
            
            'feature': "Thank you for your question about our features. I'd like to provide you with the most relevant information. Could you tell me more about what you're trying to accomplish?",
            
            'general': "I'm here to help! To provide you with the best assistance, could you give me a bit more detail about your question or concern?"
        }
        
        # Update metadata
        state.metadata['response_confidence'] = 0.6
        state.metadata['response_type'] = 'fallback'
        
        return fallback_responses.get(category, fallback_responses['general'])
    
    def _calculate_processing_time(self, state: AgentState) -> Optional[float]:
        """Calculate total processing time"""
        if state.processing_start_time and state.response_generated_at:
            delta = state.response_generated_at - state.processing_start_time
            return delta.total_seconds()
        return None
    
    def personalize_response(self, response: str, state: AgentState) -> str:
        """Personalize response based on user profile and context"""
        personalized = response
        
        # Add user name if available
        if state.user_profile and state.user_profile.name:
            # Check if response doesn't already contain name
            if state.user_profile.name.lower() not in response.lower():
                personalized = f"Hi {state.user_profile.name}! {response}"
        
        # Adjust tone for account type
        if state.user_profile:
            if state.user_profile.account_type == 'enterprise':
                # More formal tone for enterprise customers
                personalized = personalized.replace("I'd", "I would")
                personalized = personalized.replace("can't", "cannot")
            elif state.user_profile.account_type == 'premium':
                # Add premium acknowledgment if appropriate
                if 'priority' not in personalized.lower():
                    personalized += " As a premium customer, your request will receive priority handling."
        
        return personalized
    
    def add_helpful_resources(self, response: str, category: str) -> str:
        """Add helpful resources based on category"""
        resources = {
            'technical': "\n\nAdditional Resources:\n• Check our Status Page for known issues\n• Visit our Troubleshooting Guide\n• Browse common solutions in our Help Center",
            
            'billing': "\n\nAdditional Resources:\n• View billing FAQs\n• Access your account dashboard\n• Download invoices and receipts",
            
            'feature': "\n\nAdditional Resources:\n• Explore our Feature Documentation\n• Watch tutorial videos\n• Join our user community for tips",
            
            'account': "\n\nAdditional Resources:\n• Security best practices guide\n• Account settings tutorial\n• Two-factor authentication setup"
        }
        
        if category in resources:
            return response + resources[category]
        
        return response


# Function to use in LangGraph node
def generate_response_node(state: AgentState) -> AgentState:
    """LangGraph node function for response generation"""
    # This would typically receive the LLM instance from the main agent
    generator = ResponseGenerator()
    return generator.generate_response(state)
