"""Input validation utilities for Customer Support Agent"""

import re
import email.utils
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class ValidationError(Exception):
    """Custom validation error"""
    pass


class InputValidator:
    """Validate user inputs and system data"""
    
    @staticmethod
    def validate_email(email_address: str) -> Tuple[bool, str]:
        """Validate email address format"""
        if not email_address:
            return False, "Email address is required"
        
        if len(email_address) > 254:
            return False, "Email address is too long"
        
        # Use email.utils for validation
        parsed = email.utils.parseaddr(email_address)
        if not parsed[1] or '@' not in parsed[1]:
            return False, "Invalid email format"
        
        # Additional regex check
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email_address):
            return False, "Invalid email format"
        
        return True, "Valid email"
    
    @staticmethod
    def validate_user_id(user_id: str) -> Tuple[bool, str]:
        """Validate user ID format"""
        if not user_id:
            return False, "User ID is required"
        
        if not isinstance(user_id, str):
            return False, "User ID must be a string"
        
        if len(user_id) < 3:
            return False, "User ID must be at least 3 characters"
        
        if len(user_id) > 100:
            return False, "User ID is too long"
        
        # Allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', user_id):
            return False, "User ID contains invalid characters"
        
        return True, "Valid user ID"
    
    @staticmethod
    def validate_message_content(content: str) -> Tuple[bool, str]:
        """Validate message content"""
        if not content:
            return False, "Message content cannot be empty"
        
        if not isinstance(content, str):
            return False, "Message content must be a string"
        
        # Remove whitespace for length check
        trimmed_content = content.strip()
        if not trimmed_content:
            return False, "Message content cannot be only whitespace"
        
        if len(trimmed_content) > 10000:
            return False, "Message content is too long (max 10,000 characters)"
        
        # Check for suspicious patterns (basic XSS prevention)
        suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>'
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, "Message content contains potentially malicious code"
        
        return True, "Valid message content"
    
    @staticmethod
    def validate_thread_id(thread_id: str) -> Tuple[bool, str]:
        """Validate thread ID format"""
        if not thread_id:
            return False, "Thread ID is required"
        
        if not isinstance(thread_id, str):
            return False, "Thread ID must be a string"
        
        # Should be UUID format or similar
        uuid_pattern = r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$'
        if not re.match(uuid_pattern, thread_id):
            return False, "Thread ID must be in UUID format"
        
        return True, "Valid thread ID"
    
    @staticmethod
    def validate_session_id(session_id: str) -> Tuple[bool, str]:
        """Validate session ID format"""
        if not session_id:
            return False, "Session ID is required"
        
        if not isinstance(session_id, str):
            return False, "Session ID must be a string"
        
        if len(session_id) < 8:
            return False, "Session ID must be at least 8 characters"
        
        if len(session_id) > 128:
            return False, "Session ID is too long"
        
        # Allow alphanumeric and hyphens
        if not re.match(r'^[a-zA-Z0-9-]+$', session_id):
            return False, "Session ID contains invalid characters"
        
        return True, "Valid session ID"
    
    @staticmethod
    def validate_confidence_score(score: float) -> Tuple[bool, str]:
        """Validate confidence score"""
        if score is None:
            return False, "Confidence score is required"
        
        if not isinstance(score, (int, float)):
            return False, "Confidence score must be a number"
        
        if not 0 <= score <= 1:
            return False, "Confidence score must be between 0 and 1"
        
        return True, "Valid confidence score"
    
    @staticmethod
    def validate_satisfaction_rating(rating: int) -> Tuple[bool, str]:
        """Validate user satisfaction rating"""
        if rating is None:
            return True, "Rating is optional"  # Rating can be None
        
        if not isinstance(rating, int):
            return False, "Rating must be an integer"
        
        if not 1 <= rating <= 5:
            return False, "Rating must be between 1 and 5"
        
        return True, "Valid rating"
    
    @staticmethod
    def validate_category(category: str) -> Tuple[bool, str]:
        """Validate query category"""
        if not category:
            return False, "Category is required"
        
        if not isinstance(category, str):
            return False, "Category must be a string"
        
        valid_categories = [
            'technical', 'billing', 'feature', 'account', 'general'
        ]
        
        if category.lower() not in valid_categories:
            return False, f"Category must be one of: {', '.join(valid_categories)}"
        
        return True, "Valid category"
    
    @staticmethod
    def validate_priority(priority: str) -> Tuple[bool, str]:
        """Validate priority level"""
        if not priority:
            return False, "Priority is required"
        
        if not isinstance(priority, str):
            return False, "Priority must be a string"
        
        valid_priorities = ['low', 'normal', 'high', 'urgent']
        
        if priority.lower() not in valid_priorities:
            return False, f"Priority must be one of: {', '.join(valid_priorities)}"
        
        return True, "Valid priority"
    
    @staticmethod
    def validate_account_type(account_type: str) -> Tuple[bool, str]:
        """Validate account type"""
        if not account_type:
            return False, "Account type is required"
        
        if not isinstance(account_type, str):
            return False, "Account type must be a string"
        
        valid_types = ['standard', 'premium', 'enterprise']
        
        if account_type.lower() not in valid_types:
            return False, f"Account type must be one of: {', '.join(valid_types)}"
        
        return True, "Valid account type"
    
    @staticmethod
    def validate_name(name: str) -> Tuple[bool, str]:
        """Validate user name"""
        if not name:
            return False, "Name is required"
        
        if not isinstance(name, str):
            return False, "Name must be a string"
        
        trimmed_name = name.strip()
        if not trimmed_name:
            return False, "Name cannot be only whitespace"
        
        if len(trimmed_name) < 2:
            return False, "Name must be at least 2 characters"
        
        if len(trimmed_name) > 100:
            return False, "Name is too long (max 100 characters)"
        
        # Allow letters, spaces, hyphens, apostrophes
        if not re.match(r"^[a-zA-Z\s\-'\.]+$", trimmed_name):
            return False, "Name contains invalid characters"
        
        return True, "Valid name"
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate metadata dictionary"""
        if metadata is None:
            return True, "Metadata is optional"
        
        if not isinstance(metadata, dict):
            return False, "Metadata must be a dictionary"
        
        # Check for reasonable size
        try:
            import json
            metadata_str = json.dumps(metadata)
            if len(metadata_str) > 50000:  # 50KB limit
                return False, "Metadata is too large"
        except (TypeError, ValueError):
            return False, "Metadata contains non-serializable data"
        
        # Check for suspicious keys or values
        for key, value in metadata.items():
            if not isinstance(key, str):
                return False, "Metadata keys must be strings"
            
            if len(key) > 100:
                return False, "Metadata key is too long"
            
            # Check for script injection in values
            if isinstance(value, str) and len(value) > 1000:
                return False, "Metadata string value is too long"
        
        return True, "Valid metadata"
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize text input by removing potentially harmful content"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove script content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: protocols
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Remove on* event handlers
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        # Limit length
        return text[:10000]
    
    @staticmethod
    def validate_phone_number(phone: str) -> Tuple[bool, str]:
        """Validate phone number format"""
        if not phone:
            return True, "Phone number is optional"
        
        # Remove common formatting characters
        cleaned_phone = re.sub(r'[\s\-\(\)\+\.]', '', phone)
        
        # Check if it's all digits after cleaning
        if not cleaned_phone.isdigit():
            return False, "Phone number must contain only digits and common formatting characters"
        
        # Check length (international format)
        if len(cleaned_phone) < 7 or len(cleaned_phone) > 15:
            return False, "Phone number must be between 7 and 15 digits"
        
        return True, "Valid phone number"
    
    @staticmethod
    def validate_url(url: str) -> Tuple[bool, str]:
        """Validate URL format"""
        if not url:
            return True, "URL is optional"
        
        # Basic URL validation
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, url, re.IGNORECASE):
            return False, "Invalid URL format"
        
        # Check for reasonable length
        if len(url) > 2048:
            return False, "URL is too long"
        
        return True, "Valid URL"


class BusinessRuleValidator:
    """Validate business rules and constraints"""
    
    @staticmethod
    def validate_escalation_rules(user_profile: Dict[str, Any], query_data: Dict[str, Any]) -> List[str]:
        """Validate escalation business rules"""
        violations = []
        
        # Premium customers get faster escalation
        if user_profile.get('account_type') == 'premium':
            max_auto_attempts = 2
        elif user_profile.get('account_type') == 'enterprise':
            max_auto_attempts = 1
        else:
            max_auto_attempts = 3
        
        # Check if we've exceeded attempts
        attempts = query_data.get('resolution_attempts', 0)
        if attempts > max_auto_attempts:
            violations.append(f"Exceeded maximum auto-resolution attempts ({max_auto_attempts})")
        
        # Business hours escalation rules
        if query_data.get('priority') == 'urgent':
            # Urgent queries should escalate immediately outside business hours
            from .config import Config
            if not Config.is_business_hours():
                violations.append("Urgent query outside business hours requires immediate escalation")
        
        return violations
    
    @staticmethod
    def validate_response_time_requirements(user_profile: Dict[str, Any], response_time: float) -> List[str]:
        """Validate response time against SLA requirements"""
        violations = []
        
        account_type = user_profile.get('account_type', 'standard')
        
        # SLA requirements (in seconds)
        sla_requirements = {
            'enterprise': 300,    # 5 minutes
            'premium': 600,       # 10 minutes
            'standard': 1800      # 30 minutes
        }
        
        max_time = sla_requirements.get(account_type, 1800)
        
        if response_time > max_time:
            violations.append(f"Response time ({response_time:.1f}s) exceeds SLA for {account_type} account ({max_time}s)")
        
        return violations
    
    @staticmethod
    def validate_data_retention(query_date: datetime) -> bool:
        """Validate if data should be retained based on date"""
        from .config import Config
        from datetime import timedelta
        
        retention_cutoff = datetime.now() - timedelta(days=Config.LONG_TERM_RETENTION_DAYS)
        return query_date >= retention_cutoff
