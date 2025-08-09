"""Configuration management for Customer Support Agent"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


class Config:
    """Configuration management for the Customer Support Agent"""
    
    # API Keys - Using shared Google API key from parent project
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    
    # Database Configuration
    DATABASE_URL: str = os.getenv(
        'DATABASE_URL', 
        'sqlite:///customer_support.db'
    )
    
    # Redis Configuration for Short-term Memory
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_PASSWORD: Optional[str] = os.getenv('REDIS_PASSWORD')
    
    # Agent Configuration
    MAX_MESSAGES: int = int(os.getenv('MAX_MESSAGES', '10'))
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    ESCALATION_TIMEOUT: int = int(os.getenv('ESCALATION_TIMEOUT', '86400'))  # 24 hours
    AUTO_RESPONSE_ENABLED: bool = os.getenv('AUTO_RESPONSE_ENABLED', 'true').lower() == 'true'
    
    # Memory Configuration
    SHORT_TERM_TTL: int = int(os.getenv('SHORT_TERM_TTL', '3600'))  # 1 hour
    LONG_TERM_RETENTION_DAYS: int = int(os.getenv('LONG_TERM_RETENTION_DAYS', '365'))
    MESSAGE_CACHE_LIMIT: int = int(os.getenv('MESSAGE_CACHE_LIMIT', '10'))
    
    # UI Configuration
    UI_PORT: int = int(os.getenv('UI_PORT', '8502'))  # Different from legal agent
    UI_HOST: str = os.getenv('UI_HOST', '0.0.0.0')
    DEBUG_MODE: bool = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # Security Configuration
    JWT_SECRET: str = os.getenv('JWT_SECRET', 'customer-support-secret-key-change-in-production')
    ENCRYPTION_KEY: Optional[str] = os.getenv('ENCRYPTION_KEY')
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '10'))
    RATE_LIMIT_WINDOW: int = int(os.getenv('RATE_LIMIT_WINDOW', '60'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: Optional[str] = os.getenv('LOG_FILE')
    ENABLE_ANALYTICS: bool = os.getenv('ENABLE_ANALYTICS', 'true').lower() == 'true'
    
    # Business Logic Configuration
    BUSINESS_HOURS_START: int = int(os.getenv('BUSINESS_HOURS_START', '9'))
    BUSINESS_HOURS_END: int = int(os.getenv('BUSINESS_HOURS_END', '17'))
    TIMEZONE: str = os.getenv('TIMEZONE', 'UTC')
    
    # Escalation Configuration
    AUTO_ESCALATE_COMPLEXITY: float = float(os.getenv('AUTO_ESCALATE_COMPLEXITY', '0.8'))
    AUTO_ESCALATE_KEYWORDS: list = os.getenv(
        'AUTO_ESCALATE_KEYWORDS', 
        'human,manager,supervisor,escalate'
    ).split(',')
    
    # Feature Flags
    ENABLE_SENTIMENT_ANALYSIS: bool = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true').lower() == 'true'
    ENABLE_INTENT_CLASSIFICATION: bool = os.getenv('ENABLE_INTENT_CLASSIFICATION', 'true').lower() == 'true'
    ENABLE_HUMAN_HANDOFF: bool = os.getenv('ENABLE_HUMAN_HANDOFF', 'true').lower() == 'true'
    ENABLE_METRICS_COLLECTION: bool = os.getenv('ENABLE_METRICS_COLLECTION', 'true').lower() == 'true'
    
    # Model Configuration
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'gemini-1.5-flash')
    LLM_TEMPERATURE: float = float(os.getenv('LLM_TEMPERATURE', '0.3'))
    LLM_MAX_TOKENS: int = int(os.getenv('LLM_MAX_TOKENS', '2048'))
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'models/embedding-001')
    
    # Notification Configuration
    EMAIL_ENABLED: bool = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
    EMAIL_HOST: Optional[str] = os.getenv('EMAIL_HOST')
    EMAIL_PORT: int = int(os.getenv('EMAIL_PORT', '587'))
    EMAIL_USERNAME: Optional[str] = os.getenv('EMAIL_USERNAME')
    EMAIL_PASSWORD: Optional[str] = os.getenv('EMAIL_PASSWORD')
    
    SLACK_ENABLED: bool = os.getenv('SLACK_ENABLED', 'false').lower() == 'true'
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv('SLACK_WEBHOOK_URL')
    SLACK_CHANNEL: str = os.getenv('SLACK_CHANNEL', '#customer-support')
    
    # Analytics Configuration
    ANALYTICS_PROVIDER: str = os.getenv('ANALYTICS_PROVIDER', 'internal')
    GOOGLE_ANALYTICS_ID: Optional[str] = os.getenv('GOOGLE_ANALYTICS_ID')
    MIXPANEL_TOKEN: Optional[str] = os.getenv('MIXPANEL_TOKEN')
    
    @classmethod
    def validate(cls) -> Dict[str, str]:
        """Validate required configuration and return any errors"""
        errors = {}
        
        # Check required API keys
        if not cls.GOOGLE_API_KEY:
            errors['GOOGLE_API_KEY'] = 'Google API key is required for LLM functionality'
        
        # Validate numeric configurations
        try:
            if cls.CONFIDENCE_THRESHOLD < 0 or cls.CONFIDENCE_THRESHOLD > 1:
                errors['CONFIDENCE_THRESHOLD'] = 'Confidence threshold must be between 0 and 1'
        except (ValueError, TypeError):
            errors['CONFIDENCE_THRESHOLD'] = 'Confidence threshold must be a valid number'
        
        try:
            if cls.MAX_MESSAGES < 1:
                errors['MAX_MESSAGES'] = 'Max messages must be at least 1'
        except (ValueError, TypeError):
            errors['MAX_MESSAGES'] = 'Max messages must be a valid integer'
        
        try:
            if cls.UI_PORT < 1 or cls.UI_PORT > 65535:
                errors['UI_PORT'] = 'UI port must be between 1 and 65535'
        except (ValueError, TypeError):
            errors['UI_PORT'] = 'UI port must be a valid integer'
        
        # Validate business hours
        try:
            if not (0 <= cls.BUSINESS_HOURS_START <= 23):
                errors['BUSINESS_HOURS_START'] = 'Business hours start must be between 0 and 23'
            if not (0 <= cls.BUSINESS_HOURS_END <= 23):
                errors['BUSINESS_HOURS_END'] = 'Business hours end must be between 0 and 23'
            if cls.BUSINESS_HOURS_START >= cls.BUSINESS_HOURS_END:
                errors['BUSINESS_HOURS'] = 'Business hours start must be before end'
        except (ValueError, TypeError):
            errors['BUSINESS_HOURS'] = 'Business hours must be valid integers'
        
        return errors
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'url': cls.DATABASE_URL,
            'retention_days': cls.LONG_TERM_RETENTION_DAYS,
            'echo': cls.DEBUG_MODE
        }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration"""
        config = {
            'url': cls.REDIS_URL,
            'ttl': cls.SHORT_TERM_TTL
        }
        if cls.REDIS_PASSWORD:
            config['password'] = cls.REDIS_PASSWORD
        return config
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            'model': cls.LLM_MODEL,
            'temperature': cls.LLM_TEMPERATURE,
            'max_tokens': cls.LLM_MAX_TOKENS,
            'api_key': cls.GOOGLE_API_KEY
        }
    
    @classmethod
    def get_agent_config(cls) -> Dict[str, Any]:
        """Get agent-specific configuration"""
        return {
            'max_messages': cls.MAX_MESSAGES,
            'confidence_threshold': cls.CONFIDENCE_THRESHOLD,
            'escalation_timeout': cls.ESCALATION_TIMEOUT,
            'auto_response_enabled': cls.AUTO_RESPONSE_ENABLED,
            'auto_escalate_complexity': cls.AUTO_ESCALATE_COMPLEXITY,
            'auto_escalate_keywords': cls.AUTO_ESCALATE_KEYWORDS
        }
    
    @classmethod
    def get_ui_config(cls) -> Dict[str, Any]:
        """Get UI configuration"""
        return {
            'port': cls.UI_PORT,
            'host': cls.UI_HOST,
            'debug': cls.DEBUG_MODE
        }
    
    @classmethod
    def get_security_config(cls) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'jwt_secret': cls.JWT_SECRET,
            'encryption_key': cls.ENCRYPTION_KEY,
            'rate_limit_per_minute': cls.RATE_LIMIT_PER_MINUTE,
            'rate_limit_window': cls.RATE_LIMIT_WINDOW
        }
    
    @classmethod
    def get_feature_flags(cls) -> Dict[str, bool]:
        """Get feature flags"""
        return {
            'sentiment_analysis': cls.ENABLE_SENTIMENT_ANALYSIS,
            'intent_classification': cls.ENABLE_INTENT_CLASSIFICATION,
            'human_handoff': cls.ENABLE_HUMAN_HANDOFF,
            'metrics_collection': cls.ENABLE_METRICS_COLLECTION,
            'analytics': cls.ENABLE_ANALYTICS,
            'email_notifications': cls.EMAIL_ENABLED,
            'slack_notifications': cls.SLACK_ENABLED
        }
    
    @classmethod
    def is_business_hours(cls) -> bool:
        """Check if current time is within business hours"""
        from datetime import datetime
        import pytz
        
        try:
            tz = pytz.timezone(cls.TIMEZONE)
            current_time = datetime.now(tz)
            current_hour = current_time.hour
            
            return cls.BUSINESS_HOURS_START <= current_hour < cls.BUSINESS_HOURS_END
        except Exception:
            return True  # Default to business hours if timezone check fails
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        config_dict = {}
        
        for attr_name in dir(cls):
            if (not attr_name.startswith('_') and 
                not callable(getattr(cls, attr_name)) and
                attr_name.isupper()):
                
                value = getattr(cls, attr_name)
                
                # Mask sensitive information
                if any(sensitive in attr_name.lower() for sensitive in ['key', 'secret', 'password', 'token']):
                    if value:
                        config_dict[attr_name] = f"{'*' * (len(str(value)) - 4)}{str(value)[-4:]}"
                    else:
                        config_dict[attr_name] = None
                else:
                    config_dict[attr_name] = value
        
        return config_dict


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG_MODE = True
    LOG_LEVEL = 'DEBUG'
    ESCALATION_TIMEOUT = 300  # 5 minutes for testing
    SHORT_TERM_TTL = 1800  # 30 minutes


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG_MODE = False
    LOG_LEVEL = 'WARNING'
    RATE_LIMIT_PER_MINUTE = 20  # Higher limit for production
    
    @classmethod
    def validate(cls) -> Dict[str, str]:
        """Additional production validations"""
        errors = super().validate()
        
        # Production-specific validations
        if not cls.ENCRYPTION_KEY:
            errors['ENCRYPTION_KEY'] = 'Encryption key is required in production'
        
        if cls.JWT_SECRET == 'customer-support-secret-key-change-in-production':
            errors['JWT_SECRET'] = 'Default JWT secret must be changed in production'
        
        if cls.DATABASE_URL.startswith('sqlite'):
            errors['DATABASE_URL'] = 'SQLite not recommended for production use'
        
        return errors


class TestConfig(Config):
    """Test-specific configuration"""
    DATABASE_URL = 'sqlite:///:memory:'
    REDIS_URL = 'redis://localhost:6379/1'  # Use different Redis DB for tests
    SHORT_TERM_TTL = 60  # 1 minute for tests
    ESCALATION_TIMEOUT = 60  # 1 minute for tests
    ENABLE_ANALYTICS = False
    EMAIL_ENABLED = False
    SLACK_ENABLED = False


def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    if env == 'production':
        return ProductionConfig
    elif env == 'test':
        return TestConfig
    else:
        return DevelopmentConfig


# Global configuration instance
config = get_config()
