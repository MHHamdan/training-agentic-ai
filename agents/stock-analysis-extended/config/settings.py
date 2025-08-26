"""Configuration management for the extended stock analysis system"""

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, Dict, List, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

class APISettings(BaseSettings):
    """API configuration settings"""
    
    # LLM APIs
    openai_api_key: Optional[str] = Field(None, env='OPENAI_API_KEY')
    google_api_key: Optional[str] = Field(None, env='GOOGLE_API_KEY')
    groq_api_key: Optional[str] = Field(None, env='GROQ_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, env='ANTHROPIC_API_KEY')
    
    # Financial Data APIs
    alpha_vantage_api_key: Optional[str] = Field(None, env='ALPHA_VANTAGE_API_KEY')
    finnhub_api_key: Optional[str] = Field(None, env='FINNHUB_API_KEY')
    polygon_api_key: Optional[str] = Field(None, env='POLYGON_API_KEY')
    fred_api_key: Optional[str] = Field(None, env='FRED_API_KEY')
    quandl_api_key: Optional[str] = Field(None, env='QUANDL_API_KEY')
    
    # News and Sentiment APIs
    news_api_key: Optional[str] = Field(None, env='NEWS_API_KEY')
    twitter_bearer_token: Optional[str] = Field(None, env='TWITTER_BEARER_TOKEN')
    reddit_client_id: Optional[str] = Field(None, env='REDDIT_CLIENT_ID')
    reddit_client_secret: Optional[str] = Field(None, env='REDDIT_CLIENT_SECRET')
    
    class Config:
        env_file = '.env'
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL
    postgres_host: str = Field('localhost', env='POSTGRES_HOST')
    postgres_port: int = Field(5432, env='POSTGRES_PORT')
    postgres_db: str = Field('stock_analysis', env='POSTGRES_DB')
    postgres_user: str = Field('postgres', env='POSTGRES_USER')
    postgres_password: str = Field('', env='POSTGRES_PASSWORD')
    
    # MongoDB
    mongodb_uri: str = Field('mongodb://localhost:27017', env='MONGODB_URI')
    mongodb_db: str = Field('stock_analysis', env='MONGODB_DB')
    
    # Redis
    redis_host: str = Field('localhost', env='REDIS_HOST')
    redis_port: int = Field(6379, env='REDIS_PORT')
    redis_db: int = Field(0, env='REDIS_DB')
    redis_password: Optional[str] = Field(None, env='REDIS_PASSWORD')
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class AgentSettings(BaseSettings):
    """Agent configuration settings"""
    
    # LLM Model Settings
    default_llm_provider: str = Field('openai', env='DEFAULT_LLM_PROVIDER')
    default_model: str = Field('gpt-4-turbo-preview', env='DEFAULT_MODEL')
    temperature: float = Field(0.7, env='LLM_TEMPERATURE')
    max_tokens: int = Field(2048, env='LLM_MAX_TOKENS')
    
    # Agent Behavior
    max_retries: int = Field(3, env='MAX_RETRIES')
    timeout_seconds: int = Field(300, env='TIMEOUT_SECONDS')
    enable_memory: bool = Field(True, env='ENABLE_MEMORY')
    enable_tools: bool = Field(True, env='ENABLE_TOOLS')
    
    # Workflow Settings
    max_parallel_agents: int = Field(5, env='MAX_PARALLEL_AGENTS')
    enable_human_in_loop: bool = Field(True, env='ENABLE_HUMAN_IN_LOOP')
    escalation_threshold: float = Field(0.3, env='ESCALATION_THRESHOLD')
    
    # Analysis Settings
    default_analysis_period: str = Field('1mo', env='DEFAULT_ANALYSIS_PERIOD')
    technical_indicators: List[str] = Field(
        default_factory=lambda: ['RSI', 'MACD', 'BB', 'SMA', 'EMA', 'VWAP'],
        env='TECHNICAL_INDICATORS'
    )
    risk_metrics: List[str] = Field(
        default_factory=lambda: ['sharpe', 'sortino', 'alpha', 'beta', 'var', 'cvar'],
        env='RISK_METRICS'
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration"""
    
    # Logging
    log_level: str = Field('INFO', env='LOG_LEVEL')
    log_format: str = Field('json', env='LOG_FORMAT')
    log_file: Optional[str] = Field(None, env='LOG_FILE')
    
    # Metrics
    enable_metrics: bool = Field(True, env='ENABLE_METRICS')
    metrics_port: int = Field(8000, env='METRICS_PORT')
    
    # Tracing
    enable_tracing: bool = Field(False, env='ENABLE_TRACING')
    jaeger_host: str = Field('localhost', env='JAEGER_HOST')
    jaeger_port: int = Field(6831, env='JAEGER_PORT')
    
    # Sentry
    sentry_dsn: Optional[str] = Field(None, env='SENTRY_DSN')
    sentry_environment: str = Field('development', env='SENTRY_ENVIRONMENT')


class NotificationSettings(BaseSettings):
    """Notification configuration"""
    
    # Email
    smtp_host: Optional[str] = Field(None, env='SMTP_HOST')
    smtp_port: int = Field(587, env='SMTP_PORT')
    smtp_user: Optional[str] = Field(None, env='SMTP_USER')
    smtp_password: Optional[str] = Field(None, env='SMTP_PASSWORD')
    smtp_use_tls: bool = Field(True, env='SMTP_USE_TLS')
    
    # Slack
    slack_webhook_url: Optional[str] = Field(None, env='SLACK_WEBHOOK_URL')
    slack_channel: str = Field('#stock-alerts', env='SLACK_CHANNEL')
    
    # SMS (Twilio)
    twilio_account_sid: Optional[str] = Field(None, env='TWILIO_ACCOUNT_SID')
    twilio_auth_token: Optional[str] = Field(None, env='TWILIO_AUTH_TOKEN')
    twilio_from_number: Optional[str] = Field(None, env='TWILIO_FROM_NUMBER')


class Settings(BaseSettings):
    """Main settings aggregator"""
    
    # Application settings
    app_name: str = Field('Stock Analysis Extended', env='APP_NAME')
    app_version: str = Field('1.0.0', env='APP_VERSION')
    environment: str = Field('development', env='ENVIRONMENT')
    debug: bool = Field(True, env='DEBUG')
    
    # Component settings
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    
    # Paths
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / 'data')
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / 'reports')
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / 'logs')
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = '.env'
        case_sensitive = False


# Global settings instance
settings = Settings()