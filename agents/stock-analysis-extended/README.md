# 🚀 Extended Stock Analysis Multi-Agent System

An advanced, scalable, and feature-rich multi-agent workflow system for comprehensive stock market analysis. This system extends the basic stock analysis implementation with sophisticated agents, real-time data processing, and enterprise-grade features.

## 🎯 Overview

This system provides comprehensive stock analysis through multiple specialized agents:

- **🎯 Technical Analysis Agent**: Advanced indicators, chart patterns, and trading signals
- **⚠️ Risk Assessment Agent**: Portfolio risk, volatility metrics, and VaR calculations
- **💭 Sentiment Analysis Agent**: News, social media, and market sentiment tracking
- **📊 Market Comparison Agent**: Peer analysis and sector performance (coming soon)
- **✅ Compliance Agent**: Regulatory and ESG considerations (coming soon)
- **🎨 Portfolio Optimization Agent**: Allocation and diversification strategies (coming soon)

## 🏗️ Architecture

### Multi-Agent Framework
- **CrewAI Integration**: Advanced agent coordination and task management
- **LangGraph Workflows**: State-based workflow orchestration
- **Parallel Processing**: Concurrent agent execution for optimal performance
- **Error Handling**: Graceful degradation and retry mechanisms

### Scalability Features
- **Configurable Workflows**: Multiple analysis types and execution modes
- **API Integration**: Multiple data sources with fallback mechanisms
- **Caching**: Redis-based caching for API rate limit optimization
- **Monitoring**: Prometheus metrics and structured logging

## 📁 Project Structure

```
stock-analysis-extended/
├── agents/                     # Agent implementations
│   ├── base.py                # Base agent class
│   ├── core/                  # Core analysis agents
│   │   ├── risk_assessor.py   # Risk assessment agent
│   │   ├── sentiment_analyzer.py # Sentiment analysis agent
│   │   └── technical_analyst.py  # Technical analysis agent
│   ├── specialized/           # Specialized agents (future)
│   └── supervisors/          # Supervisor agents (future)
├── workflows/                 # Workflow orchestration
│   ├── orchestration/
│   │   └── workflow_manager.py # Main workflow coordinator
│   └── strategies/           # Analysis strategies (future)
├── tools/                    # Analysis tools and utilities
│   ├── data_sources/         # Data source connectors
│   ├── analysis/             # Analysis utilities
│   │   └── risk_metrics.py   # Risk calculation tools
│   └── reporting/            # Report generation (future)
├── config/                   # Configuration management
│   └── settings.py           # Centralized settings
├── tests/                    # Test suite
├── deployment/               # Deployment scripts
├── data/                     # Data storage
├── reports/                  # Generated reports
├── app.py                    # Main Streamlit application
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd agents/stock-analysis-extended

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### 2. Environment Configuration

Create a `.env` file in the project root with the following keys:

```env
# LLM APIs
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
ANTHROPIC_API_KEY=your_anthropic_key

# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alphavantage_key
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key

# News and Sentiment APIs
NEWS_API_KEY=your_news_api_key
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Database (optional)
POSTGRES_HOST=localhost
POSTGRES_DB=stock_analysis
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py

# Or run individual agent tests
python -m pytest tests/
```

## 🔧 Configuration

### Agent Settings

Configure agent behavior in `config/settings.py`:

```python
class AgentSettings:
    default_llm_provider: str = 'openai'  # 'openai', 'google', 'anthropic'
    max_parallel_agents: int = 5
    timeout_seconds: int = 300
    enable_memory: bool = True
```

### Workflow Types

Available analysis workflows:

- **QUICK_SCAN**: Fast technical and sentiment analysis (~1 minute)
- **COMPREHENSIVE**: Full multi-agent analysis (~5 minutes)
- **RISK_FOCUSED**: Deep risk and volatility analysis
- **TECHNICAL_FOCUSED**: Detailed technical indicators
- **SENTIMENT_FOCUSED**: In-depth sentiment analysis

## 📊 Usage Examples

### Basic Analysis

```python
from workflows.orchestration.workflow_manager import WorkflowOrchestrator, WorkflowType

# Initialize orchestrator
orchestrator = WorkflowOrchestrator()

# Run comprehensive analysis
result = await orchestrator.execute_workflow(
    ticker="AAPL",
    workflow_type=WorkflowType.COMPREHENSIVE,
    custom_params={'period': '3mo'}
)

# Access results
print(result.final_report['executive_summary'])
```

### Individual Agent Usage

```python
from agents.core.technical_analyst import TechnicalAnalysisAgent

# Initialize agent
tech_agent = TechnicalAnalysisAgent()

# Run analysis
result = await tech_agent.execute({
    'ticker': 'AAPL',
    'period': '3mo'
})

print(result.data['summary'])
```

### Risk Assessment

```python
from agents.core.risk_assessor import RiskAssessmentAgent

risk_agent = RiskAssessmentAgent()

result = await risk_agent.execute({
    'ticker': 'AAPL',
    'portfolio': ['AAPL', 'GOOGL', 'MSFT'],
    'benchmark': 'SPY'
})

print(f"Risk Score: {result.data['risk_score']}")
```

## 🎯 Agent Capabilities

### Technical Analysis Agent

**Indicators Calculated:**
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- On-Balance Volume (OBV)

**Pattern Recognition:**
- Support and Resistance levels
- Trend detection and strength
- Moving average crossovers
- Candlestick patterns (Doji, Hammer, Shooting Star)
- Head and shoulders (basic implementation)

**Trading Signals:**
- Buy/Sell/Hold recommendations
- Signal strength and confidence levels
- Multi-indicator consensus

### Risk Assessment Agent

**Risk Metrics:**
- Historical and implied volatility
- Value at Risk (VaR) at multiple confidence levels
- Conditional Value at Risk (CVaR)
- Maximum drawdown analysis
- Beta and alpha calculations
- Sharpe, Sortino, and Calmar ratios

**Portfolio Analysis:**
- Correlation analysis
- Diversification metrics
- Risk-adjusted performance
- Portfolio optimization recommendations

### Sentiment Analysis Agent

**Data Sources:**
- Financial news aggregation
- Reddit discussions
- StockTwits sentiment
- General web forums

**Analysis Techniques:**
- VADER sentiment analysis
- TextBlob polarity scoring
- Composite sentiment calculation
- Trending topic extraction
- Momentum and buzz analysis

**Output:**
- Overall sentiment category (BULLISH/BEARISH/NEUTRAL)
- Confidence levels
- Source-specific sentiment breakdown
- Trading signals based on sentiment

## 🔌 API Integration

### Supported Data Sources

**Financial Data:**
- **Alpha Vantage**: Stock prices, fundamentals, technical indicators
- **Yahoo Finance**: Real-time and historical data
- **Finnhub**: Company data, news, and insider trading
- **Polygon**: Market data and options flow
- **FRED**: Economic indicators

**News and Sentiment:**
- **NewsAPI**: Global financial news
- **Reddit API**: Community discussions
- **StockTwits**: Social trading sentiment
- **DuckDuckGo**: Web search for news and sentiment

### Rate Limiting and Caching

- Intelligent API request management
- Redis caching for frequently requested data
- Fallback data sources when primary APIs are unavailable
- Exponential backoff for failed requests

## 📈 Performance Optimization

### Parallel Processing
- Concurrent agent execution
- Asynchronous API calls
- ThreadPoolExecutor for CPU-intensive tasks

### Caching Strategy
- Redis for short-term data caching
- File-based caching for large datasets
- Intelligent cache invalidation

### Memory Management
- Configurable context windows
- Automatic cleanup of old data
- Efficient data structures for large datasets

## 🧪 Testing

### Test Structure

```bash
tests/
├── unit/                   # Unit tests for individual components
├── integration/            # Integration tests for agent workflows
├── performance/            # Performance and load tests
└── fixtures/              # Test data and fixtures
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest tests/ --cov=agents --cov-report=html
```

### Test Coverage

Current test coverage goals:
- Unit tests: >90% coverage
- Integration tests: All major workflows
- Performance tests: Response time benchmarks

## 🔐 Security and Compliance

### API Key Management
- Environment variable storage
- Key rotation support
- Secure key validation

### Data Privacy
- No sensitive data logging
- Configurable data retention
- GDPR compliance considerations

### Error Handling
- Comprehensive exception handling
- Graceful degradation
- Security-conscious error messages

## 📊 Monitoring and Observability

### Metrics Collection
- Prometheus metrics export
- Agent performance tracking
- API usage monitoring
- Error rate tracking

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response tracing
- Performance profiling

### Health Checks
- Agent health endpoints
- Database connectivity checks
- API availability monitoring

## 🚀 Deployment

### Docker Deployment

```bash
# Build the container
docker build -t stock-analysis-extended .

# Run with environment variables
docker run -p 8501:8501 --env-file .env stock-analysis-extended
```

### Production Considerations

- **Scaling**: Horizontal scaling with load balancers
- **Database**: PostgreSQL for production workloads
- **Caching**: Redis cluster for high availability
- **Monitoring**: Prometheus + Grafana setup
- **Logging**: ELK stack or similar

## 🔮 Future Enhancements

### Planned Features

**Q1 2024:**
- [ ] Compliance Agent implementation
- [ ] Portfolio Optimization Agent
- [ ] Market Comparison Agent
- [ ] Options flow analysis
- [ ] Real-time streaming data

**Q2 2024:**
- [ ] ESG scoring integration
- [ ] Insider trading analysis
- [ ] Sector and industry analysis
- [ ] Advanced chart pattern recognition
- [ ] Machine learning predictions

**Q3 2024:**
- [ ] Multi-currency support
- [ ] International markets
- [ ] Advanced portfolio construction
- [ ] Risk scenario modeling
- [ ] Backtesting framework

### Extensibility

The system is designed for easy extension:

- **New Agents**: Follow the `BaseStockAgent` pattern
- **New Data Sources**: Implement tools following the `BaseTool` pattern
- **New Workflows**: Define in `workflow_manager.py`
- **Custom Metrics**: Add to the metrics calculation tools

## 🤝 Contributing

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd stock-analysis-extended

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for all functions
- Document all public methods and classes
- Write comprehensive tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CrewAI**: For the excellent multi-agent framework
- **LangChain/LangGraph**: For workflow orchestration capabilities
- **Streamlit**: For the intuitive web interface
- **Financial Data Providers**: Alpha Vantage, Yahoo Finance, and others
- **Open Source Community**: For the numerous libraries that make this possible

## 📞 Support

For support, please:

1. Check the [documentation](docs/)
2. Search [existing issues](issues)
3. Create a [new issue](issues/new) with detailed information
4. Join our [community discussions](discussions)

---

**Built with ❤️ for the financial analysis community**