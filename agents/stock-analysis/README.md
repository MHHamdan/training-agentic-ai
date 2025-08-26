# 📈 Stock Analysis Agent

**Enterprise-grade AI-powered stock analysis system with multi-agent orchestration, comprehensive risk assessment, and regulatory compliance.**

## 🌟 Features

### 🤖 Multi-Agent Architecture
- **Stock Analyst Agent**: Fundamental analysis and company evaluation
- **Technical Analyst Agent**: Chart patterns, indicators, and trading signals
- **News Researcher Agent**: Sentiment analysis and market intelligence
- **Risk Assessor Agent**: Comprehensive risk management and compliance
- **Report Writer Agent**: Professional-grade investment reports

### 📊 Comprehensive Analysis
- **Fundamental Analysis**: Financial health, valuation metrics, competitive positioning
- **Technical Analysis**: Chart patterns, support/resistance, momentum indicators
- **Sentiment Analysis**: News sentiment, social media indicators, analyst ratings
- **Risk Assessment**: VaR calculations, stress testing, regulatory compliance
- **Options Flow Analysis**: Institutional positioning and volatility insights

### 🛡️ Enterprise Compliance
- **SEC Compliance**: Investment advice regulations and disclosure requirements
- **FINRA Compliance**: Suitability assessment and best execution standards  
- **Risk Management**: Position limits, concentration rules, stress testing
- **Audit Trails**: Complete transaction logging with data integrity verification
- **Regulatory Reporting**: Automated compliance validation and reporting

### 🚀 Advanced Technology Stack
- **AgentOps Integration**: Complete observability and performance tracking
- **Multi-Model Support**: Hugging Face financial models with benchmarking
- **CrewAI Orchestration**: Coordinated multi-agent workflows
- **Real-time Monitoring**: Live performance metrics and system health
- **Streamlit Dashboard**: Professional web interface with interactive charts

## 🏗️ Architecture

```
agents/stock-analysis/
├── config.py                 # Configuration management
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                # Documentation
├── crew/                    # CrewAI agent orchestration
│   ├── stock_analysis_crew.py
│   ├── tasks.py
│   └── agents/
│       ├── stock_analyst.py
│       ├── technical_analyst.py
│       ├── news_researcher.py
│       ├── risk_assessor.py
│       └── report_writer.py
├── models/                  # AI model management
│   ├── model_manager.py
│   └── hf_models.py
├── tools/                   # Financial data tools
│   └── market_data.py
├── utils/                   # Utilities and services
│   ├── observability.py    # AgentOps integration
│   └── compliance.py       # Regulatory compliance
└── data/                   # Data storage (created at runtime)
    ├── cache/
    └── audit_logs/
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API Keys:
  - Alpha Vantage (optional, for enhanced data)
  - AgentOps (optional, for monitoring)
  - Hugging Face (optional, for premium models)

### Installation

1. **Clone and navigate to the agent directory:**
   ```bash
   cd agents/stock-analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard:**
   Open `http://localhost:8501` in your browser

### Environment Configuration

Create a `.env` file with your API keys:

```env
# AgentOps (Optional - for monitoring)
AGENTOPS_API_KEY=your_agentops_key_here

# Alpha Vantage (Optional - for enhanced financial data)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Hugging Face (Optional - for premium models)
HUGGINGFACE_API_TOKEN=your_hf_token_here

# Model Configuration
FINANCIAL_REASONING_MODEL=microsoft/DialoGPT-large
SENTIMENT_ANALYSIS_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
TECHNICAL_ANALYSIS_MODEL=microsoft/DialoGPT-large
TEXT_GENERATION_MODEL=microsoft/DialoGPT-large

# System Configuration
LOG_LEVEL=INFO
ENABLE_AGENTOPS=true
ENABLE_COMPLIANCE_LOGGING=true
```

## 🎯 Usage Examples

### Basic Stock Analysis

```python
from crew.stock_analysis_crew import StockAnalysisCrew
import asyncio

# Initialize the crew
crew = StockAnalysisCrew(config)

# Perform comprehensive analysis
result = asyncio.run(crew.analyze_stock("AAPL", "comprehensive"))

print(f"Recommendation: {result['executive_summary']['recommendation']['action']}")
print(f"Target Price: ${result['executive_summary']['recommendation']['target_price']}")
```

### Quick Analysis

```python
# For faster analysis with key insights
result = asyncio.run(crew.quick_analysis("TSLA"))
```

### Portfolio Risk Assessment

```python
portfolio = {
    "positions": {
        "AAPL": 0.30,
        "GOOGL": 0.25, 
        "MSFT": 0.20,
        "AMZN": 0.15,
        "CASH": 0.10
    },
    "total_value": 1000000
}

risk_analysis = asyncio.run(crew.analyze_portfolio(portfolio))
```

### Web Interface Features

1. **Stock Analysis Tab**: Enter ticker symbols for comprehensive analysis
2. **Real-time Monitoring**: View system performance and usage metrics
3. **Portfolio Analysis**: Analyze portfolio risk and allocations
4. **Settings**: Configure models, APIs, and system parameters

## 📊 Analysis Components

### 1. Fundamental Analysis
- Financial health scoring (1-10 scale)
- Valuation metrics (P/E, P/B, PEG ratios)
- Competitive positioning assessment
- Growth prospects evaluation
- Management effectiveness analysis

### 2. Technical Analysis
- Trend analysis with strength indicators
- Support and resistance level identification
- Chart pattern recognition
- Technical indicator calculations (RSI, MACD, Bollinger Bands)
- Trading signal generation with entry/exit points

### 3. Sentiment Analysis
- News sentiment scoring (-100 to +100)
- Social media sentiment indicators
- Analyst rating consensus
- Earnings sentiment analysis
- Market psychology assessment

### 4. Risk Assessment
- Value at Risk (VaR) calculations
- Stress testing scenarios
- Liquidity risk evaluation
- Concentration risk analysis
- Regulatory compliance validation

## 🛡️ Compliance Features

### SEC Compliance
- Required disclosure validation
- Material risk identification
- Methodology transparency
- Past performance disclaimers
- Conflict of interest statements

### FINRA Compliance
- Customer suitability assessment
- Investment objective alignment
- Risk tolerance validation
- Best execution considerations
- Know Your Customer (KYC) requirements

### Risk Management
- Position size limits (10% single security)
- Sector concentration limits (25%)
- VaR threshold monitoring (15%)
- Stress testing requirements
- Liquidity minimum standards

### Audit Trail
- Complete transaction logging
- Data integrity verification
- 7-year retention compliance
- Regulatory reporting capabilities
- Access control and monitoring

## 🔧 Model Configuration

The system supports multiple AI models for different analysis types:

```python
models = {
    "financial_reasoning": "microsoft/DialoGPT-large",
    "sentiment_analysis": "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    "technical_analysis": "microsoft/DialoGPT-large",
    "text_generation": "microsoft/DialoGPT-large"
}
```

### Model Performance Tracking
- Response time monitoring
- Accuracy benchmarking
- Cost analysis
- A/B testing capabilities
- Automatic model selection

## 📈 Monitoring and Observability

### AgentOps Integration
- Real-time performance metrics
- Agent activity tracking
- Error monitoring and alerting
- Usage analytics and reporting
- Cost optimization insights

### System Health Monitoring
- Response time tracking
- Success rate monitoring
- Resource utilization metrics
- Error rate analysis
- Performance benchmarking

## 🔒 Security Features

### Data Protection
- Input/output encryption
- Secure API key management
- Access control and authentication
- Audit logging with integrity verification
- GDPR compliance measures

### Risk Controls
- Input validation and sanitization
- Output filtering for sensitive information
- Rate limiting and throttling
- Compliance rule enforcement
- Automated risk alerts

## 🧪 Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run compliance validation tests:

```bash
pytest tests/test_compliance.py -v
```

## 📚 API Reference

### Stock Analysis Crew

```python
class StockAnalysisCrew:
    async def analyze_stock(ticker: str, analysis_type: str = "comprehensive") -> Dict
    async def quick_analysis(ticker: str) -> Dict
    async def analyze_portfolio(portfolio: Dict, benchmark: str = "SPY") -> Dict
    def get_crew_status() -> Dict
```

### Individual Agents

```python
class StockAnalystAgent:
    async def analyze_fundamentals(ticker: str, financial_data: Dict) -> Dict
    async def analyze_company_profile(ticker: str, company_data: Dict) -> Dict

class TechnicalAnalystAgent:
    async def perform_technical_analysis(ticker: str, price_data: Dict) -> Dict
    async def identify_chart_patterns(ticker: str, price_data: Dict) -> Dict

class NewsResearcherAgent:
    async def analyze_market_sentiment(ticker: str, news_data: List) -> Dict
    async def assess_news_impact(ticker: str, news_data: List, price_data: Dict) -> Dict

class RiskAssessorAgent:
    async def perform_risk_assessment(ticker: str, portfolio_data: Dict, market_data: Dict, analysis_data: Dict) -> Dict
    async def validate_regulatory_compliance(investment_decision: Dict, client_profile: Dict) -> Dict
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Ensure compliance validation passes
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code documentation
- **Issues**: Report bugs and feature requests via GitHub issues
- **Community**: Join our Discord for discussions and support

## 🔮 Roadmap

### Phase 1: Core Features ✅
- [x] Multi-agent architecture
- [x] Fundamental and technical analysis
- [x] Risk assessment and compliance
- [x] Streamlit dashboard

### Phase 2: Advanced Features 🚧
- [ ] Real-time data streaming
- [ ] Advanced portfolio optimization
- [ ] Machine learning model training
- [ ] Mobile application

### Phase 3: Enterprise Features 🔮
- [ ] Multi-user support
- [ ] Advanced reporting
- [ ] Integration APIs
- [ ] White-label solutions

## 🙏 Acknowledgments

- **CrewAI**: Multi-agent orchestration framework
- **Hugging Face**: AI model ecosystem
- **AgentOps**: Agent observability and monitoring
- **Streamlit**: Web application framework
- **yfinance**: Financial market data
- **Alpha Vantage**: Financial data API

---

**Built with ❤️ for the financial analysis community**

*This software is for educational and research purposes. Always consult with qualified financial professionals before making investment decisions.*