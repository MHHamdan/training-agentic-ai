# Research Agent V2 🔬

Production-ready AI research agent with full Langfuse observability, LangGraph orchestration, and multi-model support. Enterprise-grade research with academic compliance, fact-checking, and citation management.

## 🚀 Features

### Core Capabilities
- **🔍 Multi-Source Search**: DuckDuckGo, ArXiv, Wikipedia, News APIs
- **🧠 LangGraph Orchestration**: Multi-agent workflow with specialized roles
- **📊 Langfuse Observability**: Complete activity tracking and performance monitoring
- **🤖 Multi-Model Support**: OpenAI, Anthropic, Google, Hugging Face integration
- **✅ Fact Checking**: Advanced claim verification with confidence scoring
- **📚 Academic Citations**: Automatic citation generation in multiple formats
- **🎯 Quality Evaluation**: Comprehensive research quality assessment

### Advanced Features
- **📈 Real-time Monitoring**: Live progress tracking and performance analytics
- **🔍 Model Comparison**: Side-by-side performance evaluation
- **🛡️ Academic Compliance**: Plagiarism detection and integrity validation
- **📊 Analytics Dashboard**: Research trends and quality metrics
- **⚡ Production Ready**: Error handling, caching, and scalability

## 🏗️ Architecture

### Multi-Agent System
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Search Agent   │───▶│ Analyzer Agent  │───▶│Synthesizer Agent│
│                 │    │                 │    │                 │
│ • Web Search    │    │ • Content       │    │ • Report        │
│ • Source Rating │    │   Analysis      │    │   Generation    │
│ • Quality Check │    │ • Entity        │    │ • Citations     │
└─────────────────┘    │   Extraction    │    │ • Bibliography  │
                       │ • Insight       │    └─────────────────┘
                       │   Generation    │              │
                       └─────────────────┘              ▼
                                │                ┌─────────────────┐
                                ▼                │ Evaluator Agent │
                       ┌─────────────────┐       │                 │
                       │ Fact Checker    │       │ • Quality Score │
                       │                 │       │ • Bias Detection│
                       │ • Claim         │◀──────│ • Completeness  │
                       │   Verification  │       │ • Recommendations│
                       │ • Confidence    │       └─────────────────┘
                       │   Scoring       │
                       └─────────────────┘
```

### Technology Stack
- **Framework**: LangGraph for workflow orchestration
- **Observability**: Langfuse for complete tracking
- **UI**: Streamlit with real-time dashboard
- **Models**: Multi-provider support (OpenAI, Anthropic, Google, HF)
- **Search**: DuckDuckGo, ArXiv, Wikipedia APIs
- **Storage**: In-memory with optional Redis caching

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Copy environment configuration
cp .env.example .env

# Edit with your API keys
nano .env
```

### 2. Required API Keys
```env
# Essential (choose one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Observability (recommended)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...

# Enhanced Search (optional)
NEWS_API_KEY=...
HUGGINGFACE_API_KEY=hf_...
```

### 3. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Or run from main platform
./start_all_agents.sh
```

### 4. Access
- **Direct**: http://localhost:8514
- **Main Dashboard**: http://localhost:8500 → Research Agent V2

## 📖 Usage Examples

### Basic Research
```python
from research_agent import ResearchAgent

agent = ResearchAgent()

# Conduct research
result = await agent.conduct_research(
    query="What are the latest developments in quantum computing?",
    depth="comprehensive"
)

print(f"Quality Score: {result['quality_score']:.1%}")
print(f"Key Insights: {len(result['key_insights'])}")
```

### Advanced Configuration
```python
# Custom research parameters
result = await agent.conduct_research(
    query="Climate change impact on agriculture",
    user_preferences={
        "depth": "exhaustive",
        "citation_format": "APA",
        "max_sources": 30,
        "enable_fact_check": True,
        "quality_threshold": 0.8
    }
)
```

### Model Comparison
```python
# Compare models for research task
comparison = await agent.compare_research_models(
    query="AI ethics frameworks",
    models=["gpt-4", "claude-3-opus", "gemini-pro"]
)
```

## 🎛️ Configuration

### Research Settings
```env
# Analysis depth: quick, standard, comprehensive, exhaustive
ANALYSIS_DEPTH=comprehensive

# Maximum search results per source
MAX_SEARCH_RESULTS=20

# Quality score minimum threshold
QUALITY_SCORE_MINIMUM=0.7

# Fact checking confidence threshold
FACT_CHECK_THRESHOLD=0.8

# Citation format: APA, MLA, Chicago, IEEE, Harvard
CITATION_FORMAT=APA
```

### Performance Settings
```env
# Maximum research timeout (seconds)
MAX_RESEARCH_TIMEOUT_SECONDS=900

# Maximum synthesis length (characters)
SYNTHESIS_MAX_LENGTH=5000

# Enable caching for repeated queries
CACHE_ENABLED=true
```

### Model Configuration
```env
# Default model for research tasks
DEFAULT_MODEL=microsoft/phi-3-mini-4k-instruct

# Enable model performance comparison
ENABLE_MODEL_COMPARISON=true

# Maximum concurrent models for comparison
MAX_CONCURRENT_MODELS=3
```

## 📊 Monitoring & Observability

### Langfuse Integration
The agent provides complete observability through Langfuse:

- **Trace-level tracking**: Every research session
- **Agent interactions**: Multi-agent communication
- **Model performance**: Token usage, latency, costs
- **Quality metrics**: Research quality trends
- **Error tracking**: Failed operations and retries

### Dashboard Features
- **Real-time progress**: Live research workflow status
- **Quality trends**: Historical quality score analysis
- **Source analytics**: Source reliability and usage patterns
- **Model performance**: Comparative analysis across models
- **Cost tracking**: Token usage and estimated costs

## 🔍 API Reference

### Main Research Endpoint
```python
async def conduct_research(
    query: str,
    depth: str = "comprehensive",
    user_preferences: Optional[Dict] = None
) -> ResearchState
```

### Quality Evaluation
```python
async def evaluate_research_quality(
    research_output: Dict,
    original_query: str
) -> Dict[str, float]
```

### Fact Verification
```python
async def verify_facts(
    claims: List[str],
    sources: List[Dict]
) -> Dict[str, Any]
```

### Model Comparison
```python
async def compare_models(
    query: str,
    models: List[str],
    task_type: str = "general_research"
) -> Dict[str, Any]
```

## 🧪 Testing

### Run Tests
```bash
# All tests
pytest tests/

# Specific test categories
pytest tests/test_agent.py          # Core agent tests
pytest tests/test_graph.py          # Workflow tests
pytest tests/test_models.py         # Model integration tests
pytest tests/test_observability.py  # Langfuse tests
```

### Performance Testing
```bash
# Research quality benchmarks
python tests/benchmark_quality.py

# Model performance comparison
python tests/benchmark_models.py

# Load testing
python tests/load_test.py
```

## 📈 Quality Metrics

### Research Quality Scoring
- **Accuracy**: Fact verification success rate
- **Completeness**: Coverage of query aspects
- **Relevance**: Information relevance to query
- **Bias Detection**: Balance and objectivity
- **Source Reliability**: Quality of sources used
- **Citation Quality**: Academic standard compliance

### Performance Targets
- **Research Time**: < 300 seconds for comprehensive analysis
- **Quality Score**: > 0.8 for production research
- **Accuracy Rate**: > 90% for fact verification
- **Uptime**: 99.7% availability target

## 🛠️ Development

### Project Structure
```
agents/research-agent/
├── config.py                 # Configuration management
├── app.py                    # Streamlit application
├── graph/                    # LangGraph workflow
│   ├── workflow_manager.py   # Orchestration logic
│   ├── state.py              # State management
│   └── nodes/                # Individual workflow nodes
├── agents/                   # Specialized agents
│   ├── search_agent.py       # Web search specialist
│   ├── analyzer_agent.py     # Content analysis
│   ├── synthesizer_agent.py  # Report synthesis
│   └── evaluator_agent.py    # Quality evaluation
├── models/                   # Model integration
│   └── model_manager.py      # Multi-provider support
├── tools/                    # Research tools
│   └── fact_checker.py       # Fact verification
├── utils/                    # Utilities
│   └── validators.py         # Input validation
└── tests/                    # Test suite
```

### Adding New Models
```python
# In models/model_manager.py
class ModelManager:
    def _initialize_providers(self):
        # Add new provider
        if self.config.apis.new_provider_key:
            self.providers["new_provider"] = NewProviderClient()
```

### Custom Search Sources
```python
# In agents/search_agent.py
class SearchAgent:
    async def _search_new_source(self, query, max_results):
        # Implement new search source
        return search_results
```

## 🔒 Security

### Data Protection
- **API Key Security**: Environment variable storage
- **Input Validation**: Comprehensive sanitization
- **Content Filtering**: Malicious content detection
- **Rate Limiting**: API abuse prevention

### Privacy
- **No Data Storage**: Research data not persisted
- **Anonymization**: PII removal from logs
- **Secure Transmission**: HTTPS/TLS encryption
- **Access Control**: Role-based permissions

## 📋 Troubleshooting

### Common Issues

**Agent not starting**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify API keys
python -c "from config import config; print(config.validate_configuration())"
```

**Low quality scores**
- Increase `MAX_SEARCH_RESULTS`
- Add more API keys for diverse sources
- Adjust `QUALITY_SCORE_MINIMUM`

**Slow research performance**
- Reduce `ANALYSIS_DEPTH` to "standard"
- Decrease `SYNTHESIS_MAX_LENGTH`
- Enable caching with `CACHE_ENABLED=true`

### Logs and Debugging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check specific component logs
tail -f logs/search_agent.log
tail -f logs/fact_checker.log
```

## 🤝 Contributing

### Development Setup
```bash
# Clone and setup
git clone <repository>
cd agents/research-agent
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest

# Code quality
black .
flake8 .
mypy .
```

### Guidelines
- Follow type hints for all functions
- Add Langfuse observability to new components
- Include comprehensive tests
- Update documentation for new features

## 📄 License

This project is part of the Training Agentic AI platform. See main repository for license details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs in the main repository
- **Langfuse Dashboard**: Monitor performance at configured host
- **Logs**: Check application logs for detailed error information

---

**🎯 Ready for production research with enterprise-grade observability and quality assurance!**