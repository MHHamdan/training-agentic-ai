# 🔬 ARIA - Autogen Research Intelligence Agent

**Advanced AI-powered research assistant with human-in-the-loop control, built on Microsoft Autogen framework**

## 🌟 Overview

ARIA (Autogen Research Intelligence Agent) is a sophisticated research assistant that leverages Microsoft Autogen's multi-agent framework to provide comprehensive topic analysis and investigation. With human-in-the-loop control, ARIA offers an interactive research experience that adapts to different audiences and research depths.

## 🚀 Key Features

### 🤖 Microsoft Autogen Integration
- **Enhanced Research Assistant**: Built on Autogen's AssistantAgent with specialized research capabilities
- **Streamlit User Proxy**: Custom UserProxyAgent integrated with Streamlit for seamless UI control
- **Conversation Manager**: Advanced conversation orchestration with state management

### 🔍 Research Capabilities
- **Multi-perspective Research**: Systematic investigation with balanced viewpoints
- **Academic & Web Search**: Integration with arXiv, PubMed, Crossref, and web search APIs
- **Content Analysis**: Advanced text analysis with sentiment, complexity, and structure evaluation
- **Audience Adaptation**: Tailored content for general, academic, business, and technical audiences

### 🎮 Human-in-the-Loop Control
- **Interactive Conversations**: Real-time research conversations with AI agents
- **Session Management**: Persistent conversation history and state management
- **Research Controls**: Start, pause, continue, and manage research sessions

### 📁 Multi-format Export
- **PDF Export**: Professional research reports with ReportLab
- **Word Documents**: Formatted .docx files with python-docx
- **CSV Data**: Structured conversation and metadata export
- **Markdown**: Clean, readable markdown format
- **JSON**: Complete data export for programmatic access

## 🏗️ Architecture

```
ARIA/
├── app.py                          # Main Streamlit application
├── config/
│   ├── autogen_config.py          # Autogen and LLM configuration
│   └── __init__.py
├── autogen_components/
│   ├── research_assistant.py      # Enhanced Autogen AssistantAgent
│   ├── user_proxy.py             # Streamlit-integrated UserProxyAgent
│   ├── conversation_manager.py   # Multi-agent conversation orchestration
│   └── __init__.py
├── tools/
│   ├── research_tools.py         # Web, academic search, content analysis
│   ├── export_tools.py           # Multi-format export functionality
│   └── __init__.py
├── ui/
│   ├── streamlit_interface.py    # UI components and interfaces
│   └── __init__.py
├── test_aria.py                  # Comprehensive testing suite
└── README.md
```

## 🛠️ Technology Stack

- **Microsoft Autogen**: Multi-agent framework foundation
- **Streamlit**: Interactive web interface
- **Google Gemini**: Primary LLM provider (with fallbacks)
- **BeautifulSoup4**: Web scraping and content parsing
- **ReportLab**: PDF generation
- **python-docx**: Word document creation
- **feedparser**: Academic search (arXiv)
- **pandas**: Data manipulation and CSV export

## 🔧 Configuration

### API Keys
Configure the following API keys in your `.env` file:

```bash
# Primary LLM Provider (recommended)
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key

# Fallback LLM Providers
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
HUGGING_FACE_API=your_huggingface_api_key

# Optional: Search APIs
GOOGLE_SEARCH_API_KEY=your_google_search_key
BING_SEARCH_API_KEY=your_bing_search_key
```

### Research Depths
- **Basic**: Foundational overview with key concepts
- **Intermediate**: Detailed analysis with multiple perspectives (default)
- **Comprehensive**: Exhaustive analysis with historical context and trends

### Target Audiences
- **General**: Accessible language with practical examples
- **Academic**: Scholarly language with peer-reviewed sources
- **Business**: Strategic focus with actionable insights
- **Technical**: Detailed specifications and implementation guidance

## 🎯 Usage

### Starting ARIA
```bash
# From the project root
streamlit run agents/autogen-research-intelligence-agent/app.py --server.port 8510
```

### Basic Research Flow
1. **Configure Research**: Enter topic, select depth and audience
2. **Start Research**: Initiate conversation with AI research assistant
3. **Interact**: Ask follow-up questions, request elaboration
4. **Export Results**: Save research in preferred format

### Example Research Topics
- "Impact of artificial intelligence on healthcare diagnostics"
- "Climate change effects on global agriculture and food security"
- "Economic implications of remote work trends"
- "Quantum computing applications in cryptography"

## 📊 Research Tools

### Web Search
- **DuckDuckGo**: Primary search provider (no API key required)
- **Google Custom Search**: Enhanced results with API key
- **Fallback Mode**: Generates research guidance when APIs unavailable

### Academic Search
- **arXiv**: Open access research papers
- **PubMed**: Medical and biological literature
- **Crossref**: Academic publication metadata

### Content Analysis
- **Basic Analysis**: Word count, readability, key terms
- **Comprehensive Analysis**: Sentiment, topics, entities, complexity
- **Summary Generation**: Extractive summarization

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd agents/autogen-research-intelligence-agent
python test_aria.py
```

Tests cover:
- File structure and imports
- Configuration functionality
- Research tools operation
- Export capabilities
- Autogen components
- API configuration

## 🔄 Integration

ARIA is integrated into the main agent dashboard at:
- **Port**: 8510
- **Status Monitoring**: Automatic health checks
- **Dashboard Access**: Available through main app navigation

## 📈 Performance

- **Concurrent Agent Support**: Multi-threaded conversation management
- **Caching**: Research results and configuration caching
- **Error Handling**: Graceful fallbacks for API failures
- **Resource Management**: Efficient memory and processing usage

## 🛡️ Security & Privacy

- **API Key Protection**: Secure environment variable handling
- **Data Isolation**: Session-based conversation management
- **No Data Persistence**: Research conversations are session-local
- **Export Control**: User-controlled data export only

## 🚧 Future Enhancements

- **Advanced Integrations**: More academic databases and search engines
- **Collaborative Research**: Multi-user research sessions
- **AI Model Expansion**: Support for additional LLM providers
- **Research Templates**: Pre-configured research workflows
- **Citation Management**: Automatic bibliography generation

## 📝 License

Part of the Training Agentic AI project. See main project license.

## 🤝 Contributing

ARIA follows the main project's contribution guidelines. Key areas for contribution:
- Additional search provider integrations
- Enhanced export formats
- Research workflow templates
- UI/UX improvements
- Performance optimizations

---

**ARIA** - *Advancing Research through Intelligent Automation* 🔬✨