# Multi-Agent AI Platform

A simple workspace where different AI assistants work together to help with various tasks. No complicated setup - just run and use.

## What You Get

Three helpful AI assistants that each specialize in different areas:

- **Dashboard**: Central place to access all assistants at http://localhost:8500
- **Easy Management**: Start and stop all assistants with one command
- **Simple Setup**: One configuration file and shared environment
- **Works Anywhere**: Docker support for easy deployment

## Repository Structure

```
multi-agent-platform/
├── agents/
│   ├── customer-support-agent/     # Customer service helper
│   ├── legal-document-review/      # Document analysis tool
│   └── Finance-Advaisor-Agent/     # Personal finance assistant
├── app.py                          # Main dashboard
├── docker-compose.yml              # Docker setup
├── requirements.txt                # Python dependencies
└── .env.example                    # Settings template
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/MHHamdan/training-agentic-ai.git
cd training-agentic-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start All Services

#### Option A: Docker Compose (Recommended)
```bash
# Build and start all agents with Docker
docker compose up -d

# Or use pre-built Docker Hub images
docker compose -f docker-compose.hub.yml up -d
```

#### Option B: Local Python Environment
```bash
# Start orchestrator
streamlit run app.py --server.port 8500 &

# Start legal document review
streamlit run agents/legal-document-review/app.py --server.port 8501 &

# Start customer support agent  
streamlit run agents/customer-support-agent/src/ui/app.py --server.port 8502 &
```

### 4. Access the Platform
- **Main Orchestrator**: http://localhost:8500
- **Customer Support Agent**: http://localhost:8502
- **Legal Document Review**: http://localhost:8501

## Your AI Assistants

### Customer Support Assistant (Port 8502)
- **What it does**: Answers customer questions and knows when to get human help
- **Good for**: Customer service, support requests, common questions
- **Try it**: Ask about returns, account issues, or product help

### Document Review Assistant (Port 8501)  
- **What it does**: Reads legal documents and answers questions about them
- **Good for**: Contract reviews, finding specific terms, document summaries
- **Try it**: Upload a PDF and ask "What are the key terms?"

### Finance Advisor Assistant (Port 8503)
- **What it does**: Shows stock prices, tracks spending, gives money advice
- **Good for**: Personal budgeting, investment research, financial planning  
- **Try it**: Ask "What's Apple's stock price?" or "Help me budget $3000/month"

## Orchestrator Features

The main orchestrator at `http://localhost:8500` provides:

- **Agent Status Monitoring**: Real-time status of all agents
- **Unified Access**: Direct links to all agent interfaces
- **System Metrics**: Health monitoring and analytics
- **Docker Management**: Start/stop commands and logs
- **Development Info**: Project structure and setup guides

## Development

### Adding New Agents
1. Create a new directory in `agents/`
2. Add your agent's Dockerfile and application code
3. Update `docker-compose.yml` with the new service
4. Update the orchestrator's agent registry in `app.py`
5. Add any new dependencies to `requirements.txt`

### Local Development
```bash
# Run orchestrator locally
streamlit run app.py --server.port 8500

# Run individual agents locally
cd agents/customer-support-agent
streamlit run src/ui/app.py --server.port 8502

cd agents/legal-document-review
streamlit run app.py --server.port 8501
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific agent tests
pytest tests/customer-support/
pytest tests/legal-document/
```

## Docker Hub Images

The platform is available as pre-built Docker images:
- `440930/training-agentic-ai:orchestrator` - Main dashboard
- `440930/training-agentic-ai:legal-document-review` - Legal document processor  
- `440930/training-agentic-ai:customer-support-agent` - Support chatbot

### Docker Commands
```bash
# Pull latest images from Docker Hub
docker pull 440930/training-agentic-ai:orchestrator
docker pull 440930/training-agentic-ai:legal-document-review
docker pull 440930/training-agentic-ai:customer-support-agent

# Build all services locally
docker compose build

# Start all services
docker compose up -d

# Start with Docker Hub images
docker compose -f docker-compose.hub.yml up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Restart specific service
docker compose restart customer-support-agent
```

## Monitoring

- **Health Checks**: Each agent provides health endpoints
- **Logs**: Centralized logging through Docker Compose
- **Metrics**: System metrics available in the orchestrator
- **Status**: Real-time status monitoring of all agents

## Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for Google Gemini AI services
- `ENVIRONMENT`: Set to 'development' or 'production'
- `DATABASE_URL`: Database connection string (if needed)
- `REDIS_URL`: Redis connection string (if needed)

### Port Configuration
- **Orchestrator**: 8500
- **Legal Document Review**: 8501
- **Customer Support Agent**: 8502

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your agent or improvements
4. Update the orchestrator if needed
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Latest Updates

### v1.0 - Production Ready (Latest)
- Fixed LangChain compatibility issues with updated imports and method calls
- Docker Hub integration with pre-built images for instant deployment
- Complete production deployment pipeline with automated builds
- All agents tested and verified working in production environment
- Comprehensive documentation and setup guides added
- Resolved dependency conflicts and deprecation warnings

### Key Improvements
- Updated `langchain.vectorstores` to `langchain_community.vectorstores`
- Fixed `qa_chain.__call__()` to `qa_chain.invoke()` compatibility
- Streamlined Docker deployment with hub images
- Added deployment automation scripts

## Support

For issues and questions:
1. Check the **DEPLOYMENT_SUMMARY.md** for latest setup instructions
2. Review the agent-specific README files in `agents/`
3. Check Docker logs: `docker compose logs`
4. Verify orchestrator status at `http://localhost:8500`
5. Open an issue on GitHub

## Quick Deploy
```bash
# Fastest way to get started with pre-built images
git clone https://github.com/MHHamdan/training-agentic-ai.git
cd training-agentic-ai
cp .env.example .env  # Add your GOOGLE_API_KEY
docker compose -f docker-compose.hub.yml up -d
# Access at http://localhost:8500
```

---

**Built for the AI development community**