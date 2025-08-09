# Training Agentic AI - Unified Agent Platform

A comprehensive platform for building, managing, and orchestrating AI agents with a unified landing page that serves as the central control center.

## 🎯 Overview

This repository contains a collection of AI agents organized under the `agents/` folder, all managed through a unified orchestrator interface. The platform provides:

- **Unified Landing Page**: Central orchestrator at `http://localhost:8500`
- **Agent Management**: Start, stop, monitor, and manage all agents
- **Shared Environment**: Single `.env` file and Python virtual environment
- **Docker Integration**: Complete containerization for easy deployment

## 🏗️ Repository Structure

```
training-agentic-ai/
├── agents/
│   ├── customer-support-agent/     # LangGraph-based support system
│   └── legal-document-review/      # RAG-based document analysis
├── tests/                          # Test suites for all agents
├── app.py                          # Main orchestrator (landing page)
├── docker-compose.yml              # Multi-service orchestration
├── requirements.txt                # Shared dependencies
├── .env.example                    # Environment template
└── README.md                       # This file
```

## 🚀 Quick Start

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
```bash
# Build and start all agents with Docker
docker compose up -d

# Or start individual services
docker compose up -d orchestrator
docker compose up -d customer-support-agent
docker compose up -d legal-document-review
```

### 4. Access the Platform
- **Main Orchestrator**: http://localhost:8500
- **Customer Support Agent**: http://localhost:8502
- **Legal Document Review**: http://localhost:8501

## 🤖 Available Agents

### Customer Support Agent (Port 8502)
- **Technology**: LangGraph, Streamlit, Google Gemini
- **Features**: Multi-turn conversations, context awareness, escalation handling
- **Use Case**: AI-powered customer support with human-in-the-loop capabilities

### Legal Document Review (Port 8501)
- **Technology**: LangChain, FAISS, Google Gemini, PyPDF2
- **Features**: PDF processing, semantic search, question answering
- **Use Case**: Legal document analysis and review automation

## 🎛️ Orchestrator Features

The main orchestrator at `http://localhost:8500` provides:

- **Agent Status Monitoring**: Real-time status of all agents
- **Unified Access**: Direct links to all agent interfaces
- **System Metrics**: Health monitoring and analytics
- **Docker Management**: Start/stop commands and logs
- **Development Info**: Project structure and setup guides

## 🛠️ Development

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

## 🐳 Docker Commands

```bash
# Build all services
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down

# Restart specific service
docker compose restart customer-support-agent
```

## 📊 Monitoring

- **Health Checks**: Each agent provides health endpoints
- **Logs**: Centralized logging through Docker Compose
- **Metrics**: System metrics available in the orchestrator
- **Status**: Real-time status monitoring of all agents

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for Google Gemini AI services
- `ENVIRONMENT`: Set to 'development' or 'production'
- `DATABASE_URL`: Database connection string (if needed)
- `REDIS_URL`: Redis connection string (if needed)

### Port Configuration
- **Orchestrator**: 8500
- **Legal Document Review**: 8501
- **Customer Support Agent**: 8502

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your agent or improvements
4. Update the orchestrator if needed
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the agent-specific README files in `agents/`
2. Review the Docker logs: `docker compose logs`
3. Check the orchestrator status at `http://localhost:8500`
4. Open an issue on GitHub

---

**Built with ❤️ for the AI community**

