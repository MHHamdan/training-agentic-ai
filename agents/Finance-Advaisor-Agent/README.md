# Finance Advisor Agent

Purpose
This agent provides personalized financial advice, real-time stock information, expense tracking, and budget management using LangGraph and Groq LLM with Alpha Vantage API integration.

Key features
- Real-time stock prices via Alpha Vantage API
- Personalized financial advice based on user profile
- Expense tracking and budget summaries
- Short and long-term memory for context
- Human-in-the-loop for high-risk queries
- Clear, empathetic responses for all financial literacy levels

Quick start
1) Use the shared environment at the repository root
   source ../../venv/bin/activate
   cp ../../.env.example ../../.env  # if not created yet
2) Add your API keys to .env:
   GROQ_API_KEY=your-groq-api-key
   ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
3) Run the app
   streamlit run app.py --server.port 8503

Environment
- GROQ_API_KEY is required for LLM
- ALPHA_VANTAGE_API_KEY is required for stock data

Docker
```bash
# Build and run with Docker
docker compose up -d

# Or from root directory
docker compose up finance-advisor -d
```

Example queries
- "What's the price of Apple stock?"
- "Add $50 for groceries"
- "Show my budget summary"
- "I'm 37 and earn $50,000 - suggest a savings plan"
- "Should I invest in TSLA given my risk tolerance?"

High-risk queries like "liquidate retirement" trigger human review.

Architecture
- Intent detection with LangGraph state management
- Profile collection for personalized advice
- Real-time stock data integration
- Memory management (short and long-term)
- HITL mechanism for high-risk queries

Notes
- Alpha Vantage free tier: 5 API calls/minute
- Logs available in terminal for debugging
- Run from this directory after activating shared venv
- Access at http://localhost:8503