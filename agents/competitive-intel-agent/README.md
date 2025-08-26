# Competitive Intel Agent

AI-powered competitive analysis using Agentic RAG with Cohere and LlamaIndex

## Purpose
This agent provides intelligent competitive analysis by combining advanced reasoning (ReAct framework) with retrieval-augmented generation. It helps businesses understand their competitive landscape through automated analysis of competitor data.

## Key Features
- **Agentic RAG**: Combines reasoning and action for intelligent query processing
- **ReAct Framework**: Breaks down complex queries into actionable sub-goals
- **Semantic Search**: Uses Cohere embeddings for accurate information retrieval
- **Strategic Insights**: Generates actionable competitive intelligence
- **Query History**: Tracks and displays recent analysis queries
- **Transparent Reasoning**: Shows decision-making process for each query

## Quick Start
1. Ensure Cohere API key is in `.env` file:
   ```
   COHERE_API_KEY=your_cohere_api_key
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install cohere llama-index llama-index-llms-cohere llama-index-embeddings-cohere pandas
   ```

3. Run the agent:
   ```bash
   streamlit run app.py --server.port 8504
   ```

## Data Structure
The agent expects a CSV file with competitor information including:
- Competitor Name
- Industry
- Product Description
- Marketing Strategy
- Financial Summary
- Market Share
- Strengths & Weaknesses
- Recent Updates

## Example Queries
- "Compare the marketing strategies of TechCorp and DataSoft"
- "What are CloudNet's main competitive advantages?"
- "Give me a financial overview of all competitors"
- "Analyze the weaknesses of InnovateTech"
- "What recent updates have competitors made?"

## How It Works

### ReAct Framework
1. **Reason**: Analyzes query intent (comparison, strengths, weaknesses, etc.)
2. **Act**: Executes sub-goals based on intent
3. **Observe**: Retrieves relevant information
4. **Respond**: Generates comprehensive insights

### Query Processing Pipeline
1. Intent determination
2. Sub-goal creation
3. Information retrieval via LlamaIndex
4. Analysis using Cohere's Command-R model
5. Response generation with actionable insights

## Architecture
- **Embeddings**: Cohere embed-english-v3.0
- **LLM**: Cohere command-r-plus
- **Indexing**: LlamaIndex VectorStoreIndex
- **Retrieval**: Semantic search with top-k results
- **UI**: Streamlit web interface

## Configuration
- Port: 8504
- Models: Configurable via Settings in app.py
- Data source: CSV file in data/ directory

## Notes
- Free Cohere API tier available for testing
- Sample competitor data included
- Extensible to real-time data sources
- Supports batch analysis queries