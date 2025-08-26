# Insights Explorer Agent

AI-powered data analysis with semantic memory using Phidata, Google Gemini, and Pinecone

## Purpose
This agent transforms raw tabular datasets into analyzable, semantically searchable knowledge. It automatically generates statistical summaries, embeds metadata for semantic search, and provides conversational AI interactions for data exploration.

## Key Features
- **Automated Data Analysis**: Statistical summaries using pandas.DataFrame.describe()
- **Semantic Memory**: Embeds dataset metadata in Pinecone for similarity search
- **Multi-format Support**: CSV, Excel (.xls/.xlsx), and PDF files
- **Conversational Interface**: Natural language queries about your data
- **Export Capabilities**: Download summaries as CSV files
- **Intelligent Chunking**: Breaks down metadata for fine-grained search

## Quick Start

1. Ensure API keys are configured in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install phidata pinecone sentence-transformers google-generativeai
   ```

3. Run the agent:
   ```bash
   streamlit run app.py --server.port 8505
   ```

## How It Works

### Data Processing Pipeline
1. **Upload**: Users upload CSV, Excel, or PDF files
2. **Analysis**: Automatic statistical analysis using pandas
3. **Chunking**: Metadata split into ~300 character chunks
4. **Embedding**: SentenceTransformer converts chunks to vectors
5. **Storage**: Vectors stored in Pinecone with unique IDs
6. **Search**: Semantic similarity search across all datasets

### Agent Architecture
Built using **Phidata's agent framework** with three core tools:

#### 1. `describe_data(filepath: str)`
- Generates statistical summaries using pandas
- Returns DataFrame.describe() output as string

#### 2. `embed_and_store(filepath: str)`
- Chunks dataset summaries into manageable pieces
- Creates embeddings using SentenceTransformer
- Stores vectors in Pinecone with file-based IDs

#### 3. `search_similar(query: str)`
- Converts natural language queries to embeddings
- Searches Pinecone for semantically similar datasets
- Returns matching dataset IDs and similarity scores

## Technology Stack
- **Phidata**: Agent orchestration and workflow management
- **Google Gemini**: LLM for natural language understanding
- **SentenceTransformers**: Text embeddings (all-MiniLM-L6-v2)
- **Pinecone**: Vector database for semantic search
- **Streamlit**: Web interface
- **Pandas**: Data analysis and processing

## Use Cases

### Data Discovery
- "Find datasets similar to this sales report"
- "Show me all customer-related data"
- "Find time-series datasets"

### Quick Analysis
- Upload CSV → Get instant statistical summary
- Compare datasets semantically
- Identify data patterns and structure

### Knowledge Management
- Build searchable data catalog
- Find related datasets across projects
- Maintain institutional memory of data assets

## Configuration
- **Port**: 8505
- **Vector Dimension**: 384 (SentenceTransformer model)
- **Pinecone Index**: "data-insights"
- **Chunk Size**: 300 characters
- **File Size Limit**: 200MB

## Sample Data
Includes sample datasets:
- Employee attrition data (data.csv)
- Sample analysis files (sample_data.csv)
- Test datasets for development

## Error Handling
- Robust Pinecone index creation/recovery
- File format validation
- API key verification
- Graceful error messages in UI

## Semantic Search Examples
- "employee data with demographics"
- "financial reports quarterly"
- "customer behavior analysis"
- "time series sales data"

## Integration Notes
- Uses shared `.env` configuration
- Compatible with multi-agent orchestrator
- Follows platform UI/UX patterns
- Extensible tool architecture

## Future Enhancements
- Real-time data streaming
- Advanced visualization
- Multi-modal data support
- Automated insight generation
- Data quality assessment