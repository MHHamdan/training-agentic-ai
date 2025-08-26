import os
import io
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from phi.agent import Agent
from phi.tools import tool
from phi.workflow import Workflow
from phi.model.google import Gemini
from pinecone import Pinecone, ServerlessSpec
# Using OpenAI embeddings as fallback for Python 3.13 compatibility
import openai
import numpy as np
from dotenv import load_dotenv

# ---- 1. Load Environment Variables from .env ----
from pathlib import Path
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- 2. Initialize LLM ----
model = Gemini()

# ---- 3. Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "data-insights"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- 4. Initialize OpenAI embeddings for Python 3.13 compatibility ----
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def get_embedding(text):
    """Get embedding using OpenAI API as fallback for Python 3.13"""
    try:
        if openai_client:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002", 
                input=text
            )
            return response.data[0].embedding
        else:
            raise Exception("OpenAI API key not available")
    except Exception as e:
        # Fallback to simple hash-based embedding if OpenAI fails
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert hash to simple vector
        hash_int = int(hash_obj.hexdigest(), 16)
        # Create a 1536-dimension vector (OpenAI embedding size)
        embedding = [(hash_int >> (i % 128)) & 1 for i in range(1536)]
        return [float(x) for x in embedding]

# ---- 5. Tool: Describe CSV/Excel data ----
@tool
def describe_data(filepath: str) -> str:
    df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
    return df.describe().to_string()

# ---- 6. Tool: Chunk, Embed, and Store CSV/Excel metadata ----
@tool
def embed_and_store(filepath: str) -> str:
    try:
        df = pd.read_csv(filepath) if filepath.endswith(".csv") else pd.read_excel(filepath)
        text_summary = df.describe().to_string()
        
        # Add basic info about the dataset
        dataset_info = f"""
        Dataset: {filepath}
        Rows: {len(df)}
        Columns: {len(df.columns)}
        Column Names: {', '.join(df.columns)}
        Summary: {text_summary}
        """
        
        # Chunk the dataset info into smaller pieces for better embeddings
        chunks = []
        chunk_size = 300
        for i in range(0, len(dataset_info), chunk_size):
            chunk = dataset_info[i:i + chunk_size]
            chunks.append(chunk)
        
        # Create embeddings for each chunk
        embeddings = [get_embedding(chunk) for chunk in chunks]
        
        # Store each chunk in Pinecone
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{filepath}_chunk_{i}"
            vectors.append((vector_id, embedding, {"filepath": filepath, "chunk_text": chunk}))
        
        index.upsert(vectors=vectors)
        
        return f"Successfully chunked and stored {len(chunks)} pieces of metadata for {filepath}"
        
    except Exception as e:
        return f"Error processing {filepath}: {e}"

# ---- 7. Tool: Search similar datasets by metadata ----
@tool
def search_similar(query: str) -> str:
    try:
        # Create embedding for the search query
        query_embedding = get_embedding(query)
        
        # Search Pinecone for similar vectors
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if not results.matches:
            return "No similar datasets found. Please upload and store some datasets first."
        
        # Format results
        formatted_results = []
        for match in results.matches:
            filepath = match.metadata.get('filepath', 'Unknown')
            score = match.score
            chunk_text = match.metadata.get('chunk_text', '')[:100] + '...'
            formatted_results.append(
                f"Dataset: {filepath}\n"
                f"Similarity: {score:.3f}\n"
                f"Content: {chunk_text}\n"
            )
        
        return "\n--- Matching Datasets ---\n" + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error in similarity search: {e}"

# ---- 8. Create the Agent ----
agent = Agent(
    tools=[describe_data, embed_and_store, search_similar],
    model=model,
    name="Data Analyst Agent",
    description="Analyzes CSV, Excel, and PDF files, chunks and stores metadata, and retrieves similar datasets."
)

workflow = Workflow(
    agents=[agent],
    name="data_insight_workflow"
)

# ---- 9. Streamlit UI ----
st.set_page_config(
    page_title="Insights Explorer Agent", 
    page_icon="📊", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📊 Insights Explorer Agent</h1>
    <p>AI-Powered Data Analysis with Semantic Memory</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for status and information
with st.sidebar:
    st.header("⚙️ Agent Status")
    
    # API Status
    if GOOGLE_API_KEY:
        st.success("✅ Google Gemini Connected")
        masked_key = GOOGLE_API_KEY[:8] + "*" * (len(GOOGLE_API_KEY) - 12) + GOOGLE_API_KEY[-4:]
        st.info(f"Key: {masked_key}")
    else:
        st.error("❌ Google API Key Missing")
    
    if PINECONE_API_KEY and index:
        st.success("✅ Pinecone Vector DB Connected")
        try:
            # Get index stats
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            st.info(f"📊 Vectors in Index: {total_vectors}")
        except:
            st.info("📊 Vector count unavailable")
    else:
        st.error("❌ Pinecone API Key Missing")
    
    st.divider()
    
    # Instructions
    st.header("📋 How to Use")
    st.markdown("""
    1. **Upload** a CSV, Excel, or PDF file
    2. **Analyze** - Get automatic statistical summary
    3. **Store** - Embed in semantic memory
    4. **Search** - Find similar datasets
    5. **Export** - Download summaries
    """)
    
    st.divider()
    
    # Sample queries
    st.header("💡 Sample Queries")
    sample_queries = [
        "employee data with demographics",
        "sales reports quarterly",
        "customer behavior analysis",
        "financial time series data"
    ]
    
    for query in sample_queries:
        if st.button(f"🔍 {query}", key=f"sample_{query}"):
            st.session_state.sample_query = query

uploaded_file = st.file_uploader(
    label="Upload a data file (CSV, Excel, or PDF)",
    type=["csv", "xlsx", "xls", "pdf"],
    help="Supported formats: .csv, .xls, .xlsx, .pdf"
)

st.caption("Supports CSV, Excel (.xls/.xlsx), and PDF files. Max size: 200MB.")

if uploaded_file is not None:
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
            st.success("CSV file loaded successfully!")
            file_summary_df = df.describe()

        elif file_type in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
            st.success("Excel file loaded successfully!")
            file_summary_df = df.describe()

        elif file_type == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            file_summary = text if text else "No extractable text found in PDF."
            st.success("PDF file loaded successfully!")

        else:
            st.error("Unsupported file format.")
            file_summary_df = None

        # For CSV and Excel, save file temporarily and run the agent
        if file_type in ["csv", "xlsx", "xls"]:
            temp_path = f"uploaded_data.{file_type}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            response = agent.run(f"Describe the dataset in {temp_path}")
            st.write("Dataset Summary from Agent:")
            st.text(response)

        elif file_type == "pdf":
            st.write("Extracted PDF Text Summary:")
            st.text(file_summary)

        # Show basic summary and download button for CSV/Excel
        if file_type in ["csv", "xlsx", "xls"] and file_summary_df is not None:
            st.write("Basic Summary:")
            st.dataframe(file_summary_df)

            # Convert DataFrame summary to CSV bytes
            csv_bytes = file_summary_df.to_csv().encode('utf-8')

            st.download_button(
                label="Download Summary as CSV",
                data=csv_bytes,
                file_name="summary.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
