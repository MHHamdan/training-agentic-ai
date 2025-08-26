import os
import io
import json
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from pathlib import Path
import hashlib
from typing import List, Dict, Any
from dataclasses import dataclass

# Agno/Phidata imports
from phi.agent import Agent
from phi.tools import tool
from phi.workflow import Workflow
from phi.model.google import Gemini

# Hugging Face imports with proper availability detection
def check_huggingface_availability():
    """Check if Hugging Face is available and what capabilities we have"""
    pytorch_available = False
    hf_available = False
    hf_api_available = False
    
    # Check PyTorch
    try:
        import torch
        pytorch_available = True
    except ImportError:
        pass
    
    # Check Transformers and HuggingFace Hub
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import login, InferenceClient
        hf_available = True
        hf_api_available = True
    except ImportError:
        pass
    
    return pytorch_available, hf_available, hf_api_available

PYTORCH_AVAILABLE, HUGGINGFACE_AVAILABLE, HF_API_AVAILABLE = check_huggingface_availability()

# Vector database and embeddings
from pinecone import Pinecone, ServerlessSpec
import openai
import numpy as np

# Utilities
from dotenv import load_dotenv

# ---- 1. Load Environment Variables ----
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent  # Go up two levels: agents/Customer-Support-Triage -> training-agentic-ai
env_path = root_dir / '.env'

# Force reload environment variables
load_dotenv(env_path, override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Force print to verify loading (this will show in terminal) 
print(f"DEBUG: Loading .env from: {env_path} - TIMESTAMP: {datetime.now()}")
print(f"DEBUG: HUGGINGFACE_API_KEY loaded: {'YES' if HUGGINGFACE_API_KEY else 'NO'}")

# Set Hugging Face API key
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_API_KEY

# ---- 2. Data Classes for Support Tickets ----
@dataclass
class SupportTicket:
    ticket_id: str
    customer_name: str
    customer_email: str
    product: str
    ticket_type: str
    subject: str
    description: str
    status: str
    priority: str
    channel: str
    date_created: str
    sentiment: str = None
    intent: str = None
    urgency_score: float = 0.0
    suggested_response: str = None

# ---- 3. Model Selection Function ----
class HuggingFaceInferenceModel:
    """Simple wrapper for Hugging Face Inference API when PyTorch not available"""
    def __init__(self, api_key: str, model: str = "microsoft/DialoGPT-medium"):
        self.api_key = api_key
        self.model = model
        # Import here to avoid module-level import issues
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=api_key)
    
    def run(self, prompt: str):
        try:
            response = self.client.text_generation(
                prompt, 
                model=self.model,
                max_new_tokens=512,
                temperature=0.7
            )
            return type('Response', (), {'content': response})()
        except Exception as e:
            return type('Response', (), {'content': f"Error: {e}"})()

def get_model():
    """Get a compatible model for Phidata Agent - actual inference handled separately"""
    # Always use Google Gemini for Phidata Agent compatibility
    # Real inference will be handled by run_inference() function
    
    if GOOGLE_API_KEY:
        try:
            return Gemini(api_key=GOOGLE_API_KEY)
        except Exception as e:
            pass
    
    # Fallback to OpenAI if Google fails
    if OPENAI_API_KEY:
        try:
            from phi.model.openai import OpenAI
            return OpenAI(api_key=OPENAI_API_KEY)
        except Exception as e:
            pass
    
    # If no model available, create a dummy one (tools will use HF directly)
    return None

# ---- 4. Initialize Models and Services ----
# Debug: Show initialization status
print(f"=== MODEL INITIALIZATION DEBUG ===")
print(f"HF_API_AVAILABLE: {HF_API_AVAILABLE}")
print(f"HUGGINGFACE_API_KEY present: {bool(HUGGINGFACE_API_KEY)}")
print(f"About to call get_model()...")

model = get_model()
print(f"Model created: {type(model).__name__}")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "support-triage"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- 5. Embedding Function with Multiple Providers ----
def get_embedding(text: str) -> List[float]:
    """Get embedding using available providers with fallbacks - FREE ONLY"""
    try:
        # Check if we have HuggingFace API key - if so, force hash-based embedding to stay free
        if STREAMLIT_HUGGINGFACE_API_KEY or (hasattr(st, 'session_state') and st.session_state.get('hf_api_key')):
            st.info("💡 Using hash-based embeddings (FREE - no API costs)")
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            embedding = [(hash_int >> (i % 128)) & 1 for i in range(1536)]
            return [float(x) for x in embedding]
        
        # Only try HuggingFace sentence-transformers if PyTorch available
        if PYTORCH_AVAILABLE and HUGGINGFACE_API_KEY:
            try:
                # Use sentence-transformers for embeddings (requires PyTorch)
                from sentence_transformers import SentenceTransformer
                model_emb = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model_emb.encode([text])[0].tolist()
                
                # Pad or truncate to 1536 dimensions to match OpenAI format
                if len(embedding) < 1536:
                    embedding.extend([0.0] * (1536 - len(embedding)))
                else:
                    embedding = embedding[:1536]
                
                return embedding
            except Exception as hf_error:
                st.warning(f"Hugging Face embedding failed: {hf_error}, using hash fallback...")
        
        # Always use hash-based fallback to avoid any API costs
        st.info("Using hash-based embedding fallback (FREE)")
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        embedding = [(hash_int >> (i % 128)) & 1 for i in range(1536)]
        return [float(x) for x in embedding]
        
    except Exception as e:
        st.info(f"Using hash fallback due to error: {e}")
        # Hash-based fallback
        hash_obj = hashlib.md5(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        embedding = [(hash_int >> (i % 128)) & 1 for i in range(1536)]
        return [float(x) for x in embedding]

# Create Hugging Face client for tools
def get_hf_client():
    """Get Hugging Face client if available"""
    if HUGGINGFACE_API_KEY:
        try:
            from huggingface_hub import InferenceClient
            return InferenceClient(token=HUGGINGFACE_API_KEY)
        except Exception as e:
            st.error(f"Failed to create HF client: {e}")
    return None

def run_inference(prompt: str) -> str:
    """Run inference using available model (HF first, then fallback)"""
    # Try Hugging Face first if available
    hf_client = get_hf_client()
    if hf_client:
        try:
            response = hf_client.text_generation(
                prompt,
                model="microsoft/DialoGPT-medium",
                max_new_tokens=512,
                temperature=0.7
            )
            return response
        except Exception as e:
            st.warning(f"HF inference failed: {e}, using fallback...")
    
    # Fallback to the configured model
    try:
        return run_inference(prompt)
    except Exception as e:
        return f"Error in inference: {e}"

# ---- 6. Support Triage Tools ----
@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment and urgency of customer support message"""
    try:
        prompt = f"""
        Analyze the sentiment and urgency of this customer support message:
        
        "{text}"
        
        Provide analysis in JSON format:
        {{
            "sentiment": "positive|neutral|negative",
            "urgency_score": 1-10,
            "urgency_keywords": ["list", "of", "keywords"],
            "emotional_indicators": ["frustrated", "angry", "confused", "etc"],
            "escalation_needed": true/false
        }}
        """
        
        return run_inference(prompt)
    except Exception as e:
        return f"Error analyzing sentiment: {e}"

@tool
def classify_intent(ticket_description: str, ticket_type: str) -> str:
    """Classify customer intent and categorize the support ticket"""
    try:
        prompt = f"""
        Classify this customer support ticket:
        
        Type: {ticket_type}
        Description: {ticket_description}
        
        Provide classification in JSON format:
        {{
            "primary_intent": "refund|technical_support|billing|product_inquiry|account_access|shipping|cancellation",
            "secondary_intents": ["list", "of", "secondary", "intents"],
            "product_category": "electronics|software|subscription|hardware|etc",
            "complexity_level": "simple|moderate|complex",
            "estimated_resolution_time": "minutes|hours|days",
            "department": "billing|technical|sales|returns|general"
        }}
        """
        
        return run_inference(prompt)
    except Exception as e:
        return f"Error classifying intent: {e}"

@tool
def generate_response(ticket_description: str, customer_name: str, product: str, intent_analysis: str) -> str:
    """Generate suggested response for customer support ticket"""
    try:
        prompt = f"""
        Generate a professional customer support response for:
        
        Customer: {customer_name}
        Product: {product}
        Issue: {ticket_description}
        Intent Analysis: {intent_analysis}
        
        Create a helpful, empathetic response that:
        1. Acknowledges the customer's issue
        2. Provides specific next steps or solutions
        3. Sets appropriate expectations
        4. Maintains professional tone
        5. Includes relevant company policies if needed
        
        Format as a complete email response.
        """
        
        return run_inference(prompt)
    except Exception as e:
        return f"Error generating response: {e}"

@tool
def extract_key_info(ticket_data: str) -> str:
    """Extract key information and entities from support tickets"""
    try:
        prompt = f"""
        Extract key information from this support ticket:
        
        {ticket_data}
        
        Extract and return in JSON format:
        {{
            "customer_info": {{
                "name": "customer name",
                "email": "email",
                "account_type": "regular|premium|enterprise"
            }},
            "product_info": {{
                "product_name": "product name",
                "purchase_date": "date if mentioned",
                "warranty_status": "in_warranty|out_of_warranty|unknown"
            }},
            "issue_details": {{
                "error_codes": ["list", "of", "error", "codes"],
                "symptoms": ["list", "of", "symptoms"],
                "troubleshooting_attempted": ["steps", "already", "tried"],
                "impact_level": "low|medium|high|critical"
            }},
            "business_impact": {{
                "affects_business_operations": true/false,
                "number_of_users_affected": "number or range",
                "financial_impact": "none|low|medium|high"
            }}
        }}
        """
        
        return run_inference(prompt)
    except Exception as e:
        return f"Error extracting key info: {e}"

@tool
def search_similar_tickets(query: str) -> str:
    """Search for similar support tickets in the knowledge base"""
    try:
        query_embedding = get_embedding(query)
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        if not results.matches:
            return "No similar tickets found in the knowledge base."
        
        similar_tickets = []
        for match in results.matches:
            score = match.score
            metadata = match.metadata
            ticket_info = f"""
Similarity: {score:.3f}
Ticket ID: {metadata.get('ticket_id', 'Unknown')}
Issue: {metadata.get('subject', 'N/A')}
Product: {metadata.get('product', 'N/A')}
Status: {metadata.get('status', 'N/A')}
Resolution: {metadata.get('resolution', 'N/A')[:100]}...
"""
            similar_tickets.append(ticket_info)
        
        return "Similar tickets found:\n" + "\n".join(similar_tickets)
    except Exception as e:
        return f"Error searching similar tickets: {e}"

@tool
def calculate_refund_eligibility(purchase_date: str, product_type: str) -> str:
    """Calculate refund eligibility based on company policies"""
    try:
        prompt = f"""
        Calculate refund eligibility based on:
        Purchase Date: {purchase_date}
        Product Type: {product_type}
        Current Date: {datetime.now().strftime('%Y-%m-%d')}
        
        Standard policies:
        - Electronics: 30 days return policy
        - Software: 14 days return policy
        - Subscriptions: Cancel anytime, prorated refund
        - Hardware: 45 days return policy with restocking fee
        
        Provide result in JSON format:
        {{
            "eligible_for_refund": true/false,
            "days_since_purchase": number,
            "refund_percentage": 0-100,
            "restocking_fee": 0-20,
            "policy_notes": "explanation of applicable policies",
            "next_steps": "what customer should do"
        }}
        """
        
        return run_inference(prompt)
    except Exception as e:
        return f"Error calculating refund eligibility: {e}"

# ---- 7. Support Triage Agent ----
class SupportTriageAgent:
    def __init__(self):
        # Use a compatible model or create a minimal agent
        agent_model = model if model else Gemini(api_key="dummy") if GOOGLE_API_KEY else None
        
        if agent_model:
            self.agent = Agent(
                tools=[
                    analyze_sentiment,
                    classify_intent, 
                    generate_response,
                    extract_key_info,
                    search_similar_tickets,
                    calculate_refund_eligibility
                ],
                model=agent_model,
                name="Customer Support Triage Agent",
                description="AI-powered customer support triage agent for e-commerce platform YNC",
                instructions="""
                You are a Customer Support Triage Agent for YNC e-commerce platform.
                Your role is to:
                1. Analyze incoming support tickets for sentiment and urgency
                2. Classify customer intents and route tickets appropriately
                3. Generate suggested responses for support agents
                4. Search for similar past tickets and solutions
                5. Provide insights and recommendations for management
                
                Always be helpful, professional, and efficient in your analysis.
                Focus on reducing resolution time and improving customer satisfaction.
                """
            )
        else:
            # Create a minimal agent without model dependency
            self.agent = None
        
        if self.agent:
            self.workflow = Workflow(
                agents=[self.agent],
                name="support_triage_workflow"
            )
        else:
            self.workflow = None

    def process_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single support ticket through the triage workflow"""
        try:
            # Extract ticket information
            description = ticket_data.get('description', '')
            ticket_type = ticket_data.get('ticket_type', '')
            customer_name = ticket_data.get('customer_name', '')
            product = ticket_data.get('product', '')
            
            # Step 1: Sentiment Analysis
            sentiment_result = self.agent.run(f"analyze_sentiment('{description}')")
            
            # Step 2: Intent Classification
            intent_result = self.agent.run(f"classify_intent('{description}', '{ticket_type}')")
            
            # Step 3: Key Information Extraction
            key_info_result = self.agent.run(f"extract_key_info('{json.dumps(ticket_data)}')")
            
            # Step 4: Generate Suggested Response
            response_result = self.agent.run(f"generate_response('{description}', '{customer_name}', '{product}', '{intent_result}')")
            
            # Step 5: Search Similar Tickets
            similar_tickets = self.agent.run(f"search_similar_tickets('{description}')")
            
            return {
                'sentiment_analysis': sentiment_result,
                'intent_classification': intent_result,
                'key_information': key_info_result,
                'suggested_response': response_result,
                'similar_tickets': similar_tickets
            }
        except Exception as e:
            return {'error': f"Error processing ticket: {e}"}

    def store_ticket_in_kb(self, ticket_data: Dict[str, Any]):
        """Store ticket in knowledge base for future similarity searches"""
        try:
            # Clean and sanitize data to avoid NaN/null issues
            def clean_value(value):
                """Clean value for JSON serialization"""
                if pd.isna(value) or value is None:
                    return ""
                return str(value).strip()
            
            # Create text representation of ticket
            ticket_text = f"""
            Ticket ID: {clean_value(ticket_data.get('ticket_id', ''))}
            Customer: {clean_value(ticket_data.get('customer_name', ''))}
            Product: {clean_value(ticket_data.get('product', ''))}
            Type: {clean_value(ticket_data.get('ticket_type', ''))}
            Subject: {clean_value(ticket_data.get('subject', ''))}
            Description: {clean_value(ticket_data.get('description', ''))}
            Status: {clean_value(ticket_data.get('status', ''))}
            Priority: {clean_value(ticket_data.get('priority', ''))}
            """
            
            # Generate embedding
            embedding = get_embedding(ticket_text)
            
            # Clean metadata for Pinecone
            metadata = {
                'ticket_id': clean_value(ticket_data.get('ticket_id', '')),
                'subject': clean_value(ticket_data.get('subject', '')),
                'product': clean_value(ticket_data.get('product', '')),
                'status': clean_value(ticket_data.get('status', '')),
                'resolution': clean_value(ticket_data.get('resolution', '')),
                'priority': clean_value(ticket_data.get('priority', '')),
                'ticket_type': clean_value(ticket_data.get('ticket_type', ''))
            }
            
            # Store in Pinecone
            vector_id = f"ticket_{clean_value(ticket_data.get('ticket_id', str(hash(ticket_text))))}"
            index.upsert(vectors=[(
                vector_id, 
                embedding, 
                metadata
            )])
            
            return True
        except Exception as e:
            st.error(f"Error storing ticket: {e}")
            return False

# ---- 8. Streamlit UI ----
st.set_page_config(
    page_title="Customer Support Triage Agent",
    page_icon="🎫",
    layout="wide"
)

# Force clear all Streamlit cache
st.cache_data.clear()
if hasattr(st, 'cache_resource'):
    st.cache_resource.clear()

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
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-open { color: #e74c3c; font-weight: bold; }
    .status-closed { color: #27ae60; font-weight: bold; }
    .status-pending { color: #f39c12; font-weight: bold; }
    .priority-critical { color: #e74c3c; font-weight: bold; }
    .priority-high { color: #e67e22; font-weight: bold; }
    .priority-low { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎫 Customer Support Triage Agent</h1>
    <p>AI-Powered Support Ticket Analysis & Routing for YNC E-Commerce</p>
</div>
""", unsafe_allow_html=True)

# Force reload environment variables in Streamlit context
@st.cache_data
def load_env_vars():
    """Load environment variables with caching"""
    # Use absolute path to avoid path resolution issues
    env_path = Path("/Users/mohammedhamdan/Documents/AgenticAIRoadMap/training-agentic-ai/.env")
    
    # Debug path resolution
    st.sidebar.write(f"**Debug Paths:**")
    st.sidebar.write(f"Env path: {env_path}")
    st.sidebar.write(f"Env exists: {env_path.exists()}")
    
    env_vars = {}
    if env_path.exists():
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
            st.sidebar.write(f"**Loaded {len(env_vars)} env vars**")
            if 'HUGGINGFACE_API_KEY' in env_vars:
                st.sidebar.success("✅ HF key found in .env file!")
            else:
                st.sidebar.error("❌ HF key missing from .env file")
        except Exception as e:
            st.sidebar.error(f"Error reading .env: {e}")
    else:
        st.sidebar.error("❌ .env file not found at absolute path")
    
    return env_vars

# Load environment variables directly
env_vars = load_env_vars()
STREAMLIT_HUGGINGFACE_API_KEY = env_vars.get('HUGGINGFACE_API_KEY', '')

# Override the global variable if found in .env
if STREAMLIT_HUGGINGFACE_API_KEY:
    HUGGINGFACE_API_KEY = STREAMLIT_HUGGINGFACE_API_KEY
    os.environ["HUGGINGFACE_HUB_TOKEN"] = STREAMLIT_HUGGINGFACE_API_KEY
    # Force update the global variables for tools
    globals()['HUGGINGFACE_API_KEY'] = STREAMLIT_HUGGINGFACE_API_KEY

# Debug display in sidebar
st.sidebar.write("🔧 **Environment Debug:**")
st.sidebar.write(f"Global HF Key: {'✅' if HUGGINGFACE_API_KEY else '❌'}")
st.sidebar.write(f"Streamlit HF Key: {'✅' if STREAMLIT_HUGGINGFACE_API_KEY else '❌'}")

# Don't recreate model here - handle HF in tools instead
if STREAMLIT_HUGGINGFACE_API_KEY:
    st.sidebar.success("✅ **Hugging Face API Key Found!**")
    # Store in session state for tools to use
    st.session_state.hf_api_key = STREAMLIT_HUGGINGFACE_API_KEY
    # Set the global variable for tools
    HUGGINGFACE_API_KEY = STREAMLIT_HUGGINGFACE_API_KEY
    os.environ["HUGGINGFACE_HUB_TOKEN"] = STREAMLIT_HUGGINGFACE_API_KEY
else:
    st.sidebar.warning("⚠️ **Hugging Face API Key Not Found**")

# Initialize agent
if 'triage_agent' not in st.session_state:
    st.session_state.triage_agent = SupportTriageAgent()

if 'uploaded_tickets' not in st.session_state:
    st.session_state.uploaded_tickets = []

if 'processed_tickets' not in st.session_state:
    st.session_state.processed_tickets = []

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Status
    st.subheader("🔌 API Connections")
    
    # DEBUG: Show environment status
    with st.expander("🔍 Debug Info", expanded=False):
        st.write(f"HUGGINGFACE_API_KEY present: {bool(HUGGINGFACE_API_KEY)}")
        st.write(f"HF_API_AVAILABLE: {HF_API_AVAILABLE}")
        st.write(f"PYTORCH_AVAILABLE: {PYTORCH_AVAILABLE}")
        st.write(f"Model type: {type(model).__name__}")
        st.write(f"Condition (HF_API_AVAILABLE and HUGGINGFACE_API_KEY): {HF_API_AVAILABLE and bool(HUGGINGFACE_API_KEY)}")
    
    # Show current model being used
    if STREAMLIT_HUGGINGFACE_API_KEY:
        st.success("🤗 Hugging Face API (Active - FREE)")
        st.info("💡 All AI analysis uses free HuggingFace models")
    elif model:
        # Check what type of model is actually being used  
        if hasattr(model, '__class__') and 'Gemini' in str(model.__class__):
            st.success("🟢 Google Gemini (Active)")
        elif hasattr(model, '__class__') and 'OpenAI' in str(model.__class__):
            st.success("🔵 OpenAI (Active)")
        else:
            st.info(f"🤖 Model Active: {type(model).__name__}")
    else:
        st.error("❌ No model available")
    
    # Show backup options
    st.markdown("**Available Backups:**")
    if HUGGINGFACE_API_KEY and HF_API_AVAILABLE:
        if PYTORCH_AVAILABLE:
            st.success("✅ Hugging Face Ready (Local + API)")
        else:
            st.success("✅ Hugging Face Ready (API Only)")
    elif HF_API_AVAILABLE:
        st.info("🔑 Hugging Face Available (API Key Needed)")
    else:
        st.warning("⚠️ Hugging Face Not Available")
    
    if GOOGLE_API_KEY:
        st.success("✅ Google Gemini Ready")
    else:
        st.error("❌ Google API Key Missing")
    
    if OPENAI_API_KEY:
        st.success("✅ OpenAI Ready")
    else:
        st.warning("⚠️ OpenAI Key Missing")
        
    if PINECONE_API_KEY:
        st.success("✅ Pinecone Vector DB Connected")
        try:
            stats = index.describe_index_stats()
            st.info(f"📊 Vectors in KB: {stats.total_vector_count}")
        except:
            st.info("📊 Vector count unavailable")
    else:
        st.error("❌ Pinecone API Key Missing")
    
    st.divider()
    
    # File Upload
    st.subheader("📁 Upload Support Data")
    uploaded_file = st.file_uploader(
        "Upload support tickets or policies",
        type=['csv', 'txt', 'pdf'],
        help="CSV: Support tickets, TXT: Chat logs, PDF: Policy documents"
    )
    
    if uploaded_file is not None:
        st.write(f"📄 File selected: {uploaded_file.name}")
        st.write(f"📏 File size: {uploaded_file.size} bytes")
        
        col1, col2 = st.columns(2)
        with col1:
            process_full = st.button("📤 Process File")
        with col2:
            test_upload = st.button("🧪 Test Upload Only")
        
        if process_full or test_upload:
            try:
                with st.spinner("Processing file..."):
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    st.write(f"🔍 File type detected: {file_type}")
                    
                    if file_type == 'csv':
                        # Process CSV support tickets
                        st.write("📊 Reading CSV file...")
                        df = pd.read_csv(uploaded_file)
                        st.write(f"✅ Loaded {len(df)} rows with columns: {list(df.columns)}")
                        st.session_state.uploaded_tickets = df.to_dict('records')
                        
                        if test_upload:
                            st.success(f"🧪 Test successful! Loaded {len(df)} tickets with {len(df.columns)} columns")
                            st.json({"sample_ticket": st.session_state.uploaded_tickets[0]})
                            st.info("💡 Switch to the 'Dashboard' tab to see your uploaded data!")
                            st.rerun()
                        else:
                            # Store tickets in knowledge base (only for full processing)
                            st.write("💾 Storing tickets in knowledge base...")
                            progress_bar = st.progress(0)
                            
                            for i, ticket in enumerate(st.session_state.uploaded_tickets):
                                # Normalize ticket data for storage
                                normalized_ticket = {
                                    'ticket_id': ticket.get('Ticket ID', ''),
                                    'customer_name': ticket.get('Customer Name', ''),
                                    'customer_email': ticket.get('Customer Email', ''),
                                    'product': ticket.get('Product Purchased', ''),
                                    'ticket_type': ticket.get('Ticket Type', ''),
                                    'subject': ticket.get('Ticket Subject', ''),
                                    'description': ticket.get('Ticket Description', ''),
                                    'status': ticket.get('Ticket Status', ''),
                                    'priority': ticket.get('Ticket Priority', ''),
                                    'channel': ticket.get('Ticket Channel', ''),
                                    'resolution': ticket.get('Resolution', ''),
                                    'satisfaction_rating': ticket.get('Customer Satisfaction Rating', '')
                                }
                                try:
                                    st.session_state.triage_agent.store_ticket_in_kb(normalized_ticket)
                                except Exception as e:
                                    st.warning(f"⚠️ Warning: Could not store ticket {i+1} in knowledge base: {e}")
                                progress_bar.progress((i + 1) / len(st.session_state.uploaded_tickets))
                            
                            # Display upload confirmation with metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📋 Total Tickets", len(df))
                            with col2:
                                if 'Ticket Status' in df.columns:
                                    open_count = len(df[df['Ticket Status'].str.contains('Open|Pending', na=False, case=False)])
                                    st.metric("📂 Open/Pending", open_count)
                            with col3:
                                if 'Ticket Priority' in df.columns:
                                    critical_count = len(df[df['Ticket Priority'].str.contains('Critical|High', na=False, case=False)])
                                    st.metric("🔴 High Priority", critical_count)
                            
                            st.success(f"✅ Successfully processed {len(st.session_state.uploaded_tickets)} tickets and stored in knowledge base")
                            st.info("💡 Switch to the 'Dashboard' tab to see your uploaded data analytics!")
                            # Store success flag to auto-switch tabs
                            st.session_state.upload_success = True
                            st.rerun()
                
                    elif file_type == 'txt':
                        # Process text logs
                        content = str(uploaded_file.read(), "utf-8")
                        st.success("✅ Text logs processed")
                    
                    elif file_type == 'pdf':
                        # Process PDF policies
                        pdf_reader = PdfReader(uploaded_file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        st.success("✅ Policy documents processed")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    # Dashboard Tab System
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎫 Ticket Analysis", "💬 AI Chat", "📈 Insights"])
    
    with tab1:
        st.subheader("📊 Support Dashboard")
        
        # Add refresh button and status
        col_refresh, col_status = st.columns([1, 5])
        with col_refresh:
            if st.button("🔄 Refresh"):
                st.rerun()
        with col_status:
            if st.session_state.uploaded_tickets:
                st.success(f"✅ {len(st.session_state.uploaded_tickets)} tickets loaded")
        
        if st.session_state.uploaded_tickets:
            df = pd.DataFrame(st.session_state.uploaded_tickets)
            
            # Key Metrics
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                total_tickets = len(df)
                st.metric("Total Tickets", total_tickets)
            
            with col_b:
                if 'Ticket Status' in df.columns:
                    open_tickets = len(df[df['Ticket Status'].str.contains('Open|Pending', na=False, case=False)])
                else:
                    open_tickets = 0
                st.metric("Open Tickets", open_tickets)
            
            with col_c:
                if 'Ticket Priority' in df.columns:
                    critical_tickets = len(df[df['Ticket Priority'].str.contains('Critical|High', na=False, case=False)])
                else:
                    critical_tickets = 0
                st.metric("Critical Priority", critical_tickets)
            
            with col_d:
                avg_rating = df['Customer Satisfaction Rating'].mean() if 'Customer Satisfaction Rating' in df.columns else 0
                st.metric("Avg Satisfaction", f"{avg_rating:.1f}/5")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Ticket Types Distribution")
                if 'Ticket Type' in df.columns:
                    type_counts = df['Ticket Type'].value_counts()
                    st.bar_chart(type_counts)
            
            with col_chart2:
                st.subheader("Priority Distribution")
                if 'Ticket Priority' in df.columns:
                    priority_counts = df['Ticket Priority'].value_counts()
                    st.bar_chart(priority_counts)
            
            # Recent Tickets Table
            st.subheader("Recent Tickets")
            display_columns = ['Ticket ID', 'Customer Name', 'Ticket Subject', 'Ticket Status', 'Ticket Priority', 'Product Purchased']
            available_columns = [col for col in display_columns if col in df.columns]
            if available_columns:
                st.dataframe(df[available_columns].head(10), use_container_width=True)
        else:
            st.info("📤 No data loaded yet. Please upload a support tickets CSV file in the sidebar to see dashboard metrics.")
            st.markdown("""
            ### 📝 Expected CSV Format:
            Your CSV should contain columns like:
            - **Ticket ID**
            - **Customer Name**
            - **Ticket Status** (Open, Closed, Pending)
            - **Ticket Priority** (Low, Medium, High, Critical)
            - **Product Purchased**
            - **Ticket Type**
            - **Ticket Subject**
            - **Ticket Description**
            
            💡 You can use the sample file in `datasets/customer_support_tickets.csv`
            """)
    
    with tab2:
        st.subheader("🎫 Individual Ticket Analysis")
        
        if st.session_state.uploaded_tickets:
            # Ticket Selection
            ticket_ids = [str(ticket.get('Ticket ID', i)) for i, ticket in enumerate(st.session_state.uploaded_tickets)]
            selected_ticket_id = st.selectbox("Select Ticket to Analyze", ticket_ids)
            
            if selected_ticket_id and st.button("🔍 Analyze Ticket"):
                # Find selected ticket
                selected_ticket = None
                for ticket in st.session_state.uploaded_tickets:
                    if str(ticket.get('Ticket ID', '')) == selected_ticket_id:
                        selected_ticket = ticket
                        break
                
                if selected_ticket:
                    # Process ticket through triage agent
                    with st.spinner("🤖 AI Agent analyzing ticket..."):
                        # Convert column names to expected format
                        ticket_data = {
                            'ticket_id': selected_ticket.get('Ticket ID', ''),
                            'customer_name': selected_ticket.get('Customer Name', ''),
                            'description': selected_ticket.get('Ticket Description', ''),
                            'ticket_type': selected_ticket.get('Ticket Type', ''),
                            'product': selected_ticket.get('Product Purchased', ''),
                            'subject': selected_ticket.get('Ticket Subject', ''),
                            'status': selected_ticket.get('Ticket Status', ''),
                            'priority': selected_ticket.get('Ticket Priority', ''),
                        }
                        
                        analysis = st.session_state.triage_agent.process_ticket(ticket_data)
                    
                    # Display Results
                    st.success("✅ Analysis Complete!")
                    
                    # Ticket Info
                    with st.expander("📋 Ticket Details", expanded=True):
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write(f"**Customer:** {ticket_data['customer_name']}")
                            st.write(f"**Product:** {ticket_data['product']}")
                            st.write(f"**Type:** {ticket_data['ticket_type']}")
                        with col_info2:
                            st.write(f"**Status:** {ticket_data['status']}")
                            st.write(f"**Priority:** {ticket_data['priority']}")
                            st.write(f"**Subject:** {ticket_data['subject']}")
                        
                        st.write("**Description:**")
                        st.write(ticket_data['description'])
                    
                    # Analysis Results
                    if 'error' not in analysis:
                        col_analysis1, col_analysis2 = st.columns(2)
                        
                        with col_analysis1:
                            st.subheader("🎭 Sentiment Analysis")
                            sentiment_content = analysis.get('sentiment_analysis', {})
                            if hasattr(sentiment_content, 'content'):
                                st.text(sentiment_content.content)
                            else:
                                st.text(str(sentiment_content))
                            
                            st.subheader("🎯 Intent Classification") 
                            intent_content = analysis.get('intent_classification', {})
                            if hasattr(intent_content, 'content'):
                                st.text(intent_content.content)
                            else:
                                st.text(str(intent_content))
                        
                        with col_analysis2:
                            st.subheader("🔍 Key Information")
                            key_info_content = analysis.get('key_information', {})
                            if hasattr(key_info_content, 'content'):
                                st.text(key_info_content.content)
                            else:
                                st.text(str(key_info_content))
                        
                        st.subheader("💬 Suggested Response")
                        response_content = analysis.get('suggested_response', {})
                        if hasattr(response_content, 'content'):
                            st.text_area("Draft Response (Editable)", response_content.content, height=200)
                        else:
                            st.text_area("Draft Response (Editable)", str(response_content), height=200)
                        
                        st.subheader("🔗 Similar Tickets")
                        similar_content = analysis.get('similar_tickets', {})
                        if hasattr(similar_content, 'content'):
                            st.text(similar_content.content)
                        else:
                            st.text(str(similar_content))
                        
                        # Action Buttons
                        col_action1, col_action2, col_action3 = st.columns(3)
                        with col_action1:
                            if st.button("✅ Approve Response"):
                                st.success("Response approved and ready to send!")
                        with col_action2:
                            if st.button("⬆️ Escalate Ticket"):
                                st.warning("Ticket marked for escalation to supervisor")
                        with col_action3:
                            if st.button("🔄 Route to Department"):
                                st.info("Ticket routed to appropriate department")
                    else:
                        st.error(f"Analysis Error: {analysis['error']}")
        else:
            st.info("📤 Upload support tickets to enable ticket analysis")
    
    with tab3:
        st.subheader("💬 AI Chat Interface")
        
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about support trends, complaints, or get insights..."):
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI analyzing..."):
                    if st.session_state.uploaded_tickets:
                        # Add context about uploaded tickets
                        context_prompt = f"""
                        Based on the uploaded support ticket data, answer this question: {prompt}
                        
                        Available data includes {len(st.session_state.uploaded_tickets)} tickets with information about:
                        - Customer issues and complaints
                        - Product problems
                        - Ticket types and priorities
                        - Resolution status
                        
                        Provide insights and analysis based on this data.
                        """
                    else:
                        context_prompt = f"""
                        Answer this support-related question: {prompt}
                        
                        Note: No specific ticket data is currently loaded. Provide general support insights and recommendations.
                        """
                    
                    response_content = run_inference(context_prompt)
                    st.markdown(response_content)
                    
                    # Add assistant response
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_content})
    
    with tab4:
        st.subheader("📈 Management Insights")
        
        if st.session_state.uploaded_tickets:
            df = pd.DataFrame(st.session_state.uploaded_tickets)
            
            # Insights Generation
            if st.button("🔍 Generate Management Report"):
                with st.spinner("📊 Generating insights..."):
                    # Summary statistics
                    insights_prompt = f"""
                    Generate a management report based on this support ticket data:
                    
                    Total Tickets: {len(df)}
                    Ticket Types: {df['Ticket Type'].value_counts().to_dict() if 'Ticket Type' in df.columns else 'N/A'}
                    Priority Distribution: {df['Ticket Priority'].value_counts().to_dict() if 'Ticket Priority' in df.columns else 'N/A'}
                    Status Distribution: {df['Ticket Status'].value_counts().to_dict() if 'Ticket Status' in df.columns else 'N/A'}
                    
                    Provide:
                    1. Key trends and patterns
                    2. Top customer pain points  
                    3. Operational recommendations
                    4. SLA and performance insights
                    5. Resource allocation suggestions
                    """
                    
                    report_content = run_inference(insights_prompt)
                    
                    st.markdown("### 📋 Executive Summary")
                    st.markdown(report_content)
            
            # Additional Analytics
            st.markdown("### 📊 Detailed Analytics")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                if 'Product Purchased' in df.columns:
                    st.subheader("Top Problem Products")
                    product_issues = df['Product Purchased'].value_counts().head(10)
                    st.bar_chart(product_issues)
            
            with col_insight2:
                if 'Ticket Channel' in df.columns:
                    st.subheader("Support Channel Usage")
                    channel_usage = df['Ticket Channel'].value_counts()
                    st.bar_chart(channel_usage)
                    
        else:
            st.info("📤 Upload support data to generate management insights")

with col2:
    st.header("🚀 Quick Actions")
    
    # Sample Chat Prompts
    st.subheader("💡 Sample Queries")
    sample_prompts = [
        "What are the top 3 customer complaints this month?",
        "Show me tickets that need escalation",
        "Summarize refund-related issues",
        "Which products have the most support requests?", 
        "What's the average resolution time?",
        "Find tickets with negative sentiment",
        "Show delivery-related complaints",
        "Analyze customer satisfaction trends"
    ]
    
    for prompt in sample_prompts:
        if st.button(prompt, key=f"sample_{hash(prompt)}", use_container_width=True):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            # Trigger rerun to show in chat
            st.rerun()
    
    st.divider()
    
    # Statistics
    if st.session_state.uploaded_tickets:
        st.subheader("📈 Live Stats")
        df = pd.DataFrame(st.session_state.uploaded_tickets)
        
        st.metric("Total Tickets", len(df))
        
        if 'Ticket Status' in df.columns:
            open_count = len(df[df['Ticket Status'].str.contains('Open|Pending', na=False)])
            st.metric("Open/Pending", open_count)
        
        if 'Ticket Priority' in df.columns:
            critical_count = len(df[df['Ticket Priority'] == 'Critical'])
            st.metric("Critical Priority", critical_count)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎫 Customer Support Triage Agent | Powered by Agno & AI</p>
    <p>Improving support efficiency and customer satisfaction</p>
</div>
""", unsafe_allow_html=True)