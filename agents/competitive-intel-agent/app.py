"""
Competitive Intel Agent - AI-Powered Competitive Analysis with Agentic RAG
Uses Cohere embeddings and LlamaIndex for intelligent competitor analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional, Tuple
import json

# Load environment variables from root directory
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)

# Get API keys
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import cohere
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.embeddings.cohere import CohereEmbedding
    from llama_index.llms.cohere import Cohere
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    logger.error(f"Missing dependencies: {e}")

class CompetitiveAnalysisAgent:
    """AI Agent for competitive analysis using ReAct framework"""
    
    def __init__(self, api_key: str):
        """Initialize the agent with Cohere API key"""
        self.api_key = api_key
        self.cohere_client = None
        self.index = None
        self.query_history = []
        self.reasoning_logs = []
        
        if api_key and IMPORTS_AVAILABLE:
            try:
                # Initialize Cohere client
                self.cohere_client = cohere.Client(api_key)
                
                # Configure LlamaIndex settings with Cohere
                Settings.embed_model = CohereEmbedding(
                    api_key=api_key,
                    model_name="embed-english-v3.0",
                    input_type="search_document"
                )
                Settings.llm = Cohere(
                    api_key=api_key,
                    model="command-r-plus"
                )
                
                logger.info("Cohere client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Cohere: {e}")
    
    def load_competitor_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess competitor data from CSV"""
        try:
            df = pd.read_csv(csv_path)
            
            # Clean and preprocess text
            text_columns = ['Product Description', 'Marketing Strategy', 'Financial Summary', 'Strengths', 'Weaknesses']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('')
                    df[col] = df[col].str.strip()
                    df[col] = df[col].str.replace(r'[^\w\s\.\,\!\?\-]', '', regex=True)
            
            logger.info(f"Loaded {len(df)} competitor records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def index_data(self, df: pd.DataFrame):
        """Create vector index from competitor data"""
        if not self.cohere_client or df.empty:
            return
        
        try:
            # Create documents from dataframe
            documents = []
            for _, row in df.iterrows():
                # Combine all text fields into a comprehensive document
                content = f"""
                Competitor: {row.get('Competitor Name', 'Unknown')}
                Industry: {row.get('Industry', 'Unknown')}
                Product Description: {row.get('Product Description', '')}
                Marketing Strategy: {row.get('Marketing Strategy', '')}
                Financial Summary: {row.get('Financial Summary', '')}
                Market Share: {row.get('Market Share', 'Unknown')}
                Strengths: {row.get('Strengths', '')}
                Weaknesses: {row.get('Weaknesses', '')}
                Recent Updates: {row.get('Recent Updates', '')}
                """
                
                doc = Document(text=content, metadata={
                    'competitor': row.get('Competitor Name', 'Unknown'),
                    'industry': row.get('Industry', 'Unknown')
                })
                documents.append(doc)
            
            # Create vector index
            self.index = VectorStoreIndex.from_documents(documents)
            logger.info(f"Indexed {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error indexing data: {e}")
    
    def reason_and_act(self, query: str) -> Tuple[str, List[str]]:
        """
        ReAct framework: Reason about the query and act accordingly
        Returns: (response, reasoning_steps)
        """
        reasoning_steps = []
        
        try:
            # Step 1: Analyze query intent
            reasoning_steps.append(f"🔍 Analyzing query: '{query}'")
            
            query_lower = query.lower()
            intent = self._determine_intent(query_lower)
            reasoning_steps.append(f"📊 Determined intent: {intent}")
            
            # Step 2: Break down into sub-goals
            sub_goals = self._create_subgoals(intent, query_lower)
            reasoning_steps.append(f"🎯 Sub-goals: {', '.join(sub_goals)}")
            
            # Step 3: Execute sub-goals
            results = []
            for goal in sub_goals:
                reasoning_steps.append(f"⚡ Executing: {goal}")
                result = self._execute_subgoal(goal, query)
                if result:
                    results.append(result)
            
            # Step 4: Generate final response
            if results:
                reasoning_steps.append("✅ Generating comprehensive response")
                response = self._generate_response(query, results)
            else:
                response = "I couldn't find relevant information for your query. Please try rephrasing or ask about specific competitors."
            
            # Store in history
            self.query_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'reasoning': reasoning_steps
            })
            
            return response, reasoning_steps
            
        except Exception as e:
            logger.error(f"Error in reason_and_act: {e}")
            reasoning_steps.append(f"❌ Error: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}", reasoning_steps
    
    def _determine_intent(self, query: str) -> str:
        """Determine the intent of the query"""
        if 'compare' in query or 'vs' in query or 'versus' in query:
            return 'comparison'
        elif 'strength' in query or 'advantage' in query:
            return 'strengths'
        elif 'weakness' in query or 'disadvantage' in query:
            return 'weaknesses'
        elif 'strategy' in query or 'marketing' in query:
            return 'strategy'
        elif 'financial' in query or 'revenue' in query or 'market share' in query:
            return 'financial'
        elif 'all' in query or 'everything' in query or 'overview' in query:
            return 'overview'
        else:
            return 'general'
    
    def _create_subgoals(self, intent: str, query: str) -> List[str]:
        """Create sub-goals based on intent"""
        sub_goals = []
        
        if intent == 'comparison':
            sub_goals = ['retrieve_competitors', 'analyze_differences', 'generate_comparison']
        elif intent == 'strengths':
            sub_goals = ['retrieve_strengths', 'analyze_advantages']
        elif intent == 'weaknesses':
            sub_goals = ['retrieve_weaknesses', 'analyze_disadvantages']
        elif intent == 'strategy':
            sub_goals = ['retrieve_strategies', 'analyze_tactics']
        elif intent == 'financial':
            sub_goals = ['retrieve_financial', 'analyze_performance']
        elif intent == 'overview':
            sub_goals = ['retrieve_all', 'summarize_info']
        else:
            sub_goals = ['retrieve_relevant', 'generate_insights']
        
        return sub_goals
    
    def _execute_subgoal(self, goal: str, query: str) -> Optional[str]:
        """Execute a specific sub-goal"""
        if not self.index:
            return None
        
        try:
            # Create query engine
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3
            )
            query_engine = RetrieverQueryEngine(retriever=retriever)
            
            # Execute query based on goal
            if 'retrieve' in goal:
                response = query_engine.query(query)
                return str(response)
            elif 'analyze' in goal:
                # Use Cohere for analysis
                if self.cohere_client:
                    prompt = f"Analyze the following competitive information for: {query}"
                    response = self.cohere_client.generate(
                        prompt=prompt,
                        model='command-r-plus',
                        max_tokens=300
                    )
                    return response.generations[0].text
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing sub-goal {goal}: {e}")
            return None
    
    def _generate_response(self, query: str, results: List[str]) -> str:
        """Generate final response from sub-goal results"""
        if not self.cohere_client:
            return "\n\n".join(results)
        
        try:
            combined_context = "\n\n".join(results)
            prompt = f"""Based on the following competitive analysis data, provide a comprehensive answer to the user's query.

Query: {query}

Data:
{combined_context}

Provide actionable insights and specific recommendations where applicable."""

            response = self.cohere_client.generate(
                prompt=prompt,
                model='command-r-plus',
                max_tokens=500,
                temperature=0.7
            )
            
            return response.generations[0].text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return combined_context
    
    def get_recent_queries(self, n: int = 5) -> List[Dict]:
        """Get recent query history"""
        return self.query_history[-n:] if self.query_history else []

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Competitive Intel Agent",
        page_icon="🔍",
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
        .reasoning-box {
            background: #f0f4f8;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
        }
        .query-history {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Competitive Intel Agent</h1>
        <p>AI-Powered Competitive Analysis with Agentic RAG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        if COHERE_API_KEY:
            st.success("✅ Cohere API Connected")
            masked_key = COHERE_API_KEY[:10] + "*" * (len(COHERE_API_KEY) - 14) + COHERE_API_KEY[-4:]
            st.info(f"Key: {masked_key}")
        else:
            st.error("❌ Cohere API Key Missing")
            st.warning("Add COHERE_API_KEY to .env file")
        
        if not IMPORTS_AVAILABLE:
            st.error("❌ Missing Dependencies")
            st.code("pip install cohere llama-index llama-index-llms-cohere llama-index-embeddings-cohere")
        
        st.divider()
        
        # Load data button
        if st.button("🔄 Load/Reload Data", type="primary"):
            if COHERE_API_KEY and IMPORTS_AVAILABLE:
                with st.spinner("Loading competitive data..."):
                    # Initialize agent
                    st.session_state.agent = CompetitiveAnalysisAgent(COHERE_API_KEY)
                    
                    # Load data
                    data_path = current_dir / "data" / "competitors.csv"
                    if data_path.exists():
                        df = st.session_state.agent.load_competitor_data(str(data_path))
                        st.session_state.agent.index_data(df)
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded {len(df)} competitors")
                    else:
                        st.error("❌ competitors.csv not found")
        
        st.divider()
        
        # Query history
        st.header("📜 Recent Queries")
        if st.session_state.agent:
            history = st.session_state.agent.get_recent_queries(5)
            if history:
                for item in reversed(history):
                    with st.expander(f"🕐 {item['query'][:50]}..."):
                        st.write(item['response'][:200] + "...")
            else:
                st.info("No queries yet")
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load/Reload Data' in the sidebar to begin")
        
        # Show sample data structure
        with st.expander("📊 Expected Data Structure"):
            st.markdown("""
            The system expects a CSV file with the following columns:
            - **Competitor Name**: Company name
            - **Industry**: Industry sector
            - **Product Description**: Main products/services
            - **Marketing Strategy**: Marketing approach
            - **Financial Summary**: Revenue, growth, etc.
            - **Market Share**: Market position
            - **Strengths**: Key advantages
            - **Weaknesses**: Areas of improvement
            - **Recent Updates**: Latest news/changes
            """)
    else:
        # Query interface
        st.header("💬 Ask About Competitors")
        
        # Sample queries
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Compare TechCorp vs DataSoft"):
                query = "Compare the marketing strategies of TechCorp and DataSoft"
                st.session_state.chat_history.append(("sample", query))
        with col2:
            if st.button("💪 CloudNet Strengths"):
                query = "What are CloudNet's main competitive advantages?"
                st.session_state.chat_history.append(("sample", query))
        with col3:
            if st.button("💰 Financial Overview"):
                query = "Give me a financial overview of all competitors"
                st.session_state.chat_history.append(("sample", query))
        
        # Query input
        query = st.text_area(
            "Enter your competitive analysis query:",
            placeholder="E.g., 'Compare the strengths and weaknesses of TechCorp and DataSoft' or 'What is CloudNet's marketing strategy?'",
            height=100
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_button = st.button("🔍 Analyze", type="primary")
        with col2:
            show_reasoning = st.checkbox("Show reasoning steps", value=True)
        
        # Process query
        if analyze_button and query:
            with st.spinner("🤔 Analyzing competitive landscape..."):
                response, reasoning_steps = st.session_state.agent.reason_and_act(query)
                
                # Show reasoning if enabled
                if show_reasoning:
                    with st.expander("🧠 Reasoning Process", expanded=True):
                        for step in reasoning_steps:
                            st.markdown(f"• {step}")
                
                # Show response
                st.markdown("### 💡 Analysis Results")
                st.markdown(f'<div class="reasoning-box">{response}</div>', unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.chat_history.append((query, response))
        
        # Process sample queries
        if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "sample":
            query = st.session_state.chat_history[-1][1]
            with st.spinner("🤔 Analyzing competitive landscape..."):
                response, reasoning_steps = st.session_state.agent.reason_and_act(query)
                
                if show_reasoning:
                    with st.expander("🧠 Reasoning Process", expanded=True):
                        for step in reasoning_steps:
                            st.markdown(f"• {step}")
                
                st.markdown("### 💡 Analysis Results")
                st.markdown(f'<div class="reasoning-box">{response}</div>', unsafe_allow_html=True)
                
                # Update chat history
                st.session_state.chat_history[-1] = (query, response)

if __name__ == "__main__":
    main()