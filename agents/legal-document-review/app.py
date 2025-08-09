import streamlit as st
import os
from typing import List, Optional
import tempfile
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# PDF processing
import PyPDF2

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal Document Review Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stAlert {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

class LegalDocumentProcessor:
    """Main class for processing legal documents"""
    
    def __init__(self, api_key: str):
        """Initialize the processor with API key"""
        self.api_key = api_key
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.setup_models()
    
    def setup_models(self):
        """Setup Gemini models for embeddings and generation"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key
            )
            
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.3,
                max_output_tokens=2048
            )
        except Exception as e:
            st.error(f"Error setting up models: {str(e)}")
    
    def extract_pdf_text(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                st.warning("⚠️ This PDF is encrypted. Attempting to decrypt...")
                try:
                    pdf_reader.decrypt("")  # Try with empty password
                except:
                    st.error("❌ Unable to decrypt the PDF. Please provide an unencrypted version.")
                    return None
            
            # Extract text from all pages
            num_pages = len(pdf_reader.pages)
            progress_bar = st.progress(0)
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                progress_bar.progress((i + 1) / num_pages)
            
            progress_bar.empty()
            
            if not text.strip():
                st.error("❌ No text found in PDF. This might be a scanned document.")
                return None
                
            return text
            
        except Exception as e:
            st.error(f"❌ Error reading PDF: {str(e)}")
            return None
    
    def chunk_text(self, text: str) -> List[Document]:
        """Split text into manageable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    
    def create_vector_store(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            return True
        except Exception as e:
            st.error(f"❌ Error creating vector store: {str(e)}")
            return False
    
    def answer_question(self, question: str) -> str:
        """Answer questions using RAG pipeline"""
        if not self.vector_store:
            return "Please upload a document first."
        
        try:
            # Custom prompt for legal context
            prompt_template = """You are a legal document expert assistant. Use the following context from the legal document to answer the question.
            
            Context: {context}
            
            Question: {question}
            
            Instructions:
            - Provide a clear and concise answer based solely on the document context
            - If the answer is not in the context, say so clearly
            - Reference specific sections or clauses when relevant
            - Use proper legal terminology where appropriate
            
            Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            result = qa_chain({"query": question})
            return result["result"]
            
        except Exception as e:
            return f"Error answering question: {str(e)}"
    
    def generate_summary(self) -> str:
        """Generate a summary of the legal document"""
        if not self.vector_store:
            return "Please upload a document first."
        
        try:
            # Retrieve top chunks for summary
            summary_query = "What are the main points, key terms, obligations, and important clauses in this legal document?"
            docs = self.vector_store.similarity_search(summary_query, k=6)
            
            # Combine relevant chunks
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Summary prompt
            summary_prompt = f"""As a legal expert, provide a comprehensive summary of the following legal document excerpts.
            
            Document Content:
            {context}
            
            Please provide:
            1. Document Type and Purpose
            2. Key Parties Involved (if mentioned)
            3. Main Terms and Conditions
            4. Important Obligations and Rights
            5. Critical Dates or Deadlines (if any)
            6. Termination or Cancellation Clauses
            7. Any Notable Restrictions or Limitations
            
            Keep the summary concise but comprehensive, highlighting the most legally significant aspects."""
            
            response = self.llm.invoke(summary_prompt)
            return response.content
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Review Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Try to get API key from environment first
        default_api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # API Key input
        api_key = st.text_input(
            "Enter Google Gemini API Key",
            type="password",
            value=default_api_key,
            placeholder="AIza...",
            help="Get your API key from Google AI Studio or set in .env file"
        )
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your API key to continue")
        
        st.divider()
        
        # Instructions
        st.header("📋 Instructions")
        st.markdown("""
        1. **Enter your API Key** above
        2. **Upload a PDF** document
        3. **Ask questions** or generate summaries
        4. **Review the AI responses**
        
        **Supported Documents:**
        - Contracts
        - NDAs
        - Terms of Service
        - License Agreements
        - Service Agreements
        """)
        
        st.divider()
        
        # Sample questions
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the termination clauses?",
            "What is the duration of this agreement?",
            "What are the payment terms?",
            "What are the confidentiality obligations?",
            "What are the key deliverables?",
            "What happens in case of breach?"
        ]
        
        for q in sample_questions:
            st.code(q, language=None)
    
    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = None
    
    # Main content area
    if not api_key:
        st.info("👈 Please enter your Google Gemini API key in the sidebar to get started.")
        return
    
    # Initialize processor
    if st.session_state.processor is None or st.session_state.processor.api_key != api_key:
        with st.spinner("Initializing AI models..."):
            st.session_state.processor = LegalDocumentProcessor(api_key)
    
    # File upload section
    st.header("📄 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a legal document (PDF)",
        type=['pdf'],
        help="Upload a legal document in PDF format for analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"📎 Uploaded: {uploaded_file.name}")
        
        with col2:
            if st.button("🔄 Process Document", type="primary"):
                with st.spinner("Extracting text from PDF..."):
                    text = st.session_state.processor.extract_pdf_text(uploaded_file)
                
                if text:
                    st.session_state.extracted_text = text
                    
                    # Show preview
                    with st.expander("📝 Preview Extracted Text (First 1000 characters)"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    # Create chunks and vector store
                    with st.spinner("Creating document embeddings..."):
                        documents = st.session_state.processor.chunk_text(text)
                        success = st.session_state.processor.create_vector_store(documents)
                    
                    if success:
                        st.session_state.document_uploaded = True
                        st.success(f"✅ Document processed successfully! Created {len(documents)} text chunks.")
    
    # Features section
    if st.session_state.document_uploaded:
        st.divider()
        
        # Create tabs for different features
        tab1, tab2, tab3 = st.tabs(["❓ Ask Questions", "📊 Generate Summary", "📈 Document Stats"])
        
        with tab1:
            st.header("Ask Questions About the Document")
            
            # Question input
            question = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the termination clauses in this agreement?",
                height=100
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                ask_button = st.button("🔍 Get Answer", type="primary")
            
            if ask_button and question:
                with st.spinner("Analyzing document and generating answer..."):
                    answer = st.session_state.processor.answer_question(question)
                
                st.markdown("### 💬 Answer:")
                st.markdown(f'<div class="feature-card">{answer}</div>', unsafe_allow_html=True)
        
        with tab2:
            st.header("Document Summary")
            
            if st.button("📝 Generate Summary", type="primary"):
                with st.spinner("Generating comprehensive summary..."):
                    summary = st.session_state.processor.generate_summary()
                
                st.markdown("### 📋 Summary:")
                st.markdown(f'<div class="feature-card">{summary}</div>', unsafe_allow_html=True)
        
        with tab3:
            st.header("Document Statistics")
            
            if st.session_state.extracted_text:
                text = st.session_state.extracted_text
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Characters", f"{len(text):,}")
                
                with col2:
                    word_count = len(text.split())
                    st.metric("Word Count", f"{word_count:,}")
                
                with col3:
                    line_count = len(text.split('\n'))
                    st.metric("Line Count", f"{line_count:,}")
                
                # Additional stats
                st.markdown("### 📊 Additional Information")
                
                # Common legal terms frequency
                legal_terms = ['agreement', 'contract', 'party', 'parties', 'shall', 'liability',
                             'termination', 'confidential', 'warranty', 'indemnify']
                
                term_counts = {}
                text_lower = text.lower()
                for term in legal_terms:
                    count = text_lower.count(term)
                    if count > 0:
                        term_counts[term] = count
                
                if term_counts:
                    st.markdown("**Common Legal Terms Found:**")
                    for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {term.capitalize()}: {count} occurrences")

if __name__ == "__main__":
    main()