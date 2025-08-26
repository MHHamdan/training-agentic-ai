import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_core.runnables import RunnableMap
import google.generativeai as genai
from dotenv import load_dotenv
import re
import uuid
from datetime import datetime
import traceback

# LangSmith imports for observability
from langsmith import Client, traceable

# Load environment variables
load_dotenv()

# Configure API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "resume-screening-demo")

# Validate required API keys
if not GOOGLE_API_KEY:
    st.error("❌ Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

if not LANGCHAIN_API_KEY:
    st.error("❌ Please set your LANGCHAIN_API_KEY in a .env file for LangSmith integration")
    st.stop()

# Configure LangSmith environment
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Initialize clients
genai.configure(api_key=GOOGLE_API_KEY)
langsmith_client = Client()

# Setup embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create or load Chroma vector store
VECTOR_STORE_DIR = "chroma_store"
if os.path.exists(VECTOR_STORE_DIR):
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

@traceable(name="extract_resume_text")
def extract_text_from_resume(file):
    """Extract text from uploaded resume files with LangSmith tracing"""
    temp_file_path = f"temp_{file.name}"
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(file.getbuffer())

        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        
        return {
            "text": text,
            "file_name": file.name,
            "file_type": file_extension,
            "text_length": len(text),
            "pages": len(documents)
        }
        
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@traceable(name="split_text")
def split_text(text):
    """Split text into chunks with LangSmith tracing"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = splitter.create_documents([text])
    
    return {
        "documents": documents,
        "chunk_count": len(documents),
        "avg_chunk_size": sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0
    }

@traceable(name="store_resume_analysis")
def store_resume_analysis(resume_text, analysis, doc_id):
    """Store resume analysis in vector store with LangSmith observability"""
    try:
        split_result = split_text(analysis)
        documents = split_result["documents"]
        
        vectorstore.add_documents(
            documents, 
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))]
        )
        vectorstore.persist()
        
        return {
            "status": "success", 
            "chunks_stored": len(documents),
            "doc_id": doc_id,
            "resume_length": len(resume_text),
            "analysis_length": len(analysis),
            "avg_chunk_size": split_result["avg_chunk_size"]
        }
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

def extract_suitability_score(text):
    """Extract percentage score from analysis text"""
    match = re.search(r"Suitability Score: (\d{1,3})%", text)
    return int(match.group(1)) if match else None

@traceable(name="analyze_resume_complete")
def analyze_resume_with_langsmith(job_requirements, resume_data):
    """Complete resume analysis with LangSmith tracing"""
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    prompt_template = PromptTemplate(
        input_variables=["job_requirements", "resume_text"],
        template="""
        You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

        Job Requirements:
        {job_requirements}

        Resume:
        {resume_text}

        Provide a comprehensive structured analysis covering:
        1. **Skills Match**: How well candidate's skills align with requirements
        2. **Experience Relevance**: Relevant work experience analysis  
        3. **Education & Certifications**: Educational background assessment
        4. **Strengths**: Key strengths of the candidate
        5. **Areas for Improvement**: What's missing or could be better
        6. **Overall Assessment**: Summary recommendation

        At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well the resume aligns with the job.
        Format: Suitability Score: XX%
        """
    )

    # Create chain with LangSmith tracing
    chain = (
        RunnableMap({
            "job_requirements": lambda x: x["job_requirements"],
            "resume_text": lambda x: x["resume_text"]
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Execute chain
    chain_input = {
        "job_requirements": job_requirements,
        "resume_text": resume_data["text"]
    }
    
    analysis = chain.invoke(chain_input)
    suitability_score = extract_suitability_score(analysis)
    
    return {
        "analysis": analysis,
        "suitability_score": suitability_score,
        "file_info": resume_data,
        "job_requirements_length": len(job_requirements),
        "analysis_length": len(analysis),
        "processing_metadata": {
            "model": "gemini-2.0-flash",
            "temperature": 0.2,
            "timestamp": datetime.now().isoformat()
        }
    }

def display_score_feedback(score):
    """Display color-coded feedback based on suitability score"""
    if score >= 80:
        st.success(f"🎯 **Excellent Match!** Score: {score}%")
        st.balloons()
    elif score >= 60:
        st.info(f"👍 **Good Match!** Score: {score}%")
    elif score >= 40:
        st.warning(f"⚠️ **Moderate Match** Score: {score}%")
    else:
        st.error(f"❌ **Low Match** Score: {score}%")

def main():
    st.set_page_config(
        page_title="Resume Screening with LangSmith", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with LangSmith integration info
    st.title("🔍 AI Resume Screening with LangSmith Observability")
    st.markdown(f"📊 **[View Live Traces in LangSmith Dashboard →](https://smith.langchain.com/projects/{LANGCHAIN_PROJECT})**")

    # Sidebar for LangSmith info and session tracking
    with st.sidebar:
        st.header("🔍 LangSmith Observability")
        st.success("✅ LangSmith Connected")
        st.info(f"**Project**: `{LANGCHAIN_PROJECT}`")
        
        # Session tracking
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
        
        st.metric("Session ID", st.session_state.session_id)
        st.metric("Timestamp", datetime.now().strftime('%H:%M:%S'))
        
        # Statistics
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.metric("Analyses Completed", st.session_state.analysis_count)

    # Main interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📝 Job Requirements")
        job_requirements = st.text_area(
            "Enter the job requirements and qualifications",
            height=300,
            placeholder="e.g., Required: Python, Machine Learning, 3+ years experience..."
        )
    
    with col2:
        st.header("📄 Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload candidate's resume", 
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT"
        )

    # Analysis section
    if st.button("🚀 Analyze Resume", type="primary") and uploaded_file and job_requirements:
        with st.spinner("🔄 Processing with LangSmith tracking..."):
            try:
                # Extract resume text with tracing
                resume_data = extract_text_from_resume(uploaded_file)
                
                # Display resume preview
                with st.expander("👀 View Extracted Resume Text"):
                    st.text_area("Resume Content", resume_data["text"], height=200)
                    st.json({
                        "File Name": resume_data["file_name"],
                        "File Type": resume_data["file_type"],
                        "Text Length": f"{resume_data['text_length']:,} characters",
                        "Pages/Sections": resume_data["pages"]
                    })

                # Analyze resume with LangSmith tracing
                result = analyze_resume_with_langsmith(job_requirements, resume_data)
                
                st.header("🤖 AI Analysis Results")
                st.markdown(result["analysis"])

                # Display suitability score with feedback
                if result["suitability_score"] is not None:
                    st.header("📊 Suitability Assessment")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric(
                            label="Suitability Score", 
                            value=f"{result['suitability_score']}%",
                            delta=f"{result['suitability_score'] - 50}% vs baseline"
                        )
                    
                    with col2:
                        display_score_feedback(result["suitability_score"])

                # Store analysis in vector DB
                storage_result = store_resume_analysis(
                    resume_data["text"], 
                    result["analysis"], 
                    os.path.splitext(uploaded_file.name)[0]
                )
                
                if storage_result["status"] == "success":
                    st.success("✅ Analysis stored in vector database and tracked in LangSmith")
                    
                    # Display storage metrics
                    with st.expander("📈 Storage Metrics"):
                        st.json(storage_result)

                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📥 Download Analysis",
                        result["analysis"],
                        file_name=f"analysis_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.markdown(f"**[🔍 View Detailed Traces →](https://smith.langchain.com/projects/{LANGCHAIN_PROJECT})**")

                # Update session statistics
                st.session_state.analysis_count += 1
                
                st.info("🎉 Analysis complete! Check LangSmith dashboard for detailed observability traces.")

            except Exception as e:
                st.error(f"❌ **Error occurred**: {str(e)}")
                
                with st.expander("🐛 Debug Information"):
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()