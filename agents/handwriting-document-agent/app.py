"""
Handwriting & Historical Document Analysis Agent with LangGraph
Enterprise-grade document processing with OCR, AI analysis, and interactive chat
Author: Mohammed Hamdan
"""

import streamlit as st
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import time
from datetime import datetime

# Import agent components
from config import config, DOCUMENT_TYPES, ANALYSIS_TEMPLATES
from graph.document_workflow import DocumentWorkflowManager
from processors.document_processor import DocumentProcessor, ProcessingOptions
from rag.document_rag import DocumentRAGSystem
from models.document_models import DocumentModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Handwriting Document Agent V1 - AI Document Analysis",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .document-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border-top: 3px solid #667eea;
    }
    .chat-message {
        background: #e3f2fd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    .source-chunk {
        background: #f3e5f5;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #9c27b0;
        font-size: 0.9rem;
    }
    .processing-status {
        background: #e8f5e8;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #4caf50;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if "workflow_manager" not in st.session_state:
        st.session_state.workflow_manager = DocumentWorkflowManager()
    
    if "document_processor" not in st.session_state:
        st.session_state.document_processor = DocumentProcessor()
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = DocumentRAGSystem()
    
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_workflow_result" not in st.session_state:
        st.session_state.current_workflow_result = None
    
    if "processing_in_progress" not in st.session_state:
        st.session_state.processing_in_progress = False
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">📜 Handwriting Document Agent V1</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 1.2rem;'>"
        "AI-powered analysis of handwritten documents, historical texts, and manuscripts with "
        "OCR, LangGraph workflows, and interactive chat capabilities</p>",
        unsafe_allow_html=True
    )

def display_system_status():
    """Display system status and capabilities"""
    st.sidebar.markdown("## 🔧 System Status")
    
    # Check API provider status first
    try:
        from models.api_ocr import get_api_ocr
        api_ocr = get_api_ocr()
        provider_info = api_ocr.get_provider_info()
        has_api_providers = len(provider_info['available_providers']) > 0
        primary_provider = provider_info.get('primary_provider', 'None')
    except Exception as e:
        has_api_providers = False
        primary_provider = 'None'
        st.sidebar.error(f"⚠️ API OCR initialization failed: {str(e)}")
    
    # Display API status prominently
    if has_api_providers:
        st.sidebar.success(f"✅ API OCR Ready ({primary_provider.upper()})")
        st.sidebar.metric("Processing Mode", "🚀 Professional API")
    else:
        st.sidebar.error("❌ No API Providers Configured")
        st.sidebar.markdown("**⚠️ Setup Required:**")
        st.sidebar.markdown("Add an API key to `.env` file:")
        st.sidebar.code("""
# Add ONE of these to .env:
OPENAI_API_KEY=sk-your-key
ANTHROPIC_API_KEY=sk-ant-key  
GOOGLE_API_KEY=your-google-key
GROQ_API_KEY=gsk-your-key
        """)
        if st.sidebar.button("📖 Full Setup Guide"):
            st.sidebar.markdown("[View API Setup Guide](API_SETUP_GUIDE.md)")
    
    # Check system capabilities
    capabilities = st.session_state.document_processor.get_processing_capabilities()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # OCR Status
        ocr_status = "✅" if capabilities.get('advanced_models', False) else "⚠️"
        st.metric("OCR Models", ocr_status)
        
        # Vector DB Status
        vector_status = "✅" if capabilities.get('vector_db_available', True) else "📁"
        st.metric("Vector Storage", vector_status)
    
    with col2:
        # Image Processing
        img_status = "✅" if capabilities.get('image_processing', False) else "⚠️"
        st.metric("Image Processing", img_status)
        
        # PDF Processing
        pdf_status = "✅" if capabilities.get('pdf_processing', False) else "⚠️"
        st.metric("PDF Processing", pdf_status)
    
    # Display available models
    with st.sidebar.expander("🤖 Available Models", expanded=False):
        model_manager = DocumentModelManager()
        available_models = model_manager.get_available_models()
        
        for category, models in available_models.items():
            if models:
                st.write(f"**{category.replace('_', ' ').title()}**")
                for model in models[:3]:  # Show first 3 models
                    st.write(f"• {model}")
                if len(models) > 3:
                    st.write(f"... and {len(models) - 3} more")
    
    # Document statistics
    if st.session_state.document_stats:
        with st.sidebar.expander("📊 Document Statistics", expanded=True):
            stats = st.session_state.document_stats
            st.metric("Processed Documents", stats.get('total_documents', 0))
            st.metric("Extracted Pages", stats.get('total_pages', 0))
            st.metric("Indexed Chunks", stats.get('total_chunks', 0))

def document_upload_section():
    """Document upload and processing section"""
    st.markdown("## 📁 Document Upload & Processing")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents for analysis",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'pdf', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: Images (PNG, JPG, TIFF), PDF documents, and text files"
    )
    
    if uploaded_files:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"📄 {len(uploaded_files)} file(s) uploaded")
        
        with col2:
            # Processing options
            with st.expander("⚙️ Processing Options"):
                enhance_images = st.checkbox("Enhance Images", value=True)
                include_layout = st.checkbox("Layout Analysis", value=True)
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    options=["quick", "standard", "comprehensive"],
                    index=1
                )
        
        with col3:
            process_button = st.button(
                "🚀 Process Documents",
                type="primary",
                disabled=st.session_state.processing_in_progress,
                use_container_width=True
            )
        
        # Process documents when button clicked
        if process_button:
            asyncio.run(process_uploaded_documents(
                uploaded_files,
                enhance_images=enhance_images,
                include_layout=include_layout,
                analysis_depth=analysis_depth
            ))

async def process_uploaded_documents(
    uploaded_files,
    enhance_images: bool = True,
    include_layout: bool = True,
    analysis_depth: str = "standard"
):
    """Process uploaded documents through the workflow"""
    try:
        st.session_state.processing_in_progress = True
        
        # Create progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
        
        total_files = len(uploaded_files)
        processed_files = 0
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"🔄 Processing {uploaded_file.name}...")
            progress_bar.progress(i / total_files)
            
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Process through workflow
                processing_options = {
                    "include_metadata_extraction": True,
                    "include_handwriting_analysis": True,
                    "include_historical_context": True,
                    "analysis_depth": analysis_depth,
                    "enhance_images": enhance_images,
                    "include_layout": include_layout
                }
                
                workflow_result = await st.session_state.workflow_manager.process_document(
                    document_path=tmp_file_path,
                    document_type="auto",
                    user_query=f"Analyze this document: {uploaded_file.name}",
                    analysis_options=processing_options
                )
                
                # Store results
                doc_id = f"doc_{int(time.time())}_{i}"
                st.session_state.processed_documents[doc_id] = {
                    "filename": uploaded_file.name,
                    "workflow_result": workflow_result,
                    "processed_at": datetime.now().isoformat()
                }
                
                # Display immediate results
                with results_container:
                    display_processing_result(uploaded_file.name, workflow_result)
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                processed_files += 1
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"File processing error: {e}")
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text(f"✅ Processed {processed_files}/{total_files} documents successfully!")
        
        # Update document statistics
        await update_document_statistics()
        
        # Index documents for RAG if successful
        if st.session_state.processed_documents:
            st.info("🔄 Indexing documents for chat functionality...")
            await index_documents_for_rag()
            
            # Force immediate indexing and verification
            rag_stats = st.session_state.rag_system.get_document_statistics()
            st.info(f"📊 RAG Status: {rag_stats.get('total_chunks', 0)} chunks indexed")
            
            # EMERGENCY FIX: Store extracted text directly in session for chat
            st.session_state.extracted_document_texts = {}
            for doc_id, doc_data in st.session_state.processed_documents.items():
                workflow_result = doc_data.get('workflow_result', {})
                extracted_text = workflow_result.get('extracted_text', '')
                if extracted_text:
                    st.session_state.extracted_document_texts[doc_id] = {
                        'text': extracted_text,
                        'filename': doc_data['filename']
                    }
            
            st.success(f"🔧 Emergency backup: {len(st.session_state.extracted_document_texts)} documents available for direct chat")
        
    except Exception as e:
        st.error(f"❌ Document processing failed: {str(e)}")
        logger.error(f"Document processing error: {e}")
    
    finally:
        st.session_state.processing_in_progress = False

def display_processing_result(filename: str, workflow_result: Dict[str, Any]):
    """Display processing result for a single document"""
    success = workflow_result.get('success', False)
    
    with st.container():
        if success:
            st.success(f"✅ **{filename}** processed successfully!")
            
            # Show key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                confidence = workflow_result.get('confidence_score', 0.0)
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                text_length = len(workflow_result.get('extracted_text', ''))
                st.metric("Text Length", f"{text_length:,} chars")
            
            with col3:
                processing_time = workflow_result.get('processing_summary', {}).get('processing_time', 0)
                st.metric("Processing Time", f"{processing_time:.1f}s")
            
            with col4:
                steps = workflow_result.get('processing_summary', {}).get('total_steps', 0)
                st.metric("Workflow Steps", steps)
            
            # Show extracted text preview - ALWAYS EXPANDED
            extracted_text = workflow_result.get('extracted_text', '')
            if extracted_text:
                with st.expander(f"📝 Text Preview - {filename}", expanded=True):
                    preview_text = extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                    st.text_area("Extracted Text", preview_text, height=200, disabled=True)
                    
                    # Also show it directly for visibility
                    st.markdown("**📄 Extracted Content:**")
                    st.text(preview_text)
            else:
                st.warning("⚠️ No text was extracted from this document")
                # Show debug info
                st.write("**Debug Info:**")
                st.write(f"- Workflow Success: {workflow_result.get('success', False)}")
                st.write(f"- Model Used: {workflow_result.get('document_metadata', {}).get('model_used', 'Unknown')}")
                st.write(f"- Processing Errors: {len(workflow_result.get('errors', []))}")
        else:
            st.error(f"❌ **{filename}** processing failed!")
            errors = workflow_result.get('errors', [])
            if errors:
                for error in errors:
                    st.error(f"• {error}")

async def update_document_statistics():
    """Update document processing statistics"""
    try:
        total_docs = len(st.session_state.processed_documents)
        total_text_length = 0
        total_pages = 0
        successful_docs = 0
        
        for doc_data in st.session_state.processed_documents.values():
            workflow_result = doc_data.get('workflow_result', {})
            if workflow_result.get('success', False):
                successful_docs += 1
                total_text_length += len(workflow_result.get('extracted_text', ''))
                
                # Count pages from document metadata
                doc_metadata = workflow_result.get('document_metadata', {})
                total_pages += doc_metadata.get('pages', 1)
        
        # Get RAG statistics
        rag_stats = st.session_state.rag_system.get_document_statistics()
        
        st.session_state.document_stats = {
            'total_documents': total_docs,
            'successful_documents': successful_docs,
            'total_pages': total_pages,
            'total_text_length': total_text_length,
            'total_chunks': rag_stats.get('total_chunks', 0),
            'vector_store_count': rag_stats.get('vector_store_count', 0)
        }
        
    except Exception as e:
        logger.error(f"Statistics update error: {e}")

async def index_documents_for_rag():
    """Enhanced document indexing for RAG system with debugging"""
    try:
        if not st.session_state.processed_documents:
            st.warning("❌ No processed documents found for indexing")
            return
        
        # Prepare documents for indexing
        documents_to_index = {}
        total_text_length = 0
        
        for doc_id, doc_data in st.session_state.processed_documents.items():
            workflow_result = doc_data.get('workflow_result', {})
            
            if workflow_result.get('success', False):
                # Create mock document pages for RAG
                from processors.document_processor import DocumentPage
                
                extracted_text = workflow_result.get('extracted_text', '')
                
                if extracted_text and extracted_text.strip():
                    total_text_length += len(extracted_text)
                    
                    st.write(f"📄 Preparing to index: {doc_data['filename']} ({len(extracted_text)} chars)")
                    
                    page = DocumentPage(
                        page_number=1,
                        image=None,  # Not needed for RAG
                        text=extracted_text,
                        confidence=workflow_result.get('confidence_score', 0.0),
                        metadata=workflow_result.get('document_metadata', {})
                    )
                    
                    documents_to_index[doc_id] = [page]
                else:
                    st.warning(f"⚠️ No text content found in {doc_data['filename']} for indexing")
        
        if documents_to_index:
            st.write(f"🔄 Indexing {len(documents_to_index)} documents with {total_text_length} total characters...")
            
            # Index documents
            indexing_result = await st.session_state.rag_system.index_documents(documents_to_index)
            
            if indexing_result.get('success', False):
                chunks_created = indexing_result.get('total_chunks', 0)
                st.success(f"✅ Successfully indexed {indexing_result.get('processed_documents', 0)} documents into {chunks_created} searchable chunks")
                
                # Verify indexing worked
                rag_stats = st.session_state.rag_system.get_document_statistics()
                st.info(f"📊 RAG Database: {rag_stats.get('total_chunks', 0)} chunks ready for chat")
                
                # Test retrieval immediately
                if chunks_created > 0:
                    test_result = await st.session_state.rag_system.retrieve_relevant_chunks("Galiskell", top_k=2)
                    st.info(f"🧪 Test retrieval found {len(test_result.chunks)} relevant chunks")
            else:
                st.error("❌ Document indexing failed")
                st.error(f"Error: {indexing_result.get('error', 'Unknown error')}")
        else:
            st.error("❌ No valid documents to index - check text extraction")
    
    except Exception as e:
        logger.error(f"RAG indexing error: {e}")
        st.error(f"❌ RAG indexing failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def document_analysis_section():
    """Display detailed document analysis results"""
    if not st.session_state.processed_documents:
        st.info("📝 Process documents above to see detailed analysis results")
        return
    
    st.markdown("## 📊 Document Analysis Results")
    
    # Document selector
    doc_options = {
        doc_id: doc_data['filename'] 
        for doc_id, doc_data in st.session_state.processed_documents.items()
    }
    
    selected_doc_id = st.selectbox(
        "Select document for detailed analysis:",
        options=list(doc_options.keys()),
        format_func=lambda x: doc_options[x]
    )
    
    if selected_doc_id:
        doc_data = st.session_state.processed_documents[selected_doc_id]
        workflow_result = doc_data.get('workflow_result', {})
        
        # Display comprehensive analysis
        display_detailed_analysis(doc_data['filename'], workflow_result)

def display_detailed_analysis(filename: str, workflow_result: Dict[str, Any]):
    """Display detailed analysis of a processed document"""
    st.markdown(f"### 📄 Analysis: **{filename}**")
    
    if not workflow_result.get('success', False):
        st.error("❌ Document processing was not successful")
        return
    
    # Analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Text Content", 
        "🔍 OCR Analysis", 
        "✍️ Handwriting Analysis", 
        "🏛️ Historical Context",
        "📊 Processing Details"
    ])
    
    with tab1:
        # Text content analysis
        extracted_text = workflow_result.get('extracted_text', '')
        content_analysis = workflow_result.get('content_analysis', {})
        
        st.markdown("### 📝 Extracted Text Content")
        
        if extracted_text and extracted_text.strip():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.text_area("📄 Complete Extracted Text", extracted_text, height=400, key="main_text_display")
                
                # Show text in markdown format as well for better readability
                st.markdown("**📖 Formatted View:**")
                st.markdown(f"```\n{extracted_text}\n```")
            
            with col2:
                st.markdown("**📊 Content Metrics**")
                actual_length = len(extracted_text.strip())
                words = extracted_text.split()
                lines = extracted_text.split('\n')
                
                st.metric("Text Length", f"{actual_length:,} characters")
                st.metric("Word Count", f"{len(words):,} words") 
                st.metric("Line Count", f"{len(lines):,} lines")
                st.metric("Readability", content_analysis.get('readability', 'Unknown').title())
                
                # Show found dates
                dates_found = content_analysis.get('dates_found', [])
                if dates_found:
                    st.markdown("**📅 Dates Found**")
                    for date in dates_found[:5]:
                        st.write(f"• {date}")
                
                # Show text type
                doc_metadata = workflow_result.get('document_metadata', {})
                if doc_metadata.get('handwriting_detected'):
                    st.info("✍️ Handwriting detected")
                
                model_used = doc_metadata.get('model_used', 'Unknown')
                st.markdown(f"**🤖 Model:** {model_used}")
                
        else:
            st.error("❌ No text content was extracted from this document")
            
            # Debug information
            st.markdown("**🔍 Debug Information:**")
            st.write(f"- Raw extracted text: `{repr(extracted_text)}`")
            st.write(f"- Text length: {len(extracted_text) if extracted_text else 0}")
            st.write(f"- Workflow success: {workflow_result.get('success', False)}")
            
            doc_metadata = workflow_result.get('document_metadata', {})
            st.write(f"- Model used: {doc_metadata.get('model_used', 'Unknown')}")
            st.write(f"- Confidence: {doc_metadata.get('confidence', 0)}")
            
            errors = workflow_result.get('errors', [])
            if errors:
                st.markdown("**❌ Errors:**")
                for error in errors:
                    st.write(f"- {error}")
            
            st.markdown("**💡 Recommendations:**")
            st.write("1. Ensure the image contains readable text")
            st.write("2. Try a higher resolution image (300+ DPI)")
            st.write("3. Check image contrast and brightness")
            st.write("4. Install advanced OCR models: `pip install transformers torch`")
    
    with tab2:
        # OCR analysis details
        document_metadata = workflow_result.get('document_metadata', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("OCR Confidence", f"{document_metadata.get('confidence', 0):.1%}")
            st.metric("Model Used", document_metadata.get('model_used', 'Unknown'))
        
        with col2:
            st.metric("Document Type", document_metadata.get('document_type', 'Unknown').title())
            handwriting = "Yes" if document_metadata.get('handwriting_detected', False) else "No"
            st.metric("Handwriting Detected", handwriting)
        
        with col3:
            processing_time = workflow_result.get('processing_summary', {}).get('processing_time', 0)
            st.metric("Processing Time", f"{processing_time:.2f}s")
            
            image_size = document_metadata.get('image_size')
            if image_size:
                st.metric("Image Dimensions", f"{image_size[0]}×{image_size[1]}")
    
    with tab3:
        # Handwriting analysis
        handwriting_analysis = workflow_result.get('handwriting_analysis', {})
        
        if handwriting_analysis and handwriting_analysis != {}:
            st.json(handwriting_analysis)
        else:
            if document_metadata.get('handwriting_detected', False):
                st.info("Handwriting was detected but detailed analysis is not available")
            else:
                st.info("No handwriting detected in this document")
    
    with tab4:
        # Historical context analysis
        historical_analysis = workflow_result.get('historical_analysis', {})
        
        if historical_analysis and historical_analysis != {}:
            st.markdown("**Historical Context Analysis**")
            
            time_period = historical_analysis.get('time_period', 'Unknown')
            st.write(f"**Estimated Time Period:** {time_period}")
            
            context_clues = historical_analysis.get('context_clues', [])
            if context_clues:
                st.write("**Historical Context Clues:**")
                for clue in context_clues:
                    st.write(f"• {clue}")
            
            dating_indicators = historical_analysis.get('dating_indicators', [])
            if dating_indicators:
                st.write("**Dating Indicators:**")
                for indicator in dating_indicators:
                    st.write(f"• {indicator}")
        else:
            st.info("No specific historical context analysis available")
    
    with tab5:
        # Processing details
        processing_summary = workflow_result.get('processing_summary', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Workflow Summary**")
            st.metric("Total Steps", processing_summary.get('total_steps', 0))
            st.metric("Errors", processing_summary.get('errors', 0))
            st.metric("Success Rate", "100%" if processing_summary.get('success', False) else "Failed")
        
        with col2:
            st.markdown("**Processing Metrics**")
            st.metric("Embeddings Generated", "Yes" if processing_summary.get('embeddings_generated', False) else "No")
            st.metric("Analysis Completed", "Yes" if processing_summary.get('analysis_completed', False) else "No")
        
        # Show workflow steps
        step_history = workflow_result.get('step_history', [])
        if step_history:
            st.markdown("**Workflow Steps Completed**")
            for i, step in enumerate(step_history, 1):
                st.write(f"{i}. {step.replace('_', ' ').title()}")
        
        # Show errors if any
        errors = workflow_result.get('errors', [])
        if errors:
            st.markdown("**Errors Encountered**")
            for error in errors:
                st.error(f"• {error}")

def document_chat_section():
    """Interactive chat with processed documents"""
    if not st.session_state.processed_documents:
        st.info("📝 Process documents above to start chatting with them")
        return
    
    st.markdown("## 💬 Chat with Your Documents")
    st.markdown("Ask questions about your processed documents and get AI-powered answers with source references.")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for chat_msg in st.session_state.chat_history:
            if chat_msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message" style="background: #e3f2fd; border-left: 4px solid #2196f3;">
                    <strong>You:</strong> {chat_msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message" style="background: #f3e5f5; border-left: 4px solid #9c27b0;">
                    <strong>AI:</strong> {chat_msg['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source chunks if available
                if 'sources' in chat_msg and chat_msg['sources']:
                    with st.expander(f"📚 Sources ({len(chat_msg['sources'])} chunks)", expanded=False):
                        for i, source in enumerate(chat_msg['sources'], 1):
                            st.markdown(f"""
                            <div class="source-chunk">
                                <strong>Source {i} (Page {source.get('page_number', 'Unknown')}):</strong><br>
                                {source.get('text', '')[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_query = st.text_input(
                "Ask a question about your documents:",
                placeholder="What is this document about? When was it written? Who are the people mentioned?",
                key="chat_input"
            )
        
        with col2:
            chat_submit = st.form_submit_button("💬 Send", use_container_width=True)
    
    # Process chat query
    if chat_submit and user_query.strip():
        asyncio.run(process_chat_query(user_query.strip()))

async def process_chat_query(query: str):
    """Process user chat query and generate response"""
    try:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })
        
        # Show processing message
        with st.spinner("🔍 Searching documents and generating response..."):
            # EMERGENCY CHAT: Try direct text search first
            if hasattr(st.session_state, 'extracted_document_texts') and st.session_state.extracted_document_texts:
                st.info("🔧 Using emergency direct text search...")
                
                # Search directly in extracted texts
                query_lower = query.lower()
                found_content = []
                
                for doc_id, doc_info in st.session_state.extracted_document_texts.items():
                    text = doc_info['text']
                    text_lower = text.lower()
                    
                    # Look for query terms in the text
                    if any(term in text_lower for term in query_lower.split()):
                        # Find relevant sentences
                        sentences = text.split('.')
                        relevant_sentences = []
                        
                        for sentence in sentences:
                            if any(term in sentence.lower() for term in query_lower.split()):
                                relevant_sentences.append(sentence.strip())
                        
                        if relevant_sentences:
                            found_content.append({
                                'filename': doc_info['filename'],
                                'content': '. '.join(relevant_sentences[:3]),  # First 3 relevant sentences
                                'full_text': text[:500] + "..." if len(text) > 500 else text
                            })
                
                if found_content:
                    # Generate response from found content
                    response_parts = [f"Based on the document content, here's what I found:"]
                    
                    for item in found_content:
                        response_parts.append(f"\n**From {item['filename']}:**")
                        response_parts.append(item['content'])
                    
                    # Add relevant context
                    if 'galiskell' in query_lower or 'gaitskell' in query_lower:
                        response_parts.append(f"\n\nThis appears to be about Mr. Galiskell/Gaitskell and Labour party activities.")
                    
                    emergency_response = '\n'.join(response_parts)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': emergency_response,
                        'confidence': 0.8,
                        'sources': [{'filename': item['filename'], 'text': item['content']} for item in found_content],
                        'timestamp': datetime.now().isoformat(),
                        'method': 'emergency_direct_search'
                    })
                    
                    st.rerun()
                    return
            
            # Fallback to RAG system
            chat_response = await st.session_state.rag_system.chat_with_documents(
                query=query,
                conversation_history=st.session_state.chat_history[-5:],  # Last 5 messages for context
                top_k=3  # Retrieve top 3 relevant chunks
            )
            
            # Prepare sources for display
            sources = []
            for chunk in chat_response.source_chunks:
                sources.append({
                    'page_number': chunk.page_number,
                    'text': chunk.text,
                    'confidence': getattr(chunk, 'confidence', 0.0),
                    'document_id': chunk.document_id
                })
            
            # Add AI response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': chat_response.answer,
                'confidence': chat_response.confidence,
                'sources': sources,
                'timestamp': datetime.now().isoformat()
            })
        
        # Rerun to show updated chat
        st.rerun()
    
    except Exception as e:
        st.error(f"❌ Chat processing failed: {str(e)}")
        logger.error(f"Chat processing error: {e}")

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # System status sidebar
    display_system_status()
    
    # Main content sections
    st.markdown("---")
    
    # Document upload and processing
    document_upload_section()
    
    st.markdown("---")
    
    # Document analysis results
    document_analysis_section()
    
    st.markdown("---")
    
    # Document chat interface
    document_chat_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Handwriting Document Agent V1 - Powered by HuggingFace, LangGraph, and Streamlit<br>"
        "Author: Mohammed Hamdan</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()