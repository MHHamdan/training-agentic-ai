"""
LangGraph Workflow for Document Processing and Analysis
Orchestrates OCR, analysis, and chat functionality with observability
Author: Mohammed Hamdan
"""

import os
import logging
from typing import Dict, Any, List, Optional, TypedDict
import asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
import json

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langgraph.checkpoint.sqlite import SqliteSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config, ANALYSIS_TEMPLATES, DOCUMENT_TYPES
from models.document_models import DocumentModelManager, DocumentProcessingResult

logger = logging.getLogger(__name__)

class DocumentState(TypedDict):
    """State for document processing workflow"""
    # Input
    document_path: str
    document_type: str
    user_query: str
    analysis_options: Dict[str, Any]
    
    # Processing state
    current_step: str
    step_history: List[str]
    errors: List[str]
    
    # Document processing results
    ocr_result: Optional[DocumentProcessingResult]
    extracted_text: str
    document_metadata: Dict[str, Any]
    
    # Analysis results
    content_analysis: Dict[str, Any]
    historical_analysis: Dict[str, Any]
    handwriting_analysis: Dict[str, Any]
    
    # Chat and RAG
    embeddings: Optional[List[float]]
    retrieval_context: List[str]
    chat_history: List[Dict[str, str]]
    
    # Final output
    final_response: str
    confidence_score: float
    processing_summary: Dict[str, Any]

@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    total_processing_time: float = 0.0
    step_timings: Dict[str, float] = None
    
    def __post_init__(self):
        if self.step_timings is None:
            self.step_timings = {}

class DocumentWorkflowManager:
    """
    LangGraph-based workflow manager for document processing
    Handles OCR, analysis, and interactive chat functionality
    """
    
    def __init__(self):
        """Initialize document workflow manager"""
        self.model_manager = DocumentModelManager()
        self.workflow = None
        self.checkpointer = None
        self.metrics = WorkflowMetrics()
        
        if LANGGRAPH_AVAILABLE:
            self._initialize_workflow()
            logger.info("✅ LangGraph document workflow initialized")
        else:
            logger.warning("⚠️ LangGraph not available - using sequential processing")
    
    def _initialize_workflow(self):
        """Initialize LangGraph workflow"""
        try:
            # Create workflow graph
            workflow = StateGraph(DocumentState)
            
            # Add workflow nodes
            workflow.add_node("initialize", self.initialize_processing)
            workflow.add_node("ocr_processing", self.process_document_ocr)
            workflow.add_node("content_analysis", self.analyze_content)
            workflow.add_node("specialized_analysis", self.specialized_analysis)
            workflow.add_node("generate_embeddings", self.generate_embeddings)
            workflow.add_node("prepare_response", self.prepare_response)
            workflow.add_node("finalize", self.finalize_processing)
            
            # Define workflow edges
            workflow.add_edge("initialize", "ocr_processing")
            workflow.add_edge("ocr_processing", "content_analysis")
            workflow.add_edge("content_analysis", "specialized_analysis")
            workflow.add_edge("specialized_analysis", "generate_embeddings")
            workflow.add_edge("generate_embeddings", "prepare_response")
            workflow.add_edge("prepare_response", "finalize")
            workflow.add_edge("finalize", END)
            
            # Set entry point
            workflow.set_entry_point("initialize")
            
            # Create checkpointer for state persistence
            self.checkpointer = SqliteSaver.from_conn_string(":memory:")
            
            # Compile workflow
            self.workflow = workflow.compile(checkpointer=self.checkpointer)
            
        except Exception as e:
            logger.error(f"Workflow initialization error: {e}")
            self.workflow = None
    
    @observe(as_type="workflow_step")
    async def initialize_processing(self, state: DocumentState) -> DocumentState:
        """Initialize document processing workflow"""
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Initializing document processing: {state['document_path']}",
                    metadata={
                        "document_type": state.get("document_type", "auto"),
                        "step": "initialize",
                        "organization": "document-analysis-org",
                        "project": "handwriting-document-agent-v1"
                    }
                )
            
            # Update state
            state["current_step"] = "initialize"
            state["step_history"] = ["initialize"]
            state["errors"] = []
            state["chat_history"] = []
            
            # Initialize analysis options if not provided
            if not state.get("analysis_options"):
                state["analysis_options"] = {
                    "include_metadata_extraction": config.include_metadata_extraction,
                    "include_handwriting_analysis": config.include_handwriting_analysis,
                    "include_historical_context": config.include_historical_context,
                    "analysis_depth": config.analysis_depth
                }
            
            logger.info(f"Document processing initialized for: {state['document_path']}")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output="Document processing initialized successfully"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Initialization error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
    
    @observe(as_type="workflow_step")
    async def process_document_ocr(self, state: DocumentState) -> DocumentState:
        """Process document with OCR and text extraction"""
        try:
            import time
            start_time = time.time()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Processing OCR for document type: {state.get('document_type', 'auto')}",
                    metadata={"step": "ocr_processing"}
                )
            
            state["current_step"] = "ocr_processing"
            state["step_history"].append("ocr_processing")
            
            # Load and process document image
            from PIL import Image
            try:
                image = Image.open(state["document_path"])
                logger.info(f"Loaded document image: {image.size}")
            except Exception as e:
                # Try to handle as text file
                with open(state["document_path"], 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Create mock OCR result for text files
                ocr_result = DocumentProcessingResult(
                    text=text,
                    confidence=1.0,
                    processing_time=0.1,
                    model_used="text_file_reader",
                    metadata={"source": "text_file"},
                    document_type=state.get("document_type", "text")
                )
                
                state["ocr_result"] = ocr_result
                state["extracted_text"] = text
                state["document_metadata"] = {"file_type": "text", "source": state["document_path"]}
                
                processing_time = time.time() - start_time
                self.metrics.step_timings["ocr_processing"] = processing_time
                
                if langfuse_context:
                    langfuse_context.update_current_observation(
                        output=f"Text file processed - Length: {len(text)}"
                    )
                
                return state
            
            # Process image with OCR
            document_type = state.get("document_type", "auto")
            ocr_result = await self.model_manager.process_document_image(
                image=image,
                document_type=document_type,
                include_layout=True
            )
            
            # Store results in state
            state["ocr_result"] = ocr_result
            state["extracted_text"] = ocr_result.text
            state["document_metadata"] = {
                "confidence": ocr_result.confidence,
                "model_used": ocr_result.model_used,
                "document_type": ocr_result.document_type,
                "handwriting_detected": ocr_result.handwriting_detected,
                "image_size": image.size,
                **ocr_result.metadata
            }
            
            processing_time = time.time() - start_time
            self.metrics.step_timings["ocr_processing"] = processing_time
            
            logger.info(f"OCR processing completed - Text length: {len(ocr_result.text)}")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"OCR completed - Text: {len(ocr_result.text)} chars, Confidence: {ocr_result.confidence}"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"OCR processing error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            
            # Provide fallback text
            state["extracted_text"] = f"Error processing document: {error_msg}"
            state["document_metadata"] = {"error": error_msg}
            
            return state
    
    @observe(as_type="workflow_step")
    async def analyze_content(self, state: DocumentState) -> DocumentState:
        """Analyze document content for general insights"""
        try:
            import time
            start_time = time.time()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Analyzing content - Text length: {len(state.get('extracted_text', ''))}",
                    metadata={"step": "content_analysis"}
                )
            
            state["current_step"] = "content_analysis"
            state["step_history"].append("content_analysis")
            
            extracted_text = state.get("extracted_text", "")
            document_type = state.get("document_type", "unknown")
            
            # Perform content analysis
            content_analysis = await self.model_manager.analyze_document_content(
                text=extracted_text,
                document_type=document_type
            )
            
            state["content_analysis"] = content_analysis
            
            processing_time = time.time() - start_time
            self.metrics.step_timings["content_analysis"] = processing_time
            
            logger.info("Content analysis completed")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Content analysis completed - {len(content_analysis)} metrics extracted"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Content analysis error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["content_analysis"] = {"error": error_msg}
            return state
    
    @observe(as_type="workflow_step")
    async def specialized_analysis(self, state: DocumentState) -> DocumentState:
        """Perform specialized analysis based on document type"""
        try:
            import time
            start_time = time.time()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Performing specialized document analysis",
                    metadata={"step": "specialized_analysis"}
                )
            
            state["current_step"] = "specialized_analysis"
            state["step_history"].append("specialized_analysis")
            
            extracted_text = state.get("extracted_text", "")
            document_metadata = state.get("document_metadata", {})
            analysis_options = state.get("analysis_options", {})
            
            # Initialize analysis results
            state["historical_analysis"] = {}
            state["handwriting_analysis"] = {}
            
            # Historical document analysis
            if analysis_options.get("include_historical_context", True):
                state["historical_analysis"] = await self._analyze_historical_context(
                    extracted_text, document_metadata
                )
            
            # Handwriting analysis
            if (analysis_options.get("include_handwriting_analysis", True) and 
                document_metadata.get("handwriting_detected", False)):
                state["handwriting_analysis"] = await self._analyze_handwriting(
                    extracted_text, document_metadata
                )
            
            processing_time = time.time() - start_time
            self.metrics.step_timings["specialized_analysis"] = processing_time
            
            logger.info("Specialized analysis completed")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output="Specialized analysis completed"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Specialized analysis error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
    
    @observe(as_type="workflow_step")
    async def generate_embeddings(self, state: DocumentState) -> DocumentState:
        """Generate embeddings for document text"""
        try:
            import time
            start_time = time.time()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Generating document embeddings",
                    metadata={"step": "generate_embeddings"}
                )
            
            state["current_step"] = "generate_embeddings"
            state["step_history"].append("generate_embeddings")
            
            extracted_text = state.get("extracted_text", "")
            
            # Generate embeddings for the document
            embeddings = await self.model_manager.embed_text(extracted_text)
            state["embeddings"] = embeddings
            
            # Initialize retrieval context
            state["retrieval_context"] = [extracted_text] if extracted_text else []
            
            processing_time = time.time() - start_time
            self.metrics.step_timings["generate_embeddings"] = processing_time
            
            logger.info("Embeddings generation completed")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Embeddings generated - Dimensions: {len(embeddings) if embeddings else 0}"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Embeddings generation error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["embeddings"] = None
            return state
    
    @observe(as_type="workflow_step")
    async def prepare_response(self, state: DocumentState) -> DocumentState:
        """Prepare final response based on analysis results"""
        try:
            import time
            start_time = time.time()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Preparing final response",
                    metadata={"step": "prepare_response"}
                )
            
            state["current_step"] = "prepare_response"
            state["step_history"].append("prepare_response")
            
            # Compile analysis results
            response_sections = []
            
            # Document Overview
            response_sections.append("## 📄 Document Analysis Summary")
            
            # OCR Results
            if state.get("ocr_result"):
                ocr_result = state["ocr_result"]
                response_sections.append(f"""
### 🔍 Text Extraction Results
- **Model Used**: {ocr_result.model_used}
- **Confidence**: {ocr_result.confidence:.2f}
- **Processing Time**: {ocr_result.processing_time:.2f}s
- **Document Type**: {ocr_result.document_type}
- **Handwriting Detected**: {'Yes' if ocr_result.handwriting_detected else 'No'}
                """)
            
            # Extracted Text
            extracted_text = state.get("extracted_text", "")
            if extracted_text:
                preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                response_sections.append(f"""
### 📝 Extracted Text
```
{preview_text}
```
                """)
            
            # Content Analysis
            if state.get("content_analysis"):
                analysis = state["content_analysis"]
                response_sections.append(f"""
### 📊 Content Analysis
- **Text Length**: {analysis.get('text_length', 0)} characters
- **Word Count**: {analysis.get('word_count', 0)} words
- **Readability**: {analysis.get('readability', 'Unknown').title()}
- **Language**: {analysis.get('language', 'Unknown')}
                """)
                
                if analysis.get("dates_found"):
                    response_sections.append(f"- **Dates Found**: {', '.join(analysis['dates_found'])}")
                
                if analysis.get("historical_indicators"):
                    response_sections.append(f"- **Historical Terms**: {', '.join(analysis['historical_indicators'])}")
            
            # Historical Analysis
            if state.get("historical_analysis") and state["historical_analysis"]:
                response_sections.append(f"""
### 🏛️ Historical Context Analysis
{json.dumps(state['historical_analysis'], indent=2)}
                """)
            
            # Handwriting Analysis
            if state.get("handwriting_analysis") and state["handwriting_analysis"]:
                response_sections.append(f"""
### ✍️ Handwriting Analysis
{json.dumps(state['handwriting_analysis'], indent=2)}
                """)
            
            # User Query Response
            user_query = state.get("user_query", "")
            if user_query:
                response_sections.append(f"""
### ❓ Response to Your Query: "{user_query}"
Based on the document analysis, here are the key insights related to your question:

{await self._generate_query_response(state, user_query)}
                """)
            
            # Combine all sections
            final_response = "\n".join(response_sections)
            state["final_response"] = final_response
            
            # Calculate confidence score
            confidence_scores = []
            if state.get("ocr_result"):
                confidence_scores.append(state["ocr_result"].confidence)
            if state.get("content_analysis", {}).get("metadata_extracted"):
                confidence_scores.append(0.8)
            
            state["confidence_score"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            processing_time = time.time() - start_time
            self.metrics.step_timings["prepare_response"] = processing_time
            
            logger.info("Response preparation completed")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Response prepared - Length: {len(final_response)}"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Response preparation error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            state["final_response"] = f"Error preparing response: {error_msg}"
            state["confidence_score"] = 0.0
            return state
    
    @observe(as_type="workflow_step")
    async def finalize_processing(self, state: DocumentState) -> DocumentState:
        """Finalize document processing workflow"""
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input="Finalizing document processing",
                    metadata={"step": "finalize"}
                )
            
            state["current_step"] = "finalize"
            state["step_history"].append("finalize")
            
            # Create processing summary
            summary = {
                "total_steps": len(state["step_history"]),
                "errors": len(state["errors"]),
                "success": len(state["errors"]) == 0,
                "processing_time": sum(self.metrics.step_timings.values()),
                "confidence_score": state.get("confidence_score", 0.0),
                "document_processed": bool(state.get("extracted_text")),
                "embeddings_generated": bool(state.get("embeddings")),
                "analysis_completed": bool(state.get("content_analysis"))
            }
            
            state["processing_summary"] = summary
            
            # Update metrics
            self.metrics.total_steps = len(state["step_history"])
            self.metrics.successful_steps = len(state["step_history"]) - len(state["errors"])
            self.metrics.failed_steps = len(state["errors"])
            self.metrics.total_processing_time = summary["processing_time"]
            
            logger.info(f"Document processing finalized - Success: {summary['success']}")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Processing finalized - Success: {summary['success']}, Steps: {summary['total_steps']}"
                )
            
            return state
        
        except Exception as e:
            error_msg = f"Finalization error: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
            return state
    
    async def _analyze_historical_context(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical context of the document"""
        try:
            # Simple pattern-based historical analysis
            historical_analysis = {
                "time_period": "unknown",
                "historical_significance": "to_be_determined",
                "context_clues": [],
                "dating_indicators": [],
                "analysis_method": "pattern_based"
            }
            
            text_lower = text.lower()
            
            # Look for time period indicators
            century_patterns = {
                "18th": ["1700", "1701", "1702", "1750", "1799", "eighteenth"],
                "19th": ["1800", "1850", "1899", "nineteenth", "victorian"],
                "20th": ["1900", "1950", "1999", "twentieth", "modern"],
                "21st": ["2000", "2010", "2020", "twenty-first", "contemporary"]
            }
            
            for century, indicators in century_patterns.items():
                if any(indicator in text_lower for indicator in indicators):
                    historical_analysis["time_period"] = f"{century} century"
                    break
            
            # Look for historical context clues
            context_terms = [
                "colonial", "revolution", "independence", "civil war", "world war",
                "depression", "industrial", "railroad", "telegraph", "telephone",
                "automobile", "aviation", "computer", "internet"
            ]
            
            historical_analysis["context_clues"] = [
                term for term in context_terms if term in text_lower
            ]
            
            return historical_analysis
        
        except Exception as e:
            logger.error(f"Historical analysis error: {e}")
            return {"error": str(e)}
    
    async def _analyze_handwriting(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze handwriting characteristics"""
        try:
            # Simple handwriting analysis based on OCR confidence and patterns
            handwriting_analysis = {
                "legibility": "unknown",
                "style_period": "modern",
                "characteristics": [],
                "confidence_assessment": metadata.get("confidence", 0.0),
                "analysis_method": "ocr_based"
            }
            
            # Assess legibility based on OCR confidence
            confidence = metadata.get("confidence", 0.0)
            if confidence > 0.8:
                handwriting_analysis["legibility"] = "high"
            elif confidence > 0.6:
                handwriting_analysis["legibility"] = "medium"
            else:
                handwriting_analysis["legibility"] = "low"
            
            # Analyze text patterns for style indicators
            if any(char in text for char in "æœþð"):
                handwriting_analysis["characteristics"].append("archaic_letters")
            
            if text.count("&") > text.count(" and "):
                handwriting_analysis["characteristics"].append("ampersand_usage")
            
            return handwriting_analysis
        
        except Exception as e:
            logger.error(f"Handwriting analysis error: {e}")
            return {"error": str(e)}
    
    async def _generate_query_response(self, state: DocumentState, query: str) -> str:
        """Generate response to user query based on document analysis"""
        try:
            # Simple query response based on available analysis
            extracted_text = state.get("extracted_text", "")
            content_analysis = state.get("content_analysis", {})
            
            query_lower = query.lower()
            
            # Basic query matching
            if any(term in query_lower for term in ["what", "content", "text"]):
                return f"The document contains {content_analysis.get('word_count', 0)} words. {extracted_text[:200]}..."
            
            elif any(term in query_lower for term in ["when", "date", "time"]):
                dates = content_analysis.get("dates_found", [])
                if dates:
                    return f"Based on the document analysis, the following dates were identified: {', '.join(dates)}"
                else:
                    return "No specific dates could be identified in the document."
            
            elif any(term in query_lower for term in ["who", "author", "writer"]):
                return "The document analysis did not identify specific author information. This would require more advanced analysis."
            
            elif any(term in query_lower for term in ["handwriting", "written"]):
                if state.get("document_metadata", {}).get("handwriting_detected"):
                    handwriting = state.get("handwriting_analysis", {})
                    return f"Handwriting was detected in this document. Legibility assessment: {handwriting.get('legibility', 'unknown')}"
                else:
                    return "No handwriting was detected in this document - it appears to be printed text."
            
            else:
                return "Based on the document analysis, I can provide information about the text content, dates found, and document structure. Please ask more specific questions about these aspects."
        
        except Exception as e:
            logger.error(f"Query response error: {e}")
            return f"Error generating response to query: {str(e)}"
    
    @observe(as_type="document_workflow")
    async def process_document(
        self,
        document_path: str,
        document_type: str = "auto",
        user_query: str = "",
        analysis_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process document through complete workflow
        
        Args:
            document_path: Path to document file
            document_type: Type of document (auto, handwritten, printed, mixed)
            user_query: User question about the document
            analysis_options: Options for analysis depth and features
        
        Returns:
            Complete workflow results
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Starting document workflow: {document_path}",
                    metadata={
                        "document_path": document_path,
                        "document_type": document_type,
                        "user_query": user_query,
                        "organization": "document-analysis-org",
                        "project": "handwriting-document-agent-v1"
                    }
                )
            
            # Initialize state
            initial_state = DocumentState(
                document_path=document_path,
                document_type=document_type,
                user_query=user_query,
                analysis_options=analysis_options or {},
                current_step="",
                step_history=[],
                errors=[],
                ocr_result=None,
                extracted_text="",
                document_metadata={},
                content_analysis={},
                historical_analysis={},
                handwriting_analysis={},
                embeddings=None,
                retrieval_context=[],
                chat_history=[],
                final_response="",
                confidence_score=0.0,
                processing_summary={}
            )
            
            if self.workflow and LANGGRAPH_AVAILABLE:
                # Use LangGraph workflow
                logger.info("Processing document with LangGraph workflow")
                
                config_dict = {"configurable": {"thread_id": f"doc_{datetime.now().timestamp()}"}}
                
                # Run workflow
                result = await self.workflow.ainvoke(initial_state, config_dict)
                
            else:
                # Use sequential processing
                logger.info("Processing document with sequential workflow")
                result = await self._sequential_workflow(initial_state)
            
            # Extract final results
            workflow_results = {
                "success": len(result.get("errors", [])) == 0,
                "extracted_text": result.get("extracted_text", ""),
                "final_response": result.get("final_response", ""),
                "confidence_score": result.get("confidence_score", 0.0),
                "processing_summary": result.get("processing_summary", {}),
                "document_metadata": result.get("document_metadata", {}),
                "content_analysis": result.get("content_analysis", {}),
                "historical_analysis": result.get("historical_analysis", {}),
                "handwriting_analysis": result.get("handwriting_analysis", {}),
                "errors": result.get("errors", []),
                "step_history": result.get("step_history", []),
                "metrics": asdict(self.metrics)
            }
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Document workflow completed - Success: {workflow_results['success']}"
                )
            
            return workflow_results
        
        except Exception as e:
            logger.error(f"Document workflow error: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "final_response": f"Document processing failed: {str(e)}",
                "confidence_score": 0.0,
                "metrics": asdict(self.metrics)
            }
    
    async def _sequential_workflow(self, state: DocumentState) -> DocumentState:
        """Sequential workflow processing when LangGraph not available"""
        try:
            # Execute workflow steps sequentially
            state = await self.initialize_processing(state)
            state = await self.process_document_ocr(state)
            state = await self.analyze_content(state)
            state = await self.specialized_analysis(state)
            state = await self.generate_embeddings(state)
            state = await self.prepare_response(state)
            state = await self.finalize_processing(state)
            
            return state
        
        except Exception as e:
            logger.error(f"Sequential workflow error: {e}")
            state["errors"].append(str(e))
            return state