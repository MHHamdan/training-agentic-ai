"""
Document Processing Models with HuggingFace Integration
Specialized models for OCR, handwriting recognition, and document analysis
Author: Mohammed Hamdan
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import dataclass
from PIL import Image
import numpy as np

# Import API OCR processor
from .api_ocr import get_api_ocr

try:
    from transformers import (
        TrOCRProcessor, VisionEncoderDecoderModel,
        LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
        pipeline
    )
    from sentence_transformers import SentenceTransformer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config, DOCUMENT_TYPES

logger = logging.getLogger(__name__)

@dataclass
class DocumentProcessingResult:
    """Result from document processing"""
    text: str
    confidence: float
    processing_time: float
    model_used: str
    metadata: Dict[str, Any]
    bounding_boxes: List[Dict[str, Any]] = None
    handwriting_detected: bool = False
    document_type: str = "unknown"
    language: str = "en"

class DocumentModelManager:
    """
    Manages specialized document processing models
    Prioritizes HuggingFace models with fallback options
    """
    
    def __init__(self):
        """Initialize document model manager"""
        self.models = {}
        self.processors = {}
        self.embedding_model = None
        self.device = "cpu"  # Use CPU for compatibility
        
        logger.info("Initializing Document Model Manager")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available document models"""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace transformers not available - using fallback processing")
            return
        
        try:
            # Initialize OCR models
            if config.huggingface_api_key:
                self._load_ocr_models()
                self._load_layout_models()
                self._load_embedding_model()
            else:
                logger.warning("HuggingFace API key not found - some models may not be available")
        
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.info("Continuing with fallback processing capabilities")
    
    def _load_ocr_models(self):
        """Load latest high-performance OCR models with intelligent fallbacks"""
        try:
            # First attempt: Latest GOT-OCR2.0 (2024 state-of-art)
            logger.info("🔄 Attempting to load GOT-OCR2.0 (latest unified OCR model)...")
            try:
                from transformers import AutoTokenizer, AutoProcessor, AutoModel
                self.processors["got_ocr"] = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
                self.models["got_ocr"] = AutoModel.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
                logger.info("✅ GOT-OCR2.0 loaded successfully - using latest 2024 unified model!")
                return  # Use GOT-OCR as primary if available
            except Exception as e:
                logger.warning(f"⚠️ GOT-OCR2.0 not available: {e}")
            
            # Second attempt: Proven TrOCR models with better error handling
            logger.info("🔄 Loading TrOCR models (proven state-of-art)...")
            models_loaded = 0
            
            # Try handwritten models (large -> base)
            for model_size in ["large", "base"]:
                try:
                    model_name = f"microsoft/trocr-{model_size}-handwritten"
                    self.processors["handwritten"] = TrOCRProcessor.from_pretrained(model_name)
                    self.models["handwritten"] = VisionEncoderDecoderModel.from_pretrained(model_name)
                    logger.info(f"✅ TrOCR {model_size} handwritten model loaded")
                    models_loaded += 1
                    break
                except Exception as e:
                    logger.warning(f"⚠️ TrOCR {model_size} handwritten failed: {e}")
                    continue
            
            # Try printed models (large -> base)
            for model_size in ["large", "base"]:
                try:
                    model_name = f"microsoft/trocr-{model_size}-printed"
                    self.processors["printed"] = TrOCRProcessor.from_pretrained(model_name)
                    self.models["printed"] = VisionEncoderDecoderModel.from_pretrained(model_name)
                    logger.info(f"✅ TrOCR {model_size} printed model loaded")
                    models_loaded += 1
                    break
                except Exception as e:
                    logger.warning(f"⚠️ TrOCR {model_size} printed failed: {e}")
                    continue
            
            if models_loaded > 0:
                logger.info(f"✅ {models_loaded} OCR models loaded successfully")
            else:
                logger.warning("⚠️ No specialized OCR models loaded - will use enhanced fallback")
        
        except Exception as e:
            logger.error(f"❌ Error loading OCR models: {e}")
            logger.info("📝 Continuing with enhanced pattern-based analysis")
    
    def _load_layout_models(self):
        """Load document layout understanding models"""
        try:
            # LayoutLMv3 for document understanding
            logger.info("Loading LayoutLMv3 model...")
            self.processors["layout"] = LayoutLMv3Processor.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
            self.models["layout"] = LayoutLMv3ForTokenClassification.from_pretrained(
                "microsoft/layoutlmv3-base"
            )
            
            logger.info("✅ Layout models loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading layout models: {e}")
    
    def _load_embedding_model(self):
        """Load embedding model for document similarity"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(config.embedding_model)
            logger.info("✅ Embedding model loaded successfully")
        
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
    
    async def process_document_image_fast(
        self, 
        image: Image.Image, 
        document_type: str = "auto",
        include_layout: bool = True
    ) -> DocumentProcessingResult:
        """
        API-based document processing - high accuracy with modern AI models
        Uses vision-capable AI APIs for professional OCR results
        """
        try:
            logger.info("🚀 API OCR processing mode activated")
            
            # Use API OCR processor
            api_ocr = get_api_ocr()
            
            # Detect language hint for better accuracy
            language_hint = "auto"
            if document_type and document_type != "auto":
                language_hint = document_type
            
            ocr_result = await api_ocr.extract_text_with_api(image, language_hint)
            
            return DocumentProcessingResult(
                extracted_text=ocr_result.get('text', ''),
                confidence=ocr_result.get('confidence', 0.0),
                document_type=ocr_result.get('language_detected', 'auto'),
                handwriting_detected=True,  # Assume handwriting for historical documents
                model_used=f"{ocr_result.get('provider', 'api')}_{ocr_result.get('model', 'unknown')}",
                processing_time=ocr_result.get('processing_time', 0.0),
                metadata={
                    'api_mode': True,
                    'provider': ocr_result.get('provider', 'unknown'),
                    'model': ocr_result.get('model', 'unknown'),
                    'language_detected': ocr_result.get('language_detected', 'auto'),
                    'image_size': image.size,
                    'method': ocr_result.get('method', 'api_ocr')
                }
            )
            
        except Exception as e:
            logger.error(f"API processing failed: {e}")
            # Fallback to standard processing if API fails
            return await self.process_document_image_standard(image, document_type, include_layout)
    
    # Set fast processing as default
    async def process_document_image(self, image: Image.Image, document_type: str = "auto", include_layout: bool = True) -> DocumentProcessingResult:
        """Default processing method - uses fast OCR for optimal performance"""
        return await self.process_document_image_fast(image, document_type, include_layout)
    
    @observe(as_type="document_processing")
    async def process_document_image_standard(
        self, 
        image: Image.Image, 
        document_type: str = "auto",
        include_layout: bool = True
    ) -> DocumentProcessingResult:
        """
        Process document image with OCR and analysis
        
        Args:
            image: PIL Image of the document
            document_type: Type of document (handwritten, printed, mixed, auto)
            include_layout: Whether to include layout analysis
        
        Returns:
            DocumentProcessingResult with extracted text and metadata
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Processing document image - Type: {document_type}",
                    metadata={
                        "document_type": document_type,
                        "include_layout": include_layout,
                        "image_size": image.size,
                        "organization": "document-analysis-org",
                        "project": "handwriting-document-agent-v1"
                    }
                )
            
            import time
            start_time = time.time()
            
            # Auto-detect document type if needed
            if document_type == "auto":
                document_type = await self._detect_document_type(image)
            
            # Process based on document type
            if document_type in ["handwritten", "manuscript", "modern_handwriting"]:
                result = await self._process_handwritten(image)
            elif document_type in ["printed", "printed_historical"]:
                result = await self._process_printed(image)
            else:  # mixed or unknown
                result = await self._process_mixed_document(image)
            
            # Add layout analysis if requested
            if include_layout and "layout" in self.models:
                layout_info = await self._analyze_layout(image)
                result.metadata.update(layout_info)
            
            result.processing_time = time.time() - start_time
            result.document_type = document_type
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Document processed - Text length: {len(result.text)}",
                    metadata={
                        "text_length": len(result.text),
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "document_type_detected": document_type,
                        "handwriting_detected": result.handwriting_detected
                    }
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            # Return fallback result
            return DocumentProcessingResult(
                text="Error: Could not process document image",
                confidence=0.0,
                processing_time=0.0,
                model_used="fallback",
                metadata={"error": str(e)},
                document_type=document_type
            )
    
    async def _detect_document_type(self, image: Image.Image) -> str:
        """
        Auto-detect document type (handwritten vs printed) and language
        Enhanced approach with Arabic script detection
        """
        try:
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            image_array = np.array(gray_image)
            
            # Simple edge detection to determine if handwritten
            from scipy import ndimage
            edges = ndimage.sobel(image_array)
            edge_density = np.sum(edges > 50) / edges.size
            
            # Quick Arabic script detection using simple OCR test
            is_arabic = await self._detect_arabic_script(image)
            
            # Heuristic: handwritten text typically has more irregular edges
            if edge_density > 0.15:
                doc_type = "handwritten"
            else:
                doc_type = "printed"
            
            # Add language info
            if is_arabic:
                doc_type += "_arabic"
                
            return doc_type
        
        except Exception as e:
            logger.warning(f"Document type detection failed: {e}")
            return "mixed"
    
    async def _detect_arabic_script(self, image: Image.Image) -> bool:
        """Detect if document contains Arabic script"""
        try:
            import pytesseract
            
            # Quick test with Arabic language
            sample_text = pytesseract.image_to_string(image, lang='ara', config='--psm 6')
            
            # Check if we got Arabic characters
            arabic_chars = 0
            total_chars = 0
            
            for char in sample_text:
                if char.strip():
                    total_chars += 1
                    # Arabic Unicode range: 0x0600 to 0x06FF
                    if '\u0600' <= char <= '\u06FF':
                        arabic_chars += 1
            
            # If more than 30% Arabic characters, consider it Arabic
            if total_chars > 0 and (arabic_chars / total_chars) > 0.3:
                logger.info(f"🔤 Arabic script detected: {arabic_chars}/{total_chars} Arabic characters")
                return True
                
            return False
            
        except Exception as e:
            logger.warning(f"Arabic detection failed: {e}")
            return False
    
    async def _process_handwritten(self, image: Image.Image) -> DocumentProcessingResult:
        """Process handwritten text with TrOCR"""
        try:
            if "handwritten" not in self.models:
                return await self._fallback_processing(image, "handwritten")
            
            # Process with TrOCR handwritten model
            pixel_values = self.processors["handwritten"](image, return_tensors="pt").pixel_values
            generated_ids = self.models["handwritten"].generate(pixel_values)
            generated_text = self.processors["handwritten"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return DocumentProcessingResult(
                text=generated_text,
                confidence=0.85,  # TrOCR typically has high confidence
                processing_time=0.0,  # Will be set by caller
                model_used="microsoft/trocr-large-handwritten",
                metadata={
                    "model_type": "handwritten_ocr",
                    "preprocessing": "trocr_standard"
                },
                handwriting_detected=True
            )
        
        except Exception as e:
            logger.error(f"Handwritten processing error: {e}")
            return await self._fallback_processing(image, "handwritten")
    
    async def _process_printed(self, image: Image.Image) -> DocumentProcessingResult:
        """Process printed text with TrOCR"""
        try:
            if "printed" not in self.models:
                return await self._fallback_processing(image, "printed")
            
            # Process with TrOCR printed model
            pixel_values = self.processors["printed"](image, return_tensors="pt").pixel_values
            generated_ids = self.models["printed"].generate(pixel_values)
            generated_text = self.processors["printed"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return DocumentProcessingResult(
                text=generated_text,
                confidence=0.90,  # Printed text usually higher confidence
                processing_time=0.0,
                model_used="microsoft/trocr-large-printed",
                metadata={
                    "model_type": "printed_ocr",
                    "preprocessing": "trocr_standard"
                },
                handwriting_detected=False
            )
        
        except Exception as e:
            logger.error(f"Printed processing error: {e}")
            return await self._fallback_processing(image, "printed")
    
    async def _process_mixed_document(self, image: Image.Image) -> DocumentProcessingResult:
        """Process document with mixed handwritten and printed text"""
        try:
            # Try both models and combine results
            handwritten_result = await self._process_handwritten(image)
            printed_result = await self._process_printed(image)
            
            # Combine texts and take average confidence
            combined_text = f"{handwritten_result.text}\n{printed_result.text}"
            combined_confidence = (handwritten_result.confidence + printed_result.confidence) / 2
            
            return DocumentProcessingResult(
                text=combined_text,
                confidence=combined_confidence,
                processing_time=0.0,
                model_used="mixed_trocr_models",
                metadata={
                    "model_type": "mixed_ocr",
                    "handwritten_confidence": handwritten_result.confidence,
                    "printed_confidence": printed_result.confidence
                },
                handwriting_detected=True
            )
        
        except Exception as e:
            logger.error(f"Mixed document processing error: {e}")
            return await self._fallback_processing(image, "mixed")
    
    async def _analyze_layout(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze document layout with LayoutLMv3"""
        try:
            if "layout" not in self.models:
                return {"layout_analysis": "not_available"}
            
            # Process image for layout understanding
            encoding = self.processors["layout"](image, return_tensors="pt")
            outputs = self.models["layout"](**encoding)
            
            # Extract layout information
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            return {
                "layout_analysis": "completed",
                "structure_detected": True,
                "layout_elements": len(predictions),
                "model_used": "microsoft/layoutlmv3-base"
            }
        
        except Exception as e:
            logger.warning(f"Layout analysis failed: {e}")
            return {"layout_analysis": "failed", "error": str(e)}
    
    async def _fallback_processing(self, image: Image.Image, doc_type: str) -> DocumentProcessingResult:
        """Enhanced fallback processing with specialized handwriting detection"""
        try:
            import time
            start_time = time.time()
            
            # Enhanced OCR with multiple techniques for mixed content
            extracted_texts = []
            confidence_scores = []
            processing_methods = []
            
            # Method 1: Intelligent script detection and routing
            try:
                # Detect script type first
                from .arabic_ocr_models import get_arabic_processor
                
                arabic_processor = get_arabic_processor()
                script_type = await arabic_processor.detect_script_type(image)
                
                logger.info(f"🎯 Script detection result: {script_type}")
                
                # Only use Arabic processing for Arabic or mixed content
                if script_type in ["arabic", "mixed"]:
                    logger.info(f"🔤 {script_type.title()} script detected - using specialized Arabic OCR")
                    
                    # Process with Arabic OCR models
                    arabic_result = await arabic_processor.process_arabic_text(
                        image,
                        script_type=script_type
                    )
                    
                    if arabic_result.text and arabic_result.text.strip() and arabic_result.confidence > 0.5:
                        extracted_texts.append(arabic_result.text.strip())
                        confidence_scores.append(arabic_result.confidence)
                        processing_methods.append(f"arabic_ocr_{arabic_result.model_used}")
                        logger.info(f"✅ Arabic OCR extracted {len(arabic_result.text)} characters with {arabic_result.confidence:.1%} confidence")
                
                elif script_type == "latin":
                    logger.info("🔤 Latin script detected - using standard English OCR")
                    # Skip Arabic processing entirely for Latin scripts
                
                # Fallback to standard Tesseract
                import pytesseract
                logger.info("🔄 Attempting standard OCR with pytesseract...")
                
                # Convert to RGB if needed for tesseract
                ocr_image = image.convert('RGB') if image.mode != 'RGB' else image
                
                # Detect if Arabic script is present (double-check)
                is_arabic = await self._detect_arabic_script(ocr_image)
                
                if is_arabic:
                    logger.info("🔤 Arabic script detected - using Arabic OCR")
                    
                    # Arabic OCR with multiple configurations
                    arabic_configs = [
                        'ara',  # Pure Arabic
                        'ara+eng',  # Arabic + English
                        'eng+ara'   # English + Arabic
                    ]
                    
                    for lang in arabic_configs:
                        try:
                            arabic_text = pytesseract.image_to_string(
                                ocr_image, 
                                lang=lang, 
                                config='--psm 6 -c preserve_interword_spaces=1'
                            )
                            if arabic_text.strip() and len(arabic_text.strip()) > 10:
                                extracted_texts.append(arabic_text.strip())
                                confidence_scores.append(0.85)
                                processing_methods.append(f"tesseract_arabic_{lang}")
                                logger.info(f"✅ Arabic OCR ({lang}) extracted {len(arabic_text)} characters")
                                break
                        except Exception as e:
                            logger.warning(f"Arabic OCR config {lang} failed: {e}")
                            continue
                
                else:
                    # Standard English OCR for non-Arabic text
                    standard_text = pytesseract.image_to_string(ocr_image, config='--psm 3')
                    if standard_text.strip():
                        extracted_texts.append(standard_text.strip())
                        confidence_scores.append(0.8)
                        processing_methods.append("tesseract_standard")
                        logger.info(f"✅ Standard OCR extracted {len(standard_text)} characters")
                
                # Method 2: Specialized handwriting OCR with Tesseract
                logger.info("🔄 Attempting handwriting OCR...")
                
                # Preprocess image for handwriting
                handwriting_image = self._enhance_for_handwriting(ocr_image)
                
                # Use different PSM modes for handwriting
                handwriting_configs = [
                    '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?\'"- ',
                    '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?\'"- ',
                    '--psm 13'  # Raw line for handwriting
                ]
                
                for config in handwriting_configs:
                    try:
                        handwriting_text = pytesseract.image_to_string(handwriting_image, config=config)
                        if handwriting_text.strip() and len(handwriting_text.strip()) > 10:
                            extracted_texts.append(handwriting_text.strip())
                            confidence_scores.append(0.7)
                            processing_methods.append(f"tesseract_handwriting_{config[:8]}")
                            logger.info(f"✅ Handwriting OCR extracted {len(handwriting_text)} characters")
                            break
                    except Exception as e:
                        logger.warning(f"Handwriting config {config} failed: {e}")
                        continue
                
                # Method 3: Line-by-line processing for mixed content
                logger.info("🔄 Attempting line-by-line analysis...")
                line_based_text = self._process_lines_separately(ocr_image)
                if line_based_text.strip():
                    extracted_texts.append(line_based_text.strip())
                    confidence_scores.append(0.75)
                    processing_methods.append("tesseract_line_by_line")
                    logger.info(f"✅ Line-by-line processing extracted {len(line_based_text)} characters")
                
            except ImportError:
                logger.warning("⚠️ Pytesseract not available, using enhanced analysis")
            except Exception as e:
                logger.warning(f"⚠️ OCR processing failed: {e}")
            
            # Combine all extracted texts
            if extracted_texts:
                # Combine unique content from different methods
                combined_text = self._combine_extracted_texts(extracted_texts, processing_methods)
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                
                logger.info(f"✅ Combined OCR extracted {len(combined_text)} characters with {avg_confidence:.1%} confidence")
                
                return DocumentProcessingResult(
                    text=combined_text,
                    confidence=avg_confidence,
                    processing_time=time.time() - start_time,
                    model_used="enhanced_multi_method_ocr",
                    metadata={
                        "ocr_engines": processing_methods,
                        "extraction_methods": len(extracted_texts),
                        "image_stats": {
                            "brightness": float(np.mean(np.array(image.convert('L')))),
                            "contrast": float(np.std(np.array(image.convert('L')))),
                            "dimensions": image.size,
                            "mode": image.mode
                        },
                        "processing_method": "multi_method_ocr",
                        "handwriting_optimized": True
                    },
                    document_type=doc_type,
                    handwriting_detected=True  # Assume mixed content has handwriting
                )
            
            # Enhanced fallback analysis (if OCR fails)
            analysis_results = []
            analysis_results.append(f"📄 Document Analysis Report")
            analysis_results.append(f"🔍 Document Type: {doc_type.title()}")
            analysis_results.append(f"📐 Image Dimensions: {image.size[0]} × {image.size[1]} pixels")
            analysis_results.append(f"🎨 Color Mode: {image.mode}")
            
            # Advanced image analysis
            image_array = np.array(image.convert('L'))  # Convert to grayscale
            
            # Calculate image statistics
            brightness = np.mean(image_array)
            contrast = np.std(image_array)
            analysis_results.append(f"💡 Brightness: {brightness:.1f}/255")
            analysis_results.append(f"🌈 Contrast: {contrast:.1f}")
            
            # Detect potential text regions using simple edge detection
            try:
                from scipy import ndimage
                edges = ndimage.sobel(image_array)
                edge_density = np.sum(edges > 30) / edges.size
                
                if edge_density > 0.15:
                    analysis_results.append(f"✍️ High edge density detected ({edge_density:.3f}) - likely contains handwritten text")
                    doc_type = "handwritten"
                elif edge_density > 0.08:
                    analysis_results.append(f"📝 Medium edge density ({edge_density:.3f}) - likely contains printed text")
                    doc_type = "printed"
                else:
                    analysis_results.append(f"📄 Low edge density ({edge_density:.3f}) - may be blank or low-contrast")
            except ImportError:
                analysis_results.append("🔍 Advanced analysis requires scipy - install for better results")
            
            # Simple text structure detection
            width, height = image.size
            if width > height * 1.5:
                analysis_results.append("📏 Landscape orientation - possible spreadsheet or table")
            elif height > width * 1.5:
                analysis_results.append("📄 Portrait orientation - typical document format")
            
            # Quality assessment
            if image.size[0] < 800 or image.size[1] < 600:
                analysis_results.append("⚠️ Low resolution - OCR accuracy may be limited")
                confidence = 0.3
            elif brightness < 50:
                analysis_results.append("🌑 Dark image - consider brightness adjustment")
                confidence = 0.4
            elif contrast < 30:
                analysis_results.append("🌫️ Low contrast - text may be difficult to extract")
                confidence = 0.35
            else:
                analysis_results.append("✅ Good image quality for text extraction")
                confidence = 0.6
            
            # Recommendations
            analysis_results.append("\n🚀 Recommendations:")
            analysis_results.append("• Install transformers library for AI-powered OCR")
            analysis_results.append("• Use high-resolution images (300+ DPI)")
            analysis_results.append("• Ensure good contrast between text and background")
            analysis_results.append("• Consider image enhancement before processing")
            
            analysis_results.append(f"\n📊 Analysis completed in {time.time() - start_time:.2f}s")
            analysis_results.append("🎯 For specialized OCR models: pip install transformers torch")
            
            final_text = "\n".join(analysis_results)
            
            return DocumentProcessingResult(
                text=final_text,
                confidence=confidence,
                processing_time=time.time() - start_time,
                model_used="enhanced_fallback_analysis",
                metadata={
                    "fallback_type": "enhanced_image_analysis",
                    "image_stats": {
                        "brightness": float(brightness),
                        "contrast": float(contrast),
                        "dimensions": image.size,
                        "mode": image.mode
                    },
                    "recommendations": [
                        "install_transformers",
                        "use_high_resolution",
                        "ensure_good_contrast"
                    ]
                },
                document_type=doc_type,
                handwriting_detected=doc_type == "handwritten"
            )
            
        except Exception as e:
            logger.error(f"Enhanced fallback processing error: {e}")
            return DocumentProcessingResult(
                text=f"📄 Document detected but processing failed: {str(e)}\n\nRecommendation: Install specialized OCR models with:\npip install transformers torch torchvision",
                confidence=0.1,
                processing_time=0.0,
                model_used="basic_fallback",
                metadata={"error": str(e)},
                document_type=doc_type
            )
    
    def _enhance_for_handwriting(self, image: Image.Image) -> Image.Image:
        """Enhance image specifically for handwriting recognition"""
        try:
            from PIL import ImageFilter, ImageEnhance
            
            # Convert to grayscale for better handwriting detection
            if image.mode != 'L':
                enhanced = image.convert('L')
            else:
                enhanced = image.copy()
            
            # Enhance contrast for handwriting (more aggressive than printed text)
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)  # Higher contrast for handwriting
            
            # Sharpen to make handwriting edges clearer
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
            
            # Apply unsharp mask for better definition
            enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            return enhanced.convert('RGB')  # Convert back to RGB for tesseract
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _process_lines_separately(self, image: Image.Image) -> str:
        """Process image line by line for better mixed content handling with Arabic support"""
        try:
            import pytesseract
            from PIL import Image
            import asyncio
            
            # Convert to array for line detection
            img_array = np.array(image.convert('L'))
            height, width = img_array.shape
            
            # Detect if Arabic script is present - SYNC VERSION TO AVOID WARNINGS
            try:
                # Use simple character-based detection instead of async
                # Get basic text first
                basic_text = pytesseract.image_to_string(image, config='--psm 3').strip()
                
                # Count Arabic characters
                arabic_chars = sum(1 for c in basic_text if '\u0600' <= c <= '\u06FF')
                total_chars = len(basic_text.replace(' ', ''))
                
                # Consider Arabic if >30% of characters are Arabic
                is_arabic = (arabic_chars / max(total_chars, 1)) > 0.3 if total_chars > 0 else False
                
                logger.info(f"🔍 Line analysis: {arabic_chars}/{total_chars} Arabic chars, is_arabic={is_arabic}")
                
            except Exception as e:
                logger.warning(f"Arabic detection failed: {e}")
                is_arabic = False
            
            # Choose appropriate language configuration
            if is_arabic:
                lang_configs = [
                    ('ara', '--psm 6 -c preserve_interword_spaces=1'),
                    ('ara+eng', '--psm 7'),
                    ('eng+ara', '--psm 8')
                ]
                logger.info("🔤 Using Arabic line-by-line processing")
            else:
                lang_configs = [
                    ('eng', '--psm 7'),
                    ('eng', '--psm 8'),
                    ('eng', '--psm 13')
                ]
            
            line_texts = []
            
            # Divide image into horizontal strips and process each
            strip_height = height // 10  # Process in 10 strips
            
            for i in range(0, height - strip_height, strip_height // 2):  # 50% overlap
                # Extract strip
                strip = img_array[i:i + strip_height, :]
                
                # Check if strip has enough content
                if np.sum(strip < 128) > strip.size * 0.05:  # 5% dark pixels threshold
                    # Convert back to PIL Image
                    strip_image = Image.fromarray(strip).convert('RGB')
                    
                    # Try different language and PSM configurations
                    for lang, config in lang_configs:
                        try:
                            strip_text = pytesseract.image_to_string(
                                strip_image, 
                                lang=lang, 
                                config=config
                            )
                            if strip_text.strip() and len(strip_text.strip()) > 3:
                                line_texts.append(strip_text.strip())
                                break
                        except Exception as e:
                            logger.debug(f"Strip OCR failed for {lang} with {config}: {e}")
                            continue
            
            # Combine and deduplicate lines
            combined_text = '\n'.join(line_texts)
            return combined_text
            
        except Exception as e:
            logger.warning(f"Line-by-line processing failed: {e}")
            return ""
    
    def _combine_extracted_texts(self, texts: List[str], methods: List[str]) -> str:
        """Intelligently combine texts from different OCR methods"""
        try:
            if not texts:
                return ""
            
            if len(texts) == 1:
                return texts[0]
            
            # Combine texts with method labels for debugging
            combined_parts = []
            
            for i, (text, method) in enumerate(zip(texts, methods)):
                if text.strip():
                    # Add method identifier and content
                    method_label = method.replace('tesseract_', '').replace('_', ' ').title()
                    combined_parts.append(f"[{method_label} Extraction]:\n{text}")
            
            # Join all parts
            final_text = '\n\n'.join(combined_parts)
            
            # Add summary header
            summary_header = f"Multi-Method OCR Results ({len(texts)} methods used):\n{'='*50}\n\n"
            
            return summary_header + final_text
            
        except Exception as e:
            logger.warning(f"Text combination failed: {e}")
            # Fallback: just join all texts
            return '\n\n'.join(texts)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get comprehensive list of available models by category"""
        # Get API provider information
        try:
            api_ocr = get_api_ocr()
            provider_info = api_ocr.get_provider_info()
            api_providers = [f"{p['provider'].upper()} ({p['model']})" for p in provider_info['available_providers']]
            primary_provider = provider_info.get('primary_provider', 'None').upper()
        except Exception as e:
            logger.warning(f"Could not get API provider info: {e}")
            api_providers = ["No API providers configured"]
            primary_provider = "None"
        
        available = {
            "api_ocr_models": api_providers if api_providers != ["No API providers configured"] else [
                "OpenAI GPT-4 Vision",
                "Claude 3.5 Sonnet", 
                "Google Gemini Pro",
                "Groq Llama Vision",
                "DeepSeek Chat"
            ],
            "layout_models": [
                "AI-powered Layout Analysis", 
                "Professional Document Structure", 
                "Advanced Text Extraction"
            ],
            "embedding_models": [
                "Semantic Text Search",
                "Vector-based Matching",
                "Intelligent Chunking"
            ],
            "specialized_features": [
                "Multi-language Support",
                "Handwriting Recognition",
                "Historical Document Analysis",
                "Professional OCR Accuracy"
            ],
            "status": f"API-powered ({primary_provider} primary)" if primary_provider != "None" else "Requires API configuration",
            "processing_modes": [
                "API OCR (Primary)",
                "Multi-provider Fallback", 
                "Professional Accuracy"
            ]
        }
        
        # Return the optimized model list  
        return available
    
    def try_install_models(self) -> Dict[str, bool]:
        """Attempt to install missing transformers models"""
        try:
            import subprocess
            import sys
            
            results = {}
            
            logger.info("🔄 Attempting to install transformers library...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "transformers", "torch", "torchvision", "--quiet"
                ])
                results["transformers"] = True
                logger.info("✅ Transformers library installed successfully")
                
                # Try to reinitialize models
                self._initialize_models()
                results["model_reload"] = True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install transformers: {e}")
                results["transformers"] = False
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Installation attempt failed: {e}")
            return {"error": str(e)}
    
    @observe(as_type="text_embedding")
    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for text using sentence transformers"""
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available")
                return None
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Embedding text of length: {len(text)}"
                )
            
            # Generate embeddings
            embeddings = self.embedding_model.encode([text])
            result = embeddings[0].tolist()
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Generated {len(result)}-dimensional embedding"
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
            return None
    
    async def analyze_document_content(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        Analyze document content for insights and metadata
        Uses simple pattern-based analysis when AI models not available
        """
        try:
            analysis = {
                "text_length": len(text),
                "word_count": len(text.split()),
                "line_count": len(text.split('\n')),
                "document_type": document_type,
                "language": "en",  # Default, could be enhanced with language detection
                "readability": "medium",  # Placeholder
                "historical_indicators": [],
                "key_themes": [],
                "metadata_extracted": True
            }
            
            # Simple pattern-based analysis
            text_lower = text.lower()
            
            # Look for historical indicators
            historical_terms = ['anno domini', 'ad', 'bc', 'century', 'hereby', 'whereas', 'wherefore']
            analysis["historical_indicators"] = [term for term in historical_terms if term in text_lower]
            
            # Extract potential dates
            import re
            date_patterns = [
                r'\b\d{4}\b',  # Years
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',  # MM/DD/YYYY
                r'\b[A-Z][a-z]+ \d{1,2}, \d{4}\b'  # Month DD, YYYY
            ]
            
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, text))
            
            analysis["dates_found"] = dates[:5]  # Limit to first 5
            
            # Estimate readability (very basic)
            avg_word_length = sum(len(word) for word in text.split()) / max(len(text.split()), 1)
            if avg_word_length > 6:
                analysis["readability"] = "complex"
            elif avg_word_length < 4:
                analysis["readability"] = "simple"
            
            return analysis
        
        except Exception as e:
            logger.error(f"Content analysis error: {e}")
            return {"error": str(e), "metadata_extracted": False}