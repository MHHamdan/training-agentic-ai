"""
Document Processing Pipeline with OCR and Advanced Recognition
Handles various document formats with specialized processors
Author: Mohammed Hamdan
"""

import os
import io
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
from datetime import datetime

# Image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
    import cv2
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# PDF processing
try:
    import fitz  # PyMuPDF
    import pdf2image
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# OCR processing
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:
    observe = lambda **kwargs: lambda f: f
    langfuse_context = None

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from models.document_models import DocumentModelManager, DocumentProcessingResult

logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    """Options for document processing"""
    enhance_image: bool = True
    deskew: bool = True
    denoise: bool = True
    auto_rotate: bool = True
    extract_tables: bool = False
    extract_images: bool = False
    ocr_language: str = "eng"
    confidence_threshold: float = 0.7

@dataclass
class DocumentPage:
    """Represents a single document page"""
    page_number: int
    image: Image.Image
    text: str = ""
    confidence: float = 0.0
    bounding_boxes: List[Dict[str, Any]] = None
    tables: List[Dict[str, Any]] = None
    extracted_images: List[Image.Image] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.bounding_boxes is None:
            self.bounding_boxes = []
        if self.tables is None:
            self.tables = []
        if self.extracted_images is None:
            self.extracted_images = []
        if self.metadata is None:
            self.metadata = {}

class DocumentProcessor:
    """
    Advanced document processor with OCR, image enhancement, and multi-format support
    Handles images, PDFs, and various document formats
    """
    
    def __init__(self):
        """Initialize document processor"""
        self.model_manager = DocumentModelManager()
        self.supported_formats = {
            'images': ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.webp', '.jp2'],
            'pdfs': ['.pdf'],
            'text': ['.txt', '.md', '.rtf'],
            'documents': ['.docx', '.doc', '.odt']
        }
        
        # Check available processing capabilities
        self.capabilities = {
            'pil_available': PIL_AVAILABLE,
            'pdf_available': PDF_AVAILABLE,
            'tesseract_available': TESSERACT_AVAILABLE,
            'advanced_models': True  # Our HuggingFace models
        }
        
        logger.info(f"Document processor initialized with capabilities: {self.capabilities}")
    
    @observe(as_type="document_processing")
    async def process_document(
        self,
        file_path: Union[str, Path],
        options: Optional[ProcessingOptions] = None
    ) -> List[DocumentPage]:
        """
        Process document file and extract text/metadata
        
        Args:
            file_path: Path to document file
            options: Processing options
        
        Returns:
            List of processed document pages
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Processing document: {file_path}",
                    metadata={
                        "file_path": str(file_path),
                        "organization": "document-analysis-org",
                        "project": "handwriting-document-agent-v1"
                    }
                )
            
            file_path = Path(file_path)
            options = options or ProcessingOptions()
            
            # Validate file exists
            if not file_path.exists():
                raise FileNotFoundError(f"Document file not found: {file_path}")
            
            # Determine file type and processing method
            file_extension = file_path.suffix.lower()
            
            if file_extension in self.supported_formats['images']:
                pages = await self._process_image_file(file_path, options)
            elif file_extension in self.supported_formats['pdfs']:
                pages = await self._process_pdf_file(file_path, options)
            elif file_extension in self.supported_formats['text']:
                pages = await self._process_text_file(file_path, options)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Processed document: {len(pages)} pages extracted")
            
            if langfuse_context:
                total_text = sum(len(page.text) for page in pages)
                langfuse_context.update_current_observation(
                    output=f"Document processed - {len(pages)} pages, {total_text} characters"
                )
            
            return pages
        
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise
    
    async def _process_image_file(
        self,
        file_path: Path,
        options: ProcessingOptions
    ) -> List[DocumentPage]:
        """Process single image file"""
        try:
            if not PIL_AVAILABLE:
                raise RuntimeError("PIL not available for image processing")
            
            # Load image
            image = Image.open(file_path)
            logger.info(f"Loaded image: {image.size} pixels, mode: {image.mode}")
            
            # Enhance image if requested
            if options.enhance_image:
                image = await self._enhance_image(image, options)
            
            # Process with OCR models
            processing_result = await self.model_manager.process_document_image(
                image=image,
                document_type="auto",
                include_layout=True
            )
            
            # Create document page
            page = DocumentPage(
                page_number=1,
                image=image,
                text=processing_result.text,
                confidence=processing_result.confidence,
                metadata={
                    "file_path": str(file_path),
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "model_used": processing_result.model_used,
                    "document_type": processing_result.document_type,
                    "handwriting_detected": processing_result.handwriting_detected,
                    **processing_result.metadata
                }
            )
            
            return [page]
        
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise
    
    async def _process_pdf_file(
        self,
        file_path: Path,
        options: ProcessingOptions
    ) -> List[DocumentPage]:
        """Process PDF file with multiple pages"""
        try:
            if not PDF_AVAILABLE:
                # Fallback to basic text extraction
                return await self._process_pdf_basic(file_path, options)
            
            pages = []
            
            # Convert PDF pages to images
            pdf_images = pdf2image.convert_from_path(
                str(file_path),
                dpi=300,  # High DPI for better OCR
                grayscale=False
            )
            
            logger.info(f"Converted PDF to {len(pdf_images)} images")
            
            # Process each page
            for page_num, pdf_image in enumerate(pdf_images, 1):
                # Enhance image if requested
                if options.enhance_image:
                    pdf_image = await self._enhance_image(pdf_image, options)
                
                # Process with OCR models
                processing_result = await self.model_manager.process_document_image(
                    image=pdf_image,
                    document_type="auto",
                    include_layout=True
                )
                
                # Create document page
                page = DocumentPage(
                    page_number=page_num,
                    image=pdf_image,
                    text=processing_result.text,
                    confidence=processing_result.confidence,
                    metadata={
                        "file_path": str(file_path),
                        "page_number": page_num,
                        "total_pages": len(pdf_images),
                        "image_size": pdf_image.size,
                        "model_used": processing_result.model_used,
                        "document_type": processing_result.document_type,
                        "handwriting_detected": processing_result.handwriting_detected,
                        **processing_result.metadata
                    }
                )
                
                pages.append(page)
                logger.info(f"Processed PDF page {page_num}: {len(processing_result.text)} characters")
            
            return pages
        
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise
    
    async def _process_pdf_basic(
        self,
        file_path: Path,
        options: ProcessingOptions
    ) -> List[DocumentPage]:
        """Basic PDF text extraction when pdf2image not available"""
        try:
            import fitz  # PyMuPDF for text extraction
            
            pages = []
            pdf_doc = fitz.open(file_path)
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                text = page.get_text()
                
                # Create a simple white image as placeholder
                if PIL_AVAILABLE:
                    placeholder_image = Image.new('RGB', (600, 800), color='white')
                else:
                    placeholder_image = None
                
                doc_page = DocumentPage(
                    page_number=page_num + 1,
                    image=placeholder_image,
                    text=text,
                    confidence=0.9,  # High confidence for direct PDF text
                    metadata={
                        "file_path": str(file_path),
                        "page_number": page_num + 1,
                        "total_pages": pdf_doc.page_count,
                        "extraction_method": "pymupdf_direct",
                        "document_type": "printed"
                    }
                )
                
                pages.append(doc_page)
            
            pdf_doc.close()
            logger.info(f"Extracted text from {len(pages)} PDF pages using PyMuPDF")
            
            return pages
        
        except Exception as e:
            logger.error(f"Basic PDF processing error: {e}")
            raise
    
    async def _process_text_file(
        self,
        file_path: Path,
        options: ProcessingOptions
    ) -> List[DocumentPage]:
        """Process plain text file"""
        try:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create a simple white image as placeholder
            if PIL_AVAILABLE:
                placeholder_image = Image.new('RGB', (600, 800), color='white')
            else:
                placeholder_image = None
            
            page = DocumentPage(
                page_number=1,
                image=placeholder_image,
                text=text,
                confidence=1.0,  # Perfect confidence for text files
                metadata={
                    "file_path": str(file_path),
                    "file_size": len(text),
                    "extraction_method": "direct_read",
                    "document_type": "text",
                    "encoding": "utf-8"
                }
            )
            
            logger.info(f"Read text file: {len(text)} characters")
            return [page]
        
        except Exception as e:
            logger.error(f"Text file processing error: {e}")
            raise
    
    async def _enhance_image(
        self,
        image: Image.Image,
        options: ProcessingOptions
    ) -> Image.Image:
        """Enhance image quality for better OCR"""
        try:
            if not PIL_AVAILABLE:
                return image
            
            enhanced = image.copy()
            
            # Convert to grayscale if needed for processing
            if enhanced.mode != 'L' and enhanced.mode != 'RGB':
                enhanced = enhanced.convert('RGB')
            
            # Denoise
            if options.denoise:
                enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            # Deskew if requested (simple rotation detection)
            if options.deskew:
                enhanced = await self._deskew_image(enhanced)
            
            logger.debug("Image enhancement completed")
            return enhanced
        
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    async def _deskew_image(self, image: Image.Image) -> Image.Image:
        """Simple image deskewing"""
        try:
            # Convert PIL to opencv for deskewing
            if not PIL_AVAILABLE:
                return image
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale for processing
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Find contours
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image if angle is significant
                    if abs(avg_angle) > 0.5:
                        rotated = image.rotate(-avg_angle, expand=True, fillcolor='white')
                        logger.debug(f"Deskewed image by {avg_angle:.2f} degrees")
                        return rotated
            
            return image
        
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return image
    
    @observe(as_type="batch_processing")
    async def process_multiple_documents(
        self,
        file_paths: List[Union[str, Path]],
        options: Optional[ProcessingOptions] = None
    ) -> Dict[str, List[DocumentPage]]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of document file paths
            options: Processing options
        
        Returns:
            Dictionary mapping file paths to processed pages
        """
        try:
            if langfuse_context:
                langfuse_context.update_current_observation(
                    input=f"Batch processing {len(file_paths)} documents",
                    metadata={"document_count": len(file_paths)}
                )
            
            results = {}
            
            # Process documents concurrently (with limit to avoid overwhelming system)
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent processes
            
            async def process_single(file_path):
                async with semaphore:
                    try:
                        pages = await self.process_document(file_path, options)
                        return str(file_path), pages
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        return str(file_path), []
            
            # Process all documents
            tasks = [process_single(path) for path in file_paths]
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            total_pages = 0
            for result in completed_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    continue
                
                file_path, pages = result
                results[file_path] = pages
                total_pages += len(pages)
            
            logger.info(f"Batch processing completed: {len(results)} documents, {total_pages} total pages")
            
            if langfuse_context:
                langfuse_context.update_current_observation(
                    output=f"Batch processing completed - {len(results)} documents, {total_pages} pages"
                )
            
            return results
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            raise
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats"""
        return self.supported_formats
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get current processing capabilities"""
        return {
            **self.capabilities,
            "supported_formats": self.supported_formats,
            "available_models": self.model_manager.get_available_models(),
            "tesseract_available": TESSERACT_AVAILABLE,
            "pdf_processing": PDF_AVAILABLE,
            "image_processing": PIL_AVAILABLE
        }
    
    async def extract_document_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from document file"""
        try:
            file_path = Path(file_path)
            
            # Basic file metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix.lower(),
                "created_time": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            }
            
            # Format-specific metadata
            if file_path.suffix.lower() == '.pdf' and PDF_AVAILABLE:
                pdf_doc = fitz.open(file_path)
                metadata.update({
                    "pdf_pages": pdf_doc.page_count,
                    "pdf_metadata": pdf_doc.metadata
                })
                pdf_doc.close()
            
            elif file_path.suffix.lower() in self.supported_formats['images'] and PIL_AVAILABLE:
                image = Image.open(file_path)
                metadata.update({
                    "image_size": image.size,
                    "image_mode": image.mode,
                    "image_format": image.format
                })
                
                # Extract EXIF data if available
                if hasattr(image, '_getexif') and image._getexif():
                    metadata["exif_data"] = dict(image._getexif())
            
            return metadata
        
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {"error": str(e)}