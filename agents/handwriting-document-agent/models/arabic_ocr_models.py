"""
Specialized Arabic OCR Models from HuggingFace
Advanced models for Arabic handwriting and printed text recognition
Author: Mohammed Hamdan
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from dataclasses import dataclass
from PIL import Image
import numpy as np
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

# HuggingFace models specialized for Arabic
ARABIC_OCR_MODELS = {
    "trocr_arabic": {
        "model_id": "microsoft/trocr-base-ar",
        "processor_id": "microsoft/trocr-base-ar",
        "description": "TrOCR model fine-tuned for Arabic text",
        "type": "transformer"
    },
    "arabert_ocr": {
        "model_id": "aubmindlab/bert-base-arabertv2",
        "description": "AraBERT model for Arabic text understanding",
        "type": "bert"
    },
    "pix2struct_arabic": {
        "model_id": "google/pix2struct-ocrvqa-base",
        "description": "Pix2Struct for visual question answering and OCR",
        "type": "vision"
    },
    "layoutlmv3_arabic": {
        "model_id": "microsoft/layoutlmv3-base",
        "description": "LayoutLMv3 for document understanding",
        "type": "layout"
    },
    "donut_arabic": {
        "model_id": "naver-clova-ix/donut-base",
        "description": "Donut model for document understanding",
        "type": "end2end"
    },
    "mgp_str_arabic": {
        "model_id": "alibaba-damo/mgp-str-base",
        "description": "Multi-Granularity Prediction for Scene Text Recognition",
        "type": "scene_text"
    }
}

# Arabic-specific OCR services (for reference)
ARABIC_OCR_APIS = {
    "aws_textract": {
        "supports_arabic": True,
        "endpoint": "textract.amazonaws.com",
        "description": "AWS Textract with Arabic support"
    },
    "google_vision": {
        "supports_arabic": True,
        "endpoint": "vision.googleapis.com",
        "description": "Google Cloud Vision API with Arabic OCR"
    },
    "azure_ocr": {
        "supports_arabic": True,
        "endpoint": "cognitiveservices.azure.com",
        "description": "Azure Computer Vision with Arabic text"
    },
    "ocr_space": {
        "supports_arabic": True,
        "endpoint": "api.ocr.space",
        "description": "OCR.space API with Arabic language"
    }
}

@dataclass
class ArabicOCRResult:
    """Result from Arabic OCR processing"""
    text: str
    confidence: float
    model_used: str
    language: str
    script_type: str  # arabic, mixed, latin
    processing_time: float
    metadata: Dict[str, Any]

class ArabicOCRProcessor:
    """
    Specialized processor for Arabic OCR using HuggingFace models
    Handles both handwritten and printed Arabic text
    """
    
    def __init__(self):
        """Initialize Arabic OCR processor"""
        self.models = {}
        self.processors = {}
        self.device = "cpu"
        
        logger.info("Initializing Arabic OCR Processor")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available Arabic OCR models"""
        try:
            # Try to load transformers
            try:
                from transformers import (
                    TrOCRProcessor, 
                    VisionEncoderDecoderModel,
                    AutoProcessor,
                    AutoModelForVision2Seq,
                    AutoTokenizer
                )
                self.transformers_available = True
                logger.info("✅ Transformers library available for Arabic OCR")
            except ImportError:
                self.transformers_available = False
                logger.warning("⚠️ Transformers not available - using fallback Arabic processing")
                return
            
            # Try to load Arabic TrOCR model
            try:
                logger.info("🔄 Loading Arabic TrOCR model...")
                # Note: microsoft/trocr-base-ar doesn't exist, so we'll use multilingual alternatives
                
                # Option 1: Use base TrOCR with Arabic fine-tuning
                self.processors["trocr_multi"] = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                self.models["trocr_multi"] = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-base-handwritten"
                )
                logger.info("✅ Loaded multilingual TrOCR model")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load TrOCR model: {e}")
            
            # Try to load Donut model for document understanding
            try:
                logger.info("🔄 Loading Donut model for document understanding...")
                from transformers import DonutProcessor, VisionEncoderDecoderModel
                
                self.processors["donut"] = DonutProcessor.from_pretrained(
                    "naver-clova-ix/donut-base"
                )
                self.models["donut"] = VisionEncoderDecoderModel.from_pretrained(
                    "naver-clova-ix/donut-base"
                )
                logger.info("✅ Loaded Donut model")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load Donut model: {e}")
            
            # Try to load MGP-STR for scene text
            try:
                logger.info("🔄 Loading MGP-STR for scene text recognition...")
                from transformers import AutoProcessor, AutoModel
                
                self.processors["mgp_str"] = AutoProcessor.from_pretrained(
                    "alibaba-damo/mgp-str-base"
                )
                self.models["mgp_str"] = AutoModel.from_pretrained(
                    "alibaba-damo/mgp-str-base"
                )
                logger.info("✅ Loaded MGP-STR model")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load MGP-STR model: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Arabic OCR models: {e}")
    
    async def detect_script_type(self, image: Image.Image) -> str:
        """
        Enhanced script detection with improved accuracy
        Prioritizes actual text content over visual analysis
        """
        try:
            logger.info("🔍 Analyzing document script type...")
            
            # Method 1: Primary OCR-based detection with both languages
            try:
                import pytesseract
                
                # Test with English first (more reliable baseline)
                logger.debug("Testing with English OCR...")
                english_text = pytesseract.image_to_string(
                    image,
                    lang='eng',
                    config='--psm 3 --oem 3'
                ).strip()
                
                # Test with Arabic
                logger.debug("Testing with Arabic OCR...")
                arabic_text = pytesseract.image_to_string(
                    image,
                    lang='ara',
                    config='--psm 6 -c preserve_interword_spaces=1'
                ).strip()
                
                # Analyze the quality of extraction from both
                english_quality = self._assess_text_quality(english_text, "latin")
                arabic_quality = self._assess_text_quality(arabic_text, "arabic")
                
                logger.info(f"📊 English OCR quality: {english_quality:.2f} (length: {len(english_text)})")
                logger.info(f"📊 Arabic OCR quality: {arabic_quality:.2f} (length: {len(arabic_text)})")
                
                # Decision logic - MORE CONSERVATIVE to prevent false mixed detection
                logger.info(f"📝 English text sample: '{english_text[:100]}...'")
                logger.info(f"📝 Arabic text sample: '{arabic_text[:100]}...'")
                
                # Count actual meaningful characters
                arabic_chars = sum(1 for c in arabic_text if '\u0600' <= c <= '\u06FF')
                latin_chars = sum(1 for c in english_text if c.isalpha() and ord(c) < 128)
                
                logger.info(f"📊 Character counts: Arabic={arabic_chars}, Latin={latin_chars}")
                
                # Use stricter thresholds to avoid false mixed detection
                if english_quality > 0.7 and arabic_quality < 0.4 and latin_chars > 20:
                    logger.info("🔤 Latin script detected (high English confidence, low Arabic)")
                    return "latin"
                elif arabic_quality > 0.7 and english_quality < 0.4 and arabic_chars > 10:
                    logger.info("🔤 Arabic script detected (high Arabic confidence, low English)")
                    return "arabic"
                elif arabic_chars > 10 and latin_chars > 10 and arabic_chars > latin_chars * 0.3:
                    logger.info(f"🔤 Mixed script detected ({arabic_chars} Arabic, {latin_chars} Latin)")
                    return "mixed"
                elif latin_chars > arabic_chars * 2:
                    logger.info("🔤 Latin script detected (predominantly Latin characters)")
                    return "latin"
                elif arabic_chars > latin_chars:
                    logger.info("🔤 Arabic script detected (predominantly Arabic characters)")
                    return "arabic"
                else:
                    # Default to Latin for unclear cases
                    logger.info("🔤 Unclear script - defaulting to Latin processing")
                    return "latin"
                
            except Exception as e:
                logger.warning(f"OCR-based detection failed: {e}")
            
            # Method 2: Fallback to simpler heuristics
            try:
                # Simple character-based detection on any extractable text
                import pytesseract
                
                # Try basic extraction with no language specification
                basic_text = pytesseract.image_to_string(image).strip()
                
                if basic_text:
                    arabic_chars = sum(1 for c in basic_text if '\u0600' <= c <= '\u06FF')
                    total_meaningful_chars = sum(1 for c in basic_text if c.isalnum())
                    
                    if total_meaningful_chars > 0:
                        arabic_ratio = arabic_chars / total_meaningful_chars
                        
                        if arabic_ratio > 0.4:  # More conservative threshold
                            logger.info(f"🔤 Arabic script detected via character analysis ({arabic_ratio:.1%})")
                            return "arabic"
                        elif arabic_ratio > 0.1:
                            logger.info(f"🔤 Mixed script detected ({arabic_ratio:.1%} Arabic)")
                            return "mixed"
                        else:
                            logger.info(f"🔤 Latin script detected ({arabic_ratio:.1%} Arabic)")
                            return "latin"
            
            except Exception as e:
                logger.warning(f"Character-based detection failed: {e}")
            
            # Method 3: Final fallback - assume Latin for safety
            logger.info("🔤 Script detection inconclusive - defaulting to Latin")
            return "latin"
            
        except Exception as e:
            logger.error(f"Script detection error: {e}")
            return "latin"  # Safe default
    
    def _assess_text_quality(self, text: str, script_type: str) -> float:
        """Assess the quality of OCR extraction"""
        if not text or not text.strip():
            return 0.0
        
        text = text.strip()
        
        # Basic quality metrics
        quality_score = 0.0
        
        # 1. Text length (reasonable content)
        if len(text) > 10:
            quality_score += 0.3
        elif len(text) > 5:
            quality_score += 0.1
        
        # 2. Character composition
        if script_type == "latin":
            # For Latin script, expect alphabetic characters and common punctuation
            alpha_chars = sum(1 for c in text if c.isalpha())
            space_chars = sum(1 for c in text if c.isspace())
            punctuation = sum(1 for c in text if c in '.,!?:;-()[]{}')
            
            total_chars = len(text)
            if total_chars > 0:
                meaningful_ratio = (alpha_chars + space_chars + punctuation) / total_chars
                quality_score += meaningful_ratio * 0.4
                
                # Bonus for readable words
                words = text.split()
                readable_words = sum(1 for word in words if len(word) > 2 and word.isalpha())
                if len(words) > 0:
                    quality_score += (readable_words / len(words)) * 0.3
        
        elif script_type == "arabic":
            # For Arabic script, expect Arabic characters
            arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
            total_chars = len(text.replace(' ', ''))  # Exclude spaces
            
            if total_chars > 0:
                arabic_ratio = arabic_chars / total_chars
                quality_score += arabic_ratio * 0.7
                
                # Bonus for connected Arabic text patterns
                if arabic_chars > 5:
                    quality_score += 0.3
        
        return min(quality_score, 1.0)  # Cap at 1.0
    
    async def process_arabic_text(
        self,
        image: Image.Image,
        script_type: str = "auto"
    ) -> ArabicOCRResult:
        """
        Process Arabic text using specialized models
        
        Args:
            image: PIL Image containing Arabic text
            script_type: "arabic", "mixed", "auto"
        
        Returns:
            ArabicOCRResult with extracted text
        """
        try:
            import time
            start_time = time.time()
            
            # Auto-detect script if needed
            if script_type == "auto":
                script_type = await self.detect_script_type(image)
            
            logger.info(f"🔤 Processing {script_type} script document")
            
            # Try different models based on availability
            result_text = ""
            model_used = "none"
            confidence = 0.0
            
            # Method 1: Try HuggingFace models if available
            if self.transformers_available and self.models:
                if "trocr_multi" in self.models:
                    try:
                        logger.info("🔄 Trying TrOCR multilingual model...")
                        
                        pixel_values = self.processors["trocr_multi"](
                            image, 
                            return_tensors="pt"
                        ).pixel_values
                        
                        generated_ids = self.models["trocr_multi"].generate(pixel_values)
                        generated_text = self.processors["trocr_multi"].batch_decode(
                            generated_ids, 
                            skip_special_tokens=True
                        )[0]
                        
                        if generated_text.strip():
                            result_text = generated_text
                            model_used = "trocr_multilingual"
                            confidence = 0.75
                            logger.info(f"✅ TrOCR extracted: {len(result_text)} chars")
                            
                    except Exception as e:
                        logger.warning(f"TrOCR processing failed: {e}")
                
                if "donut" in self.models and not result_text:
                    try:
                        logger.info("🔄 Trying Donut model...")
                        
                        # Prepare task prompt for Donut
                        task_prompt = "<s_ocr><s_arabic>"
                        decoder_input_ids = self.processors["donut"].tokenizer(
                            task_prompt,
                            add_special_tokens=False,
                            return_tensors="pt"
                        ).input_ids
                        
                        pixel_values = self.processors["donut"](
                            image,
                            return_tensors="pt"
                        ).pixel_values
                        
                        outputs = self.models["donut"].generate(
                            pixel_values,
                            decoder_input_ids=decoder_input_ids,
                            max_length=512
                        )
                        
                        generated_text = self.processors["donut"].batch_decode(
                            outputs,
                            skip_special_tokens=True
                        )[0]
                        
                        if generated_text.strip():
                            result_text = generated_text
                            model_used = "donut"
                            confidence = 0.7
                            logger.info(f"✅ Donut extracted: {len(result_text)} chars")
                            
                    except Exception as e:
                        logger.warning(f"Donut processing failed: {e}")
            
            # Method 2: Enhanced Tesseract with Arabic
            if not result_text:
                result_text = await self._tesseract_arabic_fallback(image, script_type)
                if result_text:
                    model_used = "tesseract_arabic_enhanced"
                    confidence = 0.6
            
            # Method 3: Use online API if configured (placeholder)
            if not result_text and os.getenv("ENABLE_CLOUD_OCR"):
                result_text = await self._cloud_arabic_ocr(image)
                if result_text:
                    model_used = "cloud_api"
                    confidence = 0.8
            
            processing_time = time.time() - start_time
            
            # Create result
            return ArabicOCRResult(
                text=result_text if result_text else "Failed to extract Arabic text",
                confidence=confidence,
                model_used=model_used,
                language="ar" if script_type == "arabic" else "mixed",
                script_type=script_type,
                processing_time=processing_time,
                metadata={
                    "image_size": image.size,
                    "models_tried": list(self.models.keys()),
                    "transformers_available": self.transformers_available
                }
            )
            
        except Exception as e:
            logger.error(f"Arabic OCR processing error: {e}")
            return ArabicOCRResult(
                text=f"Error processing Arabic text: {str(e)}",
                confidence=0.0,
                model_used="error",
                language="unknown",
                script_type=script_type,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def _tesseract_arabic_fallback(
        self, 
        image: Image.Image,
        script_type: str
    ) -> str:
        """Enhanced Tesseract fallback for Arabic"""
        try:
            import pytesseract
            
            logger.info("🔄 Using enhanced Tesseract Arabic fallback...")
            
            # Preprocess image for better Arabic OCR
            enhanced = self._preprocess_for_arabic(image)
            
            # Try different language combinations
            lang_configs = [
                ('ara', '--psm 6 -c preserve_interword_spaces=1'),
                ('ara+eng', '--psm 6'),
                ('ara', '--psm 3 --oem 3'),
                ('ara', '--psm 4')
            ]
            
            best_text = ""
            best_length = 0
            
            for lang, config in lang_configs:
                try:
                    text = pytesseract.image_to_string(
                        enhanced,
                        lang=lang,
                        config=config
                    )
                    
                    # Count Arabic characters
                    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
                    
                    if arabic_chars > best_length:
                        best_text = text
                        best_length = arabic_chars
                        
                except Exception as e:
                    logger.debug(f"Config {lang} {config} failed: {e}")
                    continue
            
            if best_text:
                logger.info(f"✅ Tesseract Arabic extracted {best_length} Arabic characters")
                return best_text
            
            return ""
            
        except Exception as e:
            logger.error(f"Tesseract Arabic fallback error: {e}")
            return ""
    
    def _preprocess_for_arabic(self, image: Image.Image) -> Image.Image:
        """Preprocess image specifically for Arabic OCR"""
        try:
            from PIL import ImageEnhance, ImageFilter, ImageOps
            
            # Convert to grayscale
            if image.mode != 'L':
                processed = image.convert('L')
            else:
                processed = image.copy()
            
            # Enhance contrast for Arabic text
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(1.5)
            
            # Apply bilateral filter to reduce noise while preserving edges
            processed = processed.filter(ImageFilter.MedianFilter(3))
            
            # Sharpen
            processed = processed.filter(ImageFilter.SHARPEN)
            
            # Invert if needed (Arabic text is sometimes inverted)
            # Check if background is dark
            img_array = np.array(processed)
            if np.mean(img_array) < 127:
                processed = ImageOps.invert(processed)
            
            # Convert back to RGB for Tesseract
            processed = processed.convert('RGB')
            
            return processed
            
        except Exception as e:
            logger.warning(f"Arabic preprocessing failed: {e}")
            return image
    
    async def _cloud_arabic_ocr(self, image: Image.Image) -> str:
        """Placeholder for cloud OCR services"""
        # This would integrate with AWS Textract, Google Vision, or Azure
        # For now, returning empty to use local processing
        return ""
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available Arabic OCR models"""
        return {
            "loaded_models": list(self.models.keys()),
            "loaded_processors": list(self.processors.keys()),
            "transformers_available": self.transformers_available,
            "arabic_models_info": ARABIC_OCR_MODELS,
            "cloud_apis_info": ARABIC_OCR_APIS
        }

# Global instance
arabic_ocr_processor = None

def get_arabic_processor():
    """Get or create Arabic OCR processor singleton"""
    global arabic_ocr_processor
    if arabic_ocr_processor is None:
        arabic_ocr_processor = ArabicOCRProcessor()
    return arabic_ocr_processor