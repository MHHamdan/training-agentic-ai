"""
Fast, Accurate OCR Models - Lightweight and Efficient
Optimized for speed and accuracy without heavy dependencies
Author: Mohammed Hamdan
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
import numpy as np

logger = logging.getLogger(__name__)

class FastOCRProcessor:
    """
    Ultra-fast OCR processor optimized for performance
    No heavy dependencies, maximum accuracy
    """
    
    def __init__(self):
        """Initialize fast OCR processor"""
        self.supported_langs = self._get_available_languages()
        logger.info(f"🚀 Fast OCR initialized with {len(self.supported_langs)} languages")
    
    def _get_available_languages(self) -> list:
        """Get available Tesseract languages"""
        try:
            langs = pytesseract.get_languages(config='')
            logger.info(f"✅ Available languages: {langs}")
            return langs
        except Exception as e:
            logger.warning(f"⚠️ Could not get languages: {e}")
            return ['eng']
    
    def detect_script_fast(self, image: Image.Image) -> str:
        """
        Ultra-fast script detection using character analysis
        Returns: 'english', 'arabic', or 'mixed'
        """
        try:
            # Quick English test
            eng_text = pytesseract.image_to_string(image, lang='eng', config='--psm 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ')
            eng_chars = sum(1 for c in eng_text if c.isalpha() and ord(c) < 128)
            
            # Quick Arabic test if available
            arabic_chars = 0
            if 'ara' in self.supported_langs:
                ara_text = pytesseract.image_to_string(image, lang='ara', config='--psm 3')
                arabic_chars = sum(1 for c in ara_text if '\u0600' <= c <= '\u06FF')
            
            logger.info(f"📊 Fast detection: English={eng_chars}, Arabic={arabic_chars}")
            
            # Simple decision logic
            if eng_chars > 10 and arabic_chars < 5:
                return 'english'
            elif arabic_chars > 5 and eng_chars < 10:
                return 'arabic'
            elif arabic_chars > 5 and eng_chars > 10:
                return 'mixed'
            else:
                return 'english'  # Default to English
                
        except Exception as e:
            logger.error(f"Fast detection failed: {e}")
            return 'english'
    
    def preprocess_image_fast(self, image: Image.Image) -> Image.Image:
        """
        Fast image preprocessing for better OCR
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to grayscale
            gray = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray)
            enhanced = enhancer.enhance(1.8)
            
            # Sharpen
            sharpened = enhanced.filter(ImageFilter.SHARPEN)
            
            # Auto-invert if background is dark
            img_array = np.array(sharpened)
            if np.mean(img_array) < 127:
                sharpened = ImageOps.invert(sharpened)
            
            return sharpened.convert('RGB')
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image
    
    def extract_text_fast(self, image: Image.Image) -> Dict[str, Any]:
        """
        Ultra-fast text extraction with multiple methods
        Returns best result from all attempts
        """
        start_time = time.time()
        results = {}
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image_fast(image)
            
            # Detect script
            script_type = self.detect_script_fast(processed_image)
            logger.info(f"🎯 Detected script: {script_type}")
            
            # Method 1: Standard English OCR
            if script_type in ['english', 'mixed']:
                eng_text = pytesseract.image_to_string(
                    processed_image,
                    lang='eng',
                    config='--psm 3 --oem 3'
                )
                if eng_text.strip():
                    results['english'] = {
                        'text': eng_text,
                        'confidence': self._estimate_confidence(eng_text),
                        'method': 'tesseract_english'
                    }
            
            # Method 2: Arabic OCR (if available and needed)
            if script_type in ['arabic', 'mixed'] and 'ara' in self.supported_langs:
                ara_text = pytesseract.image_to_string(
                    processed_image,
                    lang='ara',
                    config='--psm 6 -c preserve_interword_spaces=1'
                )
                if ara_text.strip():
                    results['arabic'] = {
                        'text': ara_text,
                        'confidence': self._estimate_confidence(ara_text),
                        'method': 'tesseract_arabic'
                    }
            
            # Method 3: Handwriting specialized
            if script_type == 'english':
                hw_text = pytesseract.image_to_string(
                    processed_image,
                    lang='eng',
                    config='--psm 8 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!? '
                )
                if hw_text.strip():
                    results['handwriting'] = {
                        'text': hw_text,
                        'confidence': self._estimate_confidence(hw_text),
                        'method': 'tesseract_handwriting'
                    }
            
            # Select best result
            best_result = self._select_best_result(results)
            
            processing_time = time.time() - start_time
            
            return {
                'text': best_result.get('text', ''),
                'confidence': best_result.get('confidence', 0.0),
                'method': best_result.get('method', 'unknown'),
                'script_type': script_type,
                'processing_time': processing_time,
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Fast OCR failed: {e}")
            return {
                'text': f'OCR Error: {str(e)}',
                'confidence': 0.0,
                'method': 'error',
                'script_type': 'unknown',
                'processing_time': time.time() - start_time,
                'all_results': {}
            }
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate OCR confidence based on text characteristics
        """
        if not text or not text.strip():
            return 0.0
        
        text = text.strip()
        score = 0.0
        
        # Length score
        if len(text) > 10:
            score += 0.3
        elif len(text) > 5:
            score += 0.1
        
        # Character composition
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.replace(' ', ''))
        
        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            score += alpha_ratio * 0.4
        
        # Word structure
        words = text.split()
        readable_words = sum(1 for word in words if len(word) > 2 and word.isalpha())
        if len(words) > 0:
            score += (readable_words / len(words)) * 0.3
        
        return min(score, 1.0)
    
    def _select_best_result(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best OCR result from multiple attempts
        """
        if not results:
            return {'text': '', 'confidence': 0.0, 'method': 'none'}
        
        # Find result with highest confidence and reasonable text length
        best_score = 0.0
        best_result = {'text': '', 'confidence': 0.0, 'method': 'none'}
        
        for method, result in results.items():
            confidence = result.get('confidence', 0.0)
            text_length = len(result.get('text', '').strip())
            
            # Weighted score: confidence + text length bonus
            score = confidence + (min(text_length / 100, 0.2))  # Max 20% bonus for length
            
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result


# Global instance for fast access
fast_ocr_processor = None

def get_fast_ocr() -> FastOCRProcessor:
    """Get or create fast OCR processor singleton"""
    global fast_ocr_processor
    if fast_ocr_processor is None:
        fast_ocr_processor = FastOCRProcessor()
    return fast_ocr_processor