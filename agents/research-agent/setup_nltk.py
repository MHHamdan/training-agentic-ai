#!/usr/bin/env python3
"""
NLTK Setup Script for Research Agent
Downloads required NLTK data with SSL workaround
"""

import ssl
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Download required NLTK data with SSL workaround"""
    try:
        # Try to download normally first
        logger.info("Attempting normal NLTK downloads...")
        
        # Create SSL context that doesn't verify certificates (for local development only)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Required NLTK data packages
        packages = [
            'punkt',
            'punkt_tab', 
            'stopwords',
            'averaged_perceptron_tagger',
            'maxent_ne_chunker', 
            'words'
        ]
        
        for package in packages:
            try:
                logger.info(f"Downloading {package}...")
                nltk.download(package, quiet=True)
                logger.info(f"✅ Successfully downloaded {package}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download {package}: {e}")
        
        logger.info("NLTK setup completed!")
        return True
        
    except Exception as e:
        logger.error(f"NLTK setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_nltk()