#!/usr/bin/env python3
"""
Test script to verify Hugging Face models are working correctly
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_manager import ModelManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_generation():
    """Test that model generation works with free Hugging Face models"""
    try:
        logger.info("🧪 Testing Research Agent Model Generation")
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Test query
        test_query = "latest developments in machine learning"
        
        logger.info(f"📝 Testing query: '{test_query}'")
        
        # Test different task types
        task_types = [
            "general_research",
            "analysis", 
            "summarization",
            "fact_checking"
        ]
        
        results = {}
        
        for task_type in task_types:
            logger.info(f"🔄 Testing {task_type}...")
            
            try:
                # Get optimal model (should be free HF model)
                model = model_manager.get_optimal_model(task_type, use_free_models=True)
                logger.info(f"✅ Selected model for {task_type}: {model}")
                
                # Test text generation
                result = await model_manager.generate_text(
                    prompt=f"Research the topic: {test_query}",
                    task_type=task_type,
                    max_tokens=100,
                    temperature=0.7
                )
                
                if result.get("text"):
                    logger.info(f"✅ {task_type} generation successful: {len(result['text'])} chars")
                    results[task_type] = "SUCCESS"
                else:
                    logger.warning(f"⚠️ {task_type} generation returned empty text")
                    results[task_type] = "EMPTY_RESULT"
                
            except Exception as e:
                logger.error(f"❌ {task_type} generation failed: {e}")
                results[task_type] = f"FAILED: {str(e)}"
        
        # Summary
        logger.info("\n📊 TEST RESULTS SUMMARY:")
        logger.info("=" * 50)
        
        success_count = 0
        for task_type, result in results.items():
            status = "✅" if result == "SUCCESS" else "❌"
            logger.info(f"{status} {task_type}: {result}")
            if result == "SUCCESS":
                success_count += 1
        
        logger.info(f"\n📈 Success Rate: {success_count}/{len(task_types)} ({success_count/len(task_types)*100:.1f}%)")
        
        if success_count == len(task_types):
            logger.info("🎉 ALL TESTS PASSED! Research Agent models are working correctly.")
            return True
        else:
            logger.warning(f"⚠️ Some tests failed. Check the logs above for details.")
            return False
            
    except Exception as e:
        logger.error(f"💥 Test execution failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_model_generation())
    sys.exit(0 if result else 1)