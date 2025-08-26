#!/usr/bin/env python3
"""
Debug script to check research workflow results
"""

import asyncio
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.workflow_manager import workflow_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_research_workflow():
    """Debug the research workflow to see what results are returned"""
    try:
        logger.info("🔍 Debugging Research Workflow")
        
        # Test query
        test_query = "latest developments in artificial intelligence"
        user_preferences = {
            "depth": "comprehensive",
            "citation_format": "APA",
            "enable_fact_check": True,
            "use_free_models": True
        }
        
        logger.info(f"🧪 Running research for: '{test_query}'")
        
        # Run research workflow
        result = await workflow_manager.run_research(
            query=test_query,
            user_preferences=user_preferences
        )
        
        logger.info("📊 Research workflow completed!")
        logger.info(f"Result type: {type(result)}")
        
        # Check what keys are in the result
        if isinstance(result, dict):
            logger.info("🔑 Available keys in result:")
            for key in result.keys():
                value = result[key]
                if isinstance(value, (str, int, float, bool)):
                    logger.info(f"  - {key}: {value}")
                elif isinstance(value, (list, dict)):
                    logger.info(f"  - {key}: {type(value)} (length: {len(value)})")
                else:
                    logger.info(f"  - {key}: {type(value)}")
        
        # Check specific fields that should contain results
        important_fields = [
            "executive_summary", "synthesis", "key_insights", 
            "verified_claims", "disputed_claims", "citations",
            "search_results", "phase", "quality_score"
        ]
        
        logger.info("\n📋 Important fields check:")
        for field in important_fields:
            if field in result:
                value = result[field]
                if isinstance(value, str):
                    logger.info(f"✅ {field}: '{value[:100]}{'...' if len(value) > 100 else ''}'")
                elif isinstance(value, (list, dict)):
                    logger.info(f"✅ {field}: {type(value)} with {len(value)} items")
                else:
                    logger.info(f"✅ {field}: {value}")
            else:
                logger.info(f"❌ {field}: NOT FOUND")
        
        # Save full result to file for inspection
        debug_file = "debug_result.json"
        with open(debug_file, 'w') as f:
            # Convert result to JSON-serializable format
            serializable_result = {}
            for key, value in result.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_result[key] = value
                except (TypeError, ValueError):
                    serializable_result[key] = str(value)
            
            json.dump(serializable_result, f, indent=2, default=str)
        
        logger.info(f"💾 Full result saved to {debug_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(debug_research_workflow())
    print(f"\n🎯 Debug completed. Result available: {result is not None}")