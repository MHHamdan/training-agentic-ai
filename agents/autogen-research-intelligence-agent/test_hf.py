#!/usr/bin/env python3
"""
Test Hugging Face integration for ARIA
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from huggingface_hub import InferenceClient
    print("✅ Hugging Face hub imported successfully")
    
    # Get API token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_API")
    print(f"🔑 Token available: {bool(token)}")
    
    if token:
        # Test client initialization
        client = InferenceClient(token=token)
        print("✅ InferenceClient initialized")
        
        # Test simple text generation
        test_prompt = "Hello, can you help me with research?"
        print(f"📝 Testing prompt: {test_prompt}")
        
        models_to_test = ["gpt2", "microsoft/DialoGPT-small"]
        
        for model in models_to_test:
            try:
                print(f"🔄 Testing {model}...")
                response = client.text_generation(
                    test_prompt,
                    model=model,
                    max_new_tokens=50,
                    temperature=0.7,
                    return_full_text=False
                )
                print(f"✅ {model}: {response}")
                break
            except Exception as e:
                import traceback
                print(f"❌ {model} failed: {e}")
                print(f"📋 Full error: {traceback.format_exc()}")
    else:
        print("❌ No Hugging Face token found")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")