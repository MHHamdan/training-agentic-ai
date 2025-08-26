#!/usr/bin/env python3
"""
Comprehensive ARIA Debug Script
Tests the entire conversation flow to identify where messages are lost
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🔬 ARIA Debug Script - Testing Conversation Flow")
print("=" * 60)

# Test 1: Check imports
print("\n1️⃣ Testing imports...")
try:
    from autogen_components.research_assistant import create_research_assistant
    from autogen_components.user_proxy import create_enhanced_user_proxy
    from autogen_components.conversation_manager import AutogenConversationManager
    from config.autogen_config import get_autogen_config
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Mock Streamlit session state
print("\n2️⃣ Creating mock Streamlit session state...")
class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()

class MockSessionState:
    def __init__(self):
        self.aria_conversation_messages = []
        self.aria_research_state = {
            'current_topic': '',
            'research_depth': 'intermediate',
            'target_audience': 'general',
            'subtopics_generated': [],
            'research_completed': {},
            'conversation_active': False,
            'last_response': '',
            'session_id': None
        }

mock_st = MockStreamlit()
print("✅ Mock session state created")
print(f"   📋 Initial messages: {len(mock_st.session_state.aria_conversation_messages)}")

# Test 3: Check API configuration
print("\n3️⃣ Testing API configuration...")
api_keys = {
    "HUGGING_FACE_API": os.getenv("HUGGING_FACE_API"),
    "HF_TOKEN": os.getenv("HF_TOKEN"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
}

available_providers = [k for k, v in api_keys.items() if v]
print(f"🔑 Available API keys: {available_providers}")

# Test 4: Initialize components
print("\n4️⃣ Initializing ARIA components...")
try:
    config = get_autogen_config()
    print(f"✅ Config loaded: {config}")
    
    assistant = create_research_assistant(config)
    print(f"✅ Research assistant created: {type(assistant)}")
    
    user_proxy = create_enhanced_user_proxy(mock_st)
    print(f"✅ User proxy created: {type(user_proxy)}")
    
    conv_manager = AutogenConversationManager(assistant, user_proxy, mock_st)
    print(f"✅ Conversation manager created: {type(conv_manager)}")
    
except Exception as e:
    print(f"❌ Component initialization error: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# Test 5: Test research assistant directly
print("\n5️⃣ Testing research assistant directly...")
try:
    test_prompt = "Impact of AI in Healthcare diagnostic"
    print(f"📝 Testing prompt: {test_prompt}")
    
    response = assistant.generate_research_response(test_prompt)
    print(f"✅ Assistant response generated")
    print(f"📄 Response length: {len(response)} characters")
    print(f"📖 Response preview: {response[:200]}...")
    
except Exception as e:
    print(f"❌ Assistant error: {e}")
    import traceback
    print(traceback.format_exc())

# Test 6: Test conversation manager initiation
print("\n6️⃣ Testing conversation manager initiation...")
try:
    test_topic = "Impact of AI in Healthcare diagnostic"
    research_prompt = f"""
    Please conduct intermediate research on the following topic for a general audience:
    
    Topic: {test_topic}
    
    Please provide:
    1. An overview of the topic
    2. Key concepts and definitions
    3. Current trends and developments
    4. Relevant applications or implications
    5. Potential areas for further investigation
    
    Structure your response clearly and cite relevant sources where possible.
    """
    
    print(f"📝 Research prompt: {research_prompt[:100]}...")
    print(f"📋 Messages before initiation: {len(mock_st.session_state.aria_conversation_messages)}")
    
    # Update session state
    mock_st.session_state.aria_research_state.update({
        'conversation_active': True,
        'current_topic': test_topic,
        'session_id': f"aria_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    })
    
    result = conv_manager.initiate_research(research_prompt)
    print(f"✅ Conversation initiation result: {result}")
    print(f"📋 Messages after initiation: {len(mock_st.session_state.aria_conversation_messages)}")
    
    # Show messages
    if mock_st.session_state.aria_conversation_messages:
        print(f"📨 Messages found:")
        for i, msg in enumerate(mock_st.session_state.aria_conversation_messages):
            print(f"   {i+1}. {msg['sender']}: {msg['content'][:100]}...")
    else:
        print("⚠️ No messages found in session state")
    
except Exception as e:
    print(f"❌ Conversation manager error: {e}")
    import traceback
    print(traceback.format_exc())

# Test 7: Test user proxy initiate_chat method
print("\n7️⃣ Testing user proxy initiate_chat method...")
try:
    if hasattr(user_proxy, 'initiate_chat'):
        print("✅ initiate_chat method exists")
        
        # Test the method
        chat_result = user_proxy.initiate_chat(assistant, "Test message for debugging")
        print(f"✅ initiate_chat result: {chat_result}")
        print(f"📋 Messages after chat: {len(mock_st.session_state.aria_conversation_messages)}")
        
    else:
        print("❌ initiate_chat method not found")
        print(f"📋 Available methods: {[m for m in dir(user_proxy) if not m.startswith('_')]}")
        
except Exception as e:
    print(f"❌ User proxy chat error: {e}")
    import traceback
    print(traceback.format_exc())

# Test 8: Test LLM client directly
print("\n8️⃣ Testing LLM client directly...")
try:
    from config.llm_clients import create_llm_client
    
    llm_client = create_llm_client()
    print(f"✅ LLM client created: {type(llm_client)}")
    print(f"🔧 Provider: {llm_client.provider}")
    print(f"🤖 Model: {llm_client.model}")
    print(f"🔑 Available: {llm_client.is_available()}")
    
    # Test response generation
    test_response = llm_client.generate_response("Hello, can you help with research?")
    print(f"✅ LLM response generated")
    print(f"📄 Response length: {len(test_response)} characters") 
    print(f"📖 Response preview: {test_response[:200]}...")
    
except Exception as e:
    print(f"❌ LLM client error: {e}")
    import traceback
    print(traceback.format_exc())

print("\n" + "=" * 60)
print("🏁 Debug script completed!")
print(f"📊 Final message count: {len(mock_st.session_state.aria_conversation_messages)}")

if mock_st.session_state.aria_conversation_messages:
    print("✅ Messages were successfully created during testing")
    print("🔍 This suggests the issue is in Streamlit session state synchronization")
else:
    print("❌ No messages were created during testing")
    print("🔍 This suggests the issue is in the conversation generation logic")