#!/usr/bin/env python3
"""
Test Script for ARIA - Autogen Research Intelligence Agent
Comprehensive testing suite for all ARIA components
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    print("🔍 Testing imports...")
    
    try:
        # Test config imports
        from config.autogen_config import (
            get_autogen_config, 
            get_research_assistant_config,
            get_user_proxy_config,
            get_research_templates,
            get_conversation_starters
        )
        print("✅ Config imports successful")
        
        # Test tools imports
        from tools.research_tools import (
            WebSearchTool, 
            AcademicSearchTool, 
            ContentAnalyzer, 
            get_research_tools
        )
        from tools.export_tools import ResearchExporter, get_export_capabilities
        print("✅ Tools imports successful")
        
        # Test autogen components imports
        from autogen_components.research_assistant import (
            create_research_assistant, 
            get_research_assistant_capabilities
        )
        from autogen_components.user_proxy import (
            create_enhanced_user_proxy, 
            get_user_proxy_capabilities
        )
        from autogen_components.conversation_manager import (
            create_conversation_manager, 
            get_conversation_manager_capabilities
        )
        print("✅ Autogen components imports successful")
        
        # Test UI imports
        from ui.streamlit_interface import create_research_interface
        print("✅ UI imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration functionality"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.autogen_config import (
            get_autogen_config,
            get_research_templates,
            get_conversation_starters
        )
        
        # Test autogen config
        config = get_autogen_config()
        assert isinstance(config, dict), "Config should be a dictionary"
        assert "config_list" in config, "Config should have config_list"
        print("✅ Autogen config generation successful")
        
        # Test research templates
        templates = get_research_templates()
        assert isinstance(templates, dict), "Templates should be a dictionary"
        assert len(templates) > 0, "Should have at least one template"
        print("✅ Research templates generation successful")
        
        # Test conversation starters
        starters = get_conversation_starters()
        assert isinstance(starters, list), "Starters should be a list"
        assert len(starters) > 0, "Should have at least one starter"
        print("✅ Conversation starters generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {str(e)}")
        traceback.print_exc()
        return False


def test_research_tools():
    """Test research tools functionality"""
    print("\n🔍 Testing research tools...")
    
    try:
        from tools.research_tools import WebSearchTool, AcademicSearchTool, ContentAnalyzer
        
        # Test web search tool
        web_tool = WebSearchTool()
        search_result = web_tool.search("artificial intelligence", max_results=3)
        assert isinstance(search_result, dict), "Search result should be a dictionary"
        assert "results" in search_result, "Should have results key"
        print("✅ Web search tool functional")
        
        # Test academic search tool
        academic_tool = AcademicSearchTool()
        academic_result = academic_tool.search_academic("machine learning", max_results=3)
        assert isinstance(academic_result, dict), "Academic result should be a dictionary"
        print("✅ Academic search tool functional")
        
        # Test content analyzer
        analyzer = ContentAnalyzer()
        sample_text = "This is a sample text for analysis. It contains multiple sentences and provides a good test case for our content analysis functionality."
        analysis = analyzer.analyze_text(sample_text, "basic")
        assert isinstance(analysis, dict), "Analysis should be a dictionary"
        assert "metrics" in analysis, "Should have metrics"
        print("✅ Content analyzer functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Research tools error: {str(e)}")
        traceback.print_exc()
        return False


def test_export_tools():
    """Test export functionality"""
    print("\n📁 Testing export tools...")
    
    try:
        from tools.export_tools import ResearchExporter, get_export_capabilities
        
        # Test export capabilities
        capabilities = get_export_capabilities()
        assert isinstance(capabilities, dict), "Capabilities should be a dictionary"
        assert "supported_formats" in capabilities, "Should have supported formats"
        print("✅ Export capabilities check successful")
        
        # Test research exporter with sample data
        sample_data = {
            'conversation': [
                {
                    'sender': 'user',
                    'content': 'What is artificial intelligence?',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'sender': 'assistant', 
                    'content': 'Artificial intelligence (AI) is a field of computer science...',
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'research_state': {
                'current_topic': 'Artificial Intelligence',
                'research_depth': 'intermediate',
                'target_audience': 'general'
            }
        }
        
        exporter = ResearchExporter(sample_data)
        
        # Test JSON export
        json_export = exporter.export_to_json()
        assert isinstance(json_export, str), "JSON export should be a string"
        json.loads(json_export)  # Verify it's valid JSON
        print("✅ JSON export functional")
        
        # Test Markdown export
        md_export = exporter.export_to_markdown()
        assert isinstance(md_export, str), "Markdown export should be a string"
        assert "# ARIA Research Report" in md_export, "Should have proper title"
        print("✅ Markdown export functional")
        
        # Test CSV export
        csv_export = exporter.export_to_csv()
        assert isinstance(csv_export, str), "CSV export should be a string"
        assert "Timestamp,Sender,Content" in csv_export, "Should have proper headers"
        print("✅ CSV export functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Export tools error: {str(e)}")
        traceback.print_exc()
        return False


def test_autogen_components():
    """Test Autogen components (basic functionality without actual LLM calls)"""
    print("\n🤖 Testing Autogen components...")
    
    try:
        from autogen_components.research_assistant import get_research_assistant_capabilities
        from autogen_components.user_proxy import get_user_proxy_capabilities
        from autogen_components.conversation_manager import get_conversation_manager_capabilities
        
        # Test capabilities functions
        research_caps = get_research_assistant_capabilities()
        assert isinstance(research_caps, dict), "Research capabilities should be a dictionary"
        print("✅ Research assistant capabilities check successful")
        
        user_caps = get_user_proxy_capabilities()
        assert isinstance(user_caps, dict), "User proxy capabilities should be a dictionary"
        print("✅ User proxy capabilities check successful")
        
        conv_caps = get_conversation_manager_capabilities()
        assert isinstance(conv_caps, dict), "Conversation manager capabilities should be a dictionary"
        print("✅ Conversation manager capabilities check successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Autogen components error: {str(e)}")
        traceback.print_exc()
        return False


def test_api_configuration():
    """Test API configuration"""
    print("\n🔑 Testing API configuration...")
    
    try:
        # Check for API keys
        api_keys = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "HUGGING_FACE_API": os.getenv("HUGGING_FACE_API"),
            "HF_TOKEN": os.getenv("HF_TOKEN")
        }
        
        available_keys = [k for k, v in api_keys.items() if v]
        
        if available_keys:
            print(f"✅ Found API keys: {', '.join(available_keys)}")
        else:
            print("⚠️ No API keys found - agent will run in fallback mode")
        
        return True
        
    except Exception as e:
        print(f"❌ API configuration error: {str(e)}")
        return False


def test_file_structure():
    """Test file structure and permissions"""
    print("\n📁 Testing file structure...")
    
    try:
        base_path = Path(__file__).parent
        
        # Check required directories
        required_dirs = [
            "config",
            "autogen_components", 
            "tools",
            "ui"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_path / dir_name
            assert dir_path.exists(), f"Directory {dir_name} should exist"
            assert dir_path.is_dir(), f"{dir_name} should be a directory"
            print(f"✅ Directory {dir_name} exists")
        
        # Check required files
        required_files = [
            "app.py",
            "config/autogen_config.py",
            "autogen_components/research_assistant.py",
            "autogen_components/user_proxy.py", 
            "autogen_components/conversation_manager.py",
            "tools/research_tools.py",
            "tools/export_tools.py",
            "ui/streamlit_interface.py"
        ]
        
        for file_name in required_files:
            file_path = base_path / file_name
            assert file_path.exists(), f"File {file_name} should exist"
            assert file_path.is_file(), f"{file_name} should be a file"
            print(f"✅ File {file_name} exists")
        
        return True
        
    except Exception as e:
        print(f"❌ File structure error: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("🔬 ARIA - Autogen Research Intelligence Agent Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Research Tools", test_research_tools),
        ("Export Tools", test_export_tools),
        ("Autogen Components", test_autogen_components),
        ("API Configuration", test_api_configuration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        result = test_func()
        test_results.append((test_name, result))
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ARIA is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)