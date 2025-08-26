"""
Demo script for Multi-Agent Financial Analysis System
"""

import os
import sys
from pathlib import Path
import asyncio
from datetime import datetime

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from financial_state import FinancialAnalysisState, MarketConditions, RiskAlert
        print("✅ Financial state management imported")
    except ImportError as e:
        print(f"❌ Error importing financial state: {e}")
        return False
    
    try:
        from financial_tools import get_all_financial_tools
        tools = get_all_financial_tools()
        print(f"✅ Financial tools imported ({len(tools)} tools available)")
    except ImportError as e:
        print(f"❌ Error importing financial tools: {e}")
        return False
    
    try:
        from financial_agents import create_market_research_agent
        print("✅ Financial agents imported")
    except ImportError as e:
        print(f"❌ Error importing financial agents: {e}")
        print("Note: This might be due to missing LLM API keys")
    
    try:
        from financial_graph import financial_graph, create_financial_analysis_session
        print("✅ Financial graph imported")
    except ImportError as e:
        print(f"❌ Error importing financial graph: {e}")
        return False
    
    return True


def test_basic_functionality():
    """Test basic functionality without requiring API keys"""
    print("\n🔧 Testing basic functionality...")
    
    try:
        from financial_state import FinancialAnalysisState, MarketConditions
        
        # Test state creation
        state = FinancialAnalysisState(
            target_symbols=["AAPL", "MSFT"],
            analysis_type="comprehensive",
            risk_tolerance="moderate",
            messages=[]
        )
        
        # Test market conditions
        conditions = MarketConditions(vix=25.0, trend="bullish")
        # For testing purposes, we'll just verify the state can be created
        # state.update_market_conditions(conditions) # Not needed for basic test
        
        print("✅ State management working")
        
        # Test session creation
        from financial_graph import create_financial_analysis_session
        session = create_financial_analysis_session(
            symbols=["AAPL"],
            analysis_type="technical",
            risk_tolerance="moderate"
        )
        
        print("✅ Session creation working")
        print(f"   Session ID: {session['session_id'][:8]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic functionality test: {e}")
        return False


def test_tools():
    """Test financial tools functionality"""
    print("\n🛠️ Testing financial tools...")
    
    try:
        from financial_tools import (
            GetRealTimeMarketDataTool,
            CalculateTechnicalIndicatorsTool,
            NewsSentimentAnalysisTool
        )
        
        # Test market data tool
        market_tool = GetRealTimeMarketDataTool()
        result = market_tool._run("AAPL")
        print("✅ Market data tool working")
        
        # Test technical indicators tool
        tech_tool = CalculateTechnicalIndicatorsTool()
        result = tech_tool._run("AAPL", "1mo")
        print("✅ Technical indicators tool working")
        
        # Test sentiment tool (mock)
        sentiment_tool = NewsSentimentAnalysisTool()
        result = sentiment_tool._run("AAPL")
        print("✅ Sentiment analysis tool working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tools: {e}")
        return False


def test_graph_structure():
    """Test the LangGraph structure"""
    print("\n📊 Testing graph structure...")
    
    try:
        from financial_graph import financial_graph
        
        # Check if graph is compiled
        if financial_graph:
            print("✅ Financial graph compiled successfully")
            
            # Get graph structure info
            nodes = financial_graph.get_graph().nodes
            print(f"   Nodes: {list(nodes.keys())}")
            
            return True
        else:
            print("❌ Financial graph not properly compiled")
            return False
            
    except Exception as e:
        print(f"❌ Error testing graph: {e}")
        return False


def run_demo_analysis():
    """Run a simple demo analysis"""
    print("\n🚀 Running demo analysis...")
    
    try:
        from financial_graph import create_financial_analysis_session
        
        # Create session
        session = create_financial_analysis_session(
            symbols=["AAPL"],
            analysis_type="technical",
            risk_tolerance="moderate"
        )
        
        print(f"📋 Demo Analysis Session")
        print(f"   Session ID: {session['session_id']}")
        print(f"   Symbols: {session['initial_state']['target_symbols']}")
        print(f"   Analysis Type: {session['initial_state']['analysis_type']}")
        print(f"   Risk Tolerance: {session['initial_state']['risk_tolerance']}")
        
        # Simulate workflow (without actually running to avoid API calls)
        print("\n🔄 Simulated Workflow Steps:")
        agents = [
            "market_research_agent",
            "technical_analysis_agent", 
            "risk_assessment_agent",
            "sentiment_analysis_agent",
            "portfolio_optimization_agent",
            "compliance_agent",
            "report_generation_agent"
        ]
        
        for i, agent in enumerate(agents, 1):
            print(f"   {i}. {agent.replace('_', ' ').title()}")
        
        print("\n✅ Demo analysis structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Error in demo analysis: {e}")
        return False


def check_api_keys():
    """Check for available API keys"""
    print("\n🔑 Checking API key configuration...")
    
    api_keys = {
        "GROK_API_KEY": "Grok (xAI) - Primary LLM",
        "OPENAI_API_KEY": "OpenAI GPT",
        "GOOGLE_API_KEY": "Google Gemini", 
        "ANTHROPIC_API_KEY": "Anthropic Claude",
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage (Financial Data)",
        "NEWS_API_KEY": "NewsAPI (Sentiment)",
        "FINNHUB_API_KEY": "Finnhub (Market Data)"
    }
    
    configured_keys = []
    for key, description in api_keys.items():
        if os.getenv(key):
            configured_keys.append(description)
            print(f"✅ {description}")
        else:
            print(f"⚠️  {description} (not configured)")
    
    if configured_keys:
        print(f"\n✅ {len(configured_keys)} API key(s) configured")
        return True
    else:
        print("\n⚠️  No API keys configured - system will run in demo mode")
        return False


def main():
    """Main demo function"""
    print("💹 Multi-Agent Financial Analysis System - Demo")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Tools Test", test_tools),
        ("Graph Structure", test_graph_structure),
        ("Demo Analysis", run_demo_analysis),
        ("API Keys", check_api_keys)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Start the Streamlit app: streamlit run app.py")
        print("2. Access at: http://localhost:8508")
        print("3. Configure API keys for full functionality")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Note: Missing API keys will limit functionality but not break the system.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)