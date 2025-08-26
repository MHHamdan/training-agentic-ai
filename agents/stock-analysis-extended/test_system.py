#!/usr/bin/env python3
"""
Test script for the Extended Stock Analysis System
This script validates the system setup and runs basic functionality tests
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from workflows.orchestration.workflow_manager import WorkflowOrchestrator, WorkflowType
    from agents.core.risk_assessor import RiskAssessmentAgent
    from agents.core.sentiment_analyzer import SentimentAnalysisAgent
    from agents.core.technical_analyst import TechnicalAnalysisAgent
    from config.settings import settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class SystemTester:
    """Test runner for the stock analysis system"""
    
    def __init__(self):
        self.test_ticker = "AAPL"
        self.results = []
    
    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Log a test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results.append({
            'test': test_name,
            'success': success,
            'message': message
        })
        print(f"{status} {test_name}: {message}")
    
    def test_configuration(self):
        """Test system configuration"""
        print("\n🔧 Testing Configuration...")
        
        # Test API keys
        has_llm_key = bool(settings.api.openai_api_key or settings.api.google_api_key or settings.api.anthropic_api_key)
        self.log_result(
            "LLM API Configuration",
            has_llm_key,
            "At least one LLM API key found" if has_llm_key else "No LLM API keys configured"
        )
        
        # Test financial data APIs
        has_finance_key = bool(settings.api.alpha_vantage_api_key)
        self.log_result(
            "Financial Data API",
            has_finance_key,
            "Alpha Vantage API key found" if has_finance_key else "No financial data API keys (will use Yahoo Finance)"
        )
        
        # Test directory structure
        data_dir_exists = settings.data_dir.exists()
        self.log_result(
            "Data Directory",
            data_dir_exists,
            f"Data directory: {settings.data_dir}"
        )
    
    async def test_individual_agents(self):
        """Test individual agent functionality"""
        print("\n🤖 Testing Individual Agents...")
        
        # Test Technical Analysis Agent
        try:
            tech_agent = TechnicalAnalysisAgent()
            tech_result = await tech_agent.execute({
                'ticker': self.test_ticker,
                'period': '1mo'
            })
            
            success = tech_result.status == 'completed'
            message = "Analysis completed" if success else f"Failed: {tech_result.errors}"
            self.log_result("Technical Analysis Agent", success, message)
            
        except Exception as e:
            self.log_result("Technical Analysis Agent", False, f"Exception: {str(e)}")
        
        # Test Sentiment Analysis Agent
        try:
            sentiment_agent = SentimentAnalysisAgent()
            sentiment_result = await sentiment_agent.execute({
                'ticker': self.test_ticker,
                'company_name': 'Apple'
            })
            
            success = sentiment_result.status == 'completed'
            message = "Analysis completed" if success else f"Failed: {sentiment_result.errors}"
            self.log_result("Sentiment Analysis Agent", success, message)
            
        except Exception as e:
            self.log_result("Sentiment Analysis Agent", False, f"Exception: {str(e)}")
        
        # Test Risk Assessment Agent
        try:
            risk_agent = RiskAssessmentAgent()
            risk_result = await risk_agent.execute({
                'ticker': self.test_ticker,
                'period': '1mo'
            })
            
            success = risk_result.status == 'completed'
            message = "Analysis completed" if success else f"Failed: {risk_result.errors}"
            self.log_result("Risk Assessment Agent", success, message)
            
        except Exception as e:
            self.log_result("Risk Assessment Agent", False, f"Exception: {str(e)}")
    
    async def test_workflow_orchestration(self):
        """Test workflow orchestration"""
        print("\n🔄 Testing Workflow Orchestration...")
        
        try:
            orchestrator = WorkflowOrchestrator()
            
            # Test quick scan workflow
            workflow_result = await orchestrator.execute_workflow(
                ticker=self.test_ticker,
                workflow_type=WorkflowType.QUICK_SCAN,
                custom_params={'period': '1mo'}
            )
            
            success = workflow_result.status == 'completed'
            message = "Quick scan completed" if success else f"Failed: {workflow_result.errors}"
            self.log_result("Quick Scan Workflow", success, message)
            
            if success and workflow_result.final_report:
                has_summary = bool(workflow_result.final_report.get('executive_summary'))
                self.log_result("Report Generation", has_summary, "Executive summary generated")
            
        except Exception as e:
            self.log_result("Workflow Orchestration", False, f"Exception: {str(e)}")
    
    def test_data_access(self):
        """Test data source access"""
        print("\n📊 Testing Data Access...")
        
        try:
            import yfinance as yf
            
            # Test Yahoo Finance access
            ticker = yf.Ticker(self.test_ticker)
            hist = ticker.history(period="5d")
            
            success = not hist.empty
            message = f"Retrieved {len(hist)} days of data" if success else "No data retrieved"
            self.log_result("Yahoo Finance Access", success, message)
            
        except Exception as e:
            self.log_result("Yahoo Finance Access", False, f"Exception: {str(e)}")
        
        # Test DuckDuckGo search (for news)
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{self.test_ticker} stock news", max_results=1))
            
            success = len(results) > 0
            message = f"Retrieved {len(results)} news items" if success else "No news retrieved"
            self.log_result("News Search Access", success, message)
            
        except Exception as e:
            self.log_result("News Search Access", False, f"Exception: {str(e)}")
    
    async def run_all_tests(self):
        """Run all system tests"""
        print("🚀 Starting Extended Stock Analysis System Tests")
        print(f"Test Ticker: {self.test_ticker}")
        print(f"Timestamp: {datetime.now()}")
        print("=" * 60)
        
        # Run tests
        self.test_configuration()
        self.test_data_access()
        await self.test_individual_agents()
        await self.test_workflow_orchestration()
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['test']}: {result['message']}")
        
        print("\n🎯 Recommendations:")
        if not any(r['test'] == 'LLM API Configuration' and r['success'] for r in self.results):
            print("  - Configure at least one LLM API key (OpenAI, Google, or Anthropic)")
        
        if not any(r['test'] == 'Financial Data API' and r['success'] for r in self.results):
            print("  - Consider adding Alpha Vantage API key for enhanced financial data")
        
        if failed_tests == 0:
            print("  - ✅ System is ready for production use!")
        elif passed_tests > failed_tests:
            print("  - ⚠️  System has basic functionality, some features may be limited")
        else:
            print("  - ❌ System needs configuration before use")
        
        return failed_tests == 0


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_ticker = sys.argv[1].upper()
    else:
        test_ticker = "AAPL"
    
    tester = SystemTester()
    tester.test_ticker = test_ticker
    
    # Run async tests
    try:
        success = asyncio.run(tester.run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()