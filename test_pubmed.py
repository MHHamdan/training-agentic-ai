#!/usr/bin/env python3
"""
Test script to verify PubMed API integration with MARIA
"""

import os
import sys
import requests
from pathlib import Path

# Add MARIA path for imports
agents_dir = Path(__file__).parent / "agents" / "medical-research-intelligence-agent"
sys.path.insert(0, str(agents_dir))

try:
    from tools.medical_research_tools import PubMedSearchTool
    print("✅ Successfully imported PubMedSearchTool")
except ImportError as e:
    print(f"❌ Failed to import PubMedSearchTool: {e}")
    sys.exit(1)

def test_pubmed_api():
    """Test PubMed API connection and search functionality"""
    print("\n🔬 Testing PubMed API Integration...")
    print("=" * 50)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment")
    
    # Check API key
    api_key = os.getenv("PUBMED_API_KEY") or os.getenv("PubMed")
    if api_key:
        print(f"✅ PubMed API key found: {api_key[:8]}****")
    else:
        print("❌ No PubMed API key found in environment")
        return False
    
    # Initialize PubMed tool
    try:
        pubmed_tool = PubMedSearchTool()
        print("✅ PubMedSearchTool initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize PubMedSearchTool: {e}")
        return False
    
    # Test simple search
    print("\n🔍 Testing PubMed search...")
    try:
        # Simple test query
        query = "COVID-19 vaccines efficacy"
        print(f"Query: {query}")
        
        results = pubmed_tool.search_literature(query, max_results=5)
        
        if results.get("success"):
            print(f"✅ Search successful!")
            print(f"   Total results: {results.get('total_results', 0)}")
            print(f"   Returned: {results.get('returned_results', 0)}")
            
            # Display first article if available
            articles = results.get("articles", [])
            if articles:
                first_article = articles[0]
                print(f"\n📄 First article:")
                print(f"   PMID: {first_article.get('pmid', 'N/A')}")
                print(f"   Title: {first_article.get('title', 'N/A')[:80]}...")
                print(f"   Journal: {first_article.get('journal', 'N/A')}")
                print(f"   Authors: {', '.join(first_article.get('authors', [])[:3])}")
                print(f"   Confidence: {first_article.get('confidence_score', 'N/A')}")
            
            return True
        else:
            print(f"❌ Search failed: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return False

def test_api_connection():
    """Test direct API connection"""
    print("\n🌐 Testing direct PubMed API connection...")
    
    api_key = os.getenv("PUBMED_API_KEY") or os.getenv("PubMed")
    
    try:
        # Test basic eSearch
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": "covid",
            "retmax": 1,
            "retmode": "json",
            "tool": "MARIA_test",
            "email": "test@example.com"
        }
        
        if api_key:
            params["api_key"] = api_key
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            result_count = data.get("esearchresult", {}).get("count", "0")
            print(f"✅ Direct API connection successful!")
            print(f"   Response code: {response.status_code}")
            print(f"   Result count for 'covid': {result_count}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Direct API connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🏥 MARIA PubMed API Test Suite")
    print("=" * 50)
    
    # Test direct connection first
    connection_ok = test_api_connection()
    
    # Test PubMed tool
    tool_ok = test_pubmed_api()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   Direct API Connection: {'✅ PASS' if connection_ok else '❌ FAIL'}")
    print(f"   PubMed Tool: {'✅ PASS' if tool_ok else '❌ FAIL'}")
    
    if connection_ok and tool_ok:
        print("\n🎉 All tests passed! MARIA is ready for medical research.")
    else:
        print("\n⚠️  Some tests failed. Check configuration and try again.")
    
    print("=" * 50)