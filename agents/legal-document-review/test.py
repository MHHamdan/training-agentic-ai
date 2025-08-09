#!/usr/bin/env python
"""
Quick test script to verify the Legal Document Review Application setup
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is properly configured"""
    print("🔐 Testing API Key Configuration...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ No API key found in environment variables")
        print("   Please add GOOGLE_API_KEY to your .env file")
        return False
    
    if "your_" in api_key or "example" in api_key:
        print("❌ API key appears to be a placeholder")
        print("   Please add your actual API key to the .env file")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    # Test the API key with Google Generative AI
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try to list models
        models = genai.list_models()
        print("✅ API key is valid and working!")
        print(f"   Available models: {len(list(models))}")
        return True
        
    except Exception as e:
        print(f"❌ API key validation failed: {str(e)}")
        print("\n⚠️  IMPORTANT: If this key was exposed publicly, regenerate it at:")
        print("   https://makersuite.google.com/app/apikey")
        return False

def test_langchain_setup():
    """Test LangChain components"""
    print("\n🔗 Testing LangChain Setup...")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        
        print("✅ All LangChain components imported successfully")
        
        # Test text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        test_text = "This is a test document. It has multiple sentences. We will split it into chunks."
        chunks = splitter.split_text(test_text)
        print(f"✅ Text splitter working: {len(chunks)} chunks created")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain test failed: {str(e)}")
        return False

def test_pdf_processing():
    """Test PDF processing capability"""
    print("\n📄 Testing PDF Processing...")
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
        
        # Create a simple test to verify PDF reader works
        print("   PDF processing module is ready")
        return True
        
    except Exception as e:
        print(f"❌ PDF processing test failed: {str(e)}")
        return False

def test_streamlit():
    """Test Streamlit installation"""
    print("\n🎨 Testing Streamlit...")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit version {st.__version__} installed")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit test failed: {str(e)}")
        return False

def create_test_document():
    """Create a sample test document"""
    print("\n📝 Creating test document...")
    
    test_doc = """
    SAMPLE TEST AGREEMENT
    
    This is a test legal document for the Legal Document Review Application.
    
    1. PARTIES
    This agreement is between Party A and Party B.
    
    2. TERM
    This agreement shall be effective for a period of 12 months.
    
    3. CONFIDENTIALITY
    All information shared between parties shall remain confidential for 5 years.
    
    4. TERMINATION
    Either party may terminate this agreement with 30 days written notice.
    
    5. GOVERNING LAW
    This agreement shall be governed by the laws of California.
    """
    
    # Save test document
    with open("test_document.txt", "w") as f:
        f.write(test_doc)
    
    print("✅ Test document created: test_document.txt")
    print("   Note: Convert this to PDF for testing the application")
    return True

def run_quick_rag_test():
    """Run a quick RAG pipeline test"""
    print("\n🤖 Testing RAG Pipeline...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or "your_" in api_key:
        print("⚠️  Skipping RAG test (API key not configured)")
        return False
    
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
        from langchain.vectorstores import FAISS
        from langchain.schema import Document
        
        # Initialize models
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Create test documents
        docs = [
            Document(page_content="This agreement is valid for 12 months."),
            Document(page_content="Termination requires 30 days notice."),
            Document(page_content="Confidentiality period is 5 years.")
        ]
        
        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Test retrieval
        results = vector_store.similarity_search("What is the termination period?", k=1)
        
        if results:
            print("✅ RAG pipeline test successful!")
            print(f"   Retrieved: {results[0].page_content[:50]}...")
            return True
        else:
            print("⚠️  RAG pipeline test completed but no results retrieved")
            return False
            
    except Exception as e:
        print(f"⚠️  RAG pipeline test failed: {str(e)}")
        print("   This might be due to API quota or network issues")
        return False

def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Legal Document Review Application - System Test         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    tests_passed = 0
    tests_total = 6
    
    # Run tests
    if test_api_key():
        tests_passed += 1
    
    if test_langchain_setup():
        tests_passed += 1
    
    if test_pdf_processing():
        tests_passed += 1
    
    if test_streamlit():
        tests_passed += 1
    
    if create_test_document():
        tests_passed += 1
    
    if run_quick_rag_test():
        tests_passed += 1
    
    # Print summary
    print("\n" + "="*60)
    print(f"📊 Test Summary: {tests_passed}/{tests_total} tests passed")
    print("="*60)
    
    if tests_passed == tests_total:
        print("\n✅ All tests passed! Your application is ready to run.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run app.py")
    elif tests_passed >= 4:
        print("\n⚠️  Most tests passed. The application should work with some limitations.")
        print("   Check the failed tests above for details.")
    else:
        print("\n❌ Several tests failed. Please check your setup:")
        print("   1. Ensure all dependencies are installed")
        print("   2. Check your API key in the .env file")
        print("   3. Verify your internet connection")
    
    print("\n📚 For detailed documentation, see README.md")

if __name__ == "__main__":
    main()