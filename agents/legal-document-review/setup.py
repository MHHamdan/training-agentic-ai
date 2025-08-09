#!/usr/bin/env python
"""
Setup script for Legal Document Review Application
This script will help you set up the environment and test the application
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print application banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     Legal Document Review Assistant - Setup Script          ║
    ║                   Powered by LangChain & Gemini             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Your version: {sys.version}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\n📦 Creating virtual environment...")
    if os.path.exists("venv"):
        print("   Virtual environment already exists")
        return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n📚 Installing dependencies...")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join("venv", "Scripts", "pip")
    else:  # Unix/Linux/Mac
        pip_path = os.path.join("venv", "bin", "pip")
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        print("   Installing packages from requirements.txt...")
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_env_file():
    """Check and create .env file if needed"""
    print("\n🔐 Checking environment configuration...")
    
    if os.path.exists(".env"):
        print("✅ .env file found")
        
        # Check if API key is set
        with open(".env", "r") as f:
            content = f.read()
            if "GOOGLE_API_KEY=" in content and "your_" not in content:
                print("✅ API key appears to be configured")
                print("\n⚠️  SECURITY WARNING:")
                print("   The API key in your .env file may have been exposed.")
                print("   Please regenerate it at: https://makersuite.google.com/app/apikey")
                return True
    else:
        print("📝 Creating .env file from template...")
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as source:
                with open(".env", "w") as dest:
                    dest.write(source.read())
            print("✅ .env file created")
            print("⚠️  Please add your Google Gemini API key to the .env file")
        else:
            print("❌ .env.example file not found")
    
    return True

def create_sample_documents():
    """Create sample documents directory"""
    print("\n📄 Setting up sample documents...")
    
    if not os.path.exists("sample_documents"):
        os.makedirs("sample_documents")
        print("✅ Created sample_documents directory")
    else:
        print("   sample_documents directory already exists")
    
    # Create a sample text file with instructions
    sample_path = os.path.join("sample_documents", "README.txt")
    with open(sample_path, "w") as f:
        f.write("""Sample Legal Documents Directory
================================

Place your legal PDF documents here for testing.

You can find free legal document templates at:
- https://www.lawinsider.com/
- https://legaltemplates.net/
- https://wonder.legal/

Or convert the sample_nda.txt to PDF using any online converter.

Test Documents:
1. Non-Disclosure Agreements (NDAs)
2. Service Agreements
3. Software License Agreements
4. Terms of Service
5. Employment Contracts
""")
    print("✅ Sample documents directory ready")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    packages = [
        ("streamlit", "Streamlit"),
        ("langchain", "LangChain"),
        ("PyPDF2", "PyPDF2"),
        ("faiss", "FAISS"),
        ("dotenv", "python-dotenv"),
        ("google.generativeai", "Google Generative AI")
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name} imported successfully")
        except ImportError:
            print(f"   ❌ Failed to import {name}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Some packages failed to import: {', '.join(failed)}")
        return False
    
    print("\n✅ All packages imported successfully!")
    return True

def print_instructions():
    """Print instructions for running the app"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    print("\n📋 Next Steps:\n")
    
    print("1. IMPORTANT SECURITY NOTICE:")
    print("   ⚠️  If you've shared your API key, regenerate it immediately at:")
    print("      https://makersuite.google.com/app/apikey")
    print()
    
    print("2. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    print()
    
    print("3. Run the application:")
    print("   streamlit run app.py")
    print()
    
    print("4. The app will open in your browser at:")
    print("   http://localhost:8501")
    print()
    
    print("5. Upload a legal PDF document and start asking questions!")
    print()
    
    print("📚 Sample Questions to Try:")
    print("   - What are the termination clauses?")
    print("   - What is the duration of this agreement?")
    print("   - What are the confidentiality obligations?")
    print("   - Generate a summary of this document")
    print()
    
    print("For more information, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check environment file
    check_env_file()
    
    # Create sample documents directory
    create_sample_documents()
    
    # Test imports (using the venv Python)
    print("\n🔄 Testing installation...")
    if os.name == 'nt':  # Windows
        python_path = os.path.join("venv", "Scripts", "python")
    else:  # Unix/Linux/Mac
        python_path = os.path.join("venv", "bin", "python")
    
    test_script = """
import sys
try:
    import streamlit
    import langchain
    import PyPDF2
    import faiss
    import dotenv
    import google.generativeai
    print("SUCCESS")
except ImportError as e:
    print(f"FAILED: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run(
        [python_path, "-c", test_script],
        capture_output=True,
        text=True
    )
    
    if "SUCCESS" in result.stdout:
        print("✅ All packages verified!")
    else:
        print(f"⚠️  Some packages may not be properly installed: {result.stderr}")
    
    # Print final instructions
    print_instructions()

if __name__ == "__main__":
    main()