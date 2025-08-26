#!/bin/bash

# Docker Build Script for Handwriting & Historical Document Analysis Agent
# Includes Arabic OCR support and multilingual processing

set -e  # Exit on any error

echo "🔧 Building Handwriting Document Agent Docker Image..."
echo "=================================================="

# Configuration
IMAGE_NAME="440930/handwriting-document-agent"
IMAGE_TAG="latest"
AGENT_DIR="agents/handwriting-document-agent"

# Check if we're in the right directory
if [ ! -f "$AGENT_DIR/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "Expected to find: $AGENT_DIR/app.py"
    exit 1
fi

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Error: Docker is not running or not accessible"
    exit 1
fi

echo "📁 Agent Directory: $AGENT_DIR"
echo "🏷️  Image Name: $IMAGE_NAME:$IMAGE_TAG"
echo "📅 Build Date: $(date)"
echo ""

# Build the Docker image
echo "🚀 Building Docker image..."
docker build \
    --tag "$IMAGE_NAME:$IMAGE_TAG" \
    --tag "$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
    --file "$AGENT_DIR/Dockerfile" \
    "$AGENT_DIR"

echo ""
echo "✅ Docker image built successfully!"

# Verify the image
echo "🔍 Verifying image..."
docker images "$IMAGE_NAME:$IMAGE_TAG"

# Test if Tesseract is properly installed
echo ""
echo "🧪 Testing Tesseract installation..."
docker run --rm "$IMAGE_NAME:$IMAGE_TAG" tesseract --version
echo ""
echo "🔤 Testing available languages..."
docker run --rm "$IMAGE_NAME:$IMAGE_TAG" tesseract --list-langs

# Test Python dependencies
echo ""
echo "🐍 Testing Python dependencies..."
docker run --rm "$IMAGE_NAME:$IMAGE_TAG" python -c "
import pytesseract
import cv2
import PIL
import streamlit
import chromadb
print('✅ All core dependencies imported successfully')
"

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Test the container: docker run -p 8516:8516 $IMAGE_NAME:$IMAGE_TAG"
echo "   2. Or use docker-compose: docker-compose up handwriting-document-agent"
echo "   3. Access the agent at: http://localhost:8516"
echo ""
echo "🔧 For Arabic OCR testing:"
echo "   - Upload an Arabic document"
echo "   - Check logs for: '🔤 Arabic script detected'"
echo "   - Verify proper Arabic text extraction"
echo ""
echo "📚 Documentation:"
echo "   - Arabic setup: $AGENT_DIR/ARABIC_OCR_SETUP.md"
echo "   - Environment config: environment.txt"
echo "   - Docker compose: docker-compose.yml"