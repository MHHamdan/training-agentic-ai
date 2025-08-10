#!/bin/bash

# Docker Hub Push Script for Training Agentic AI Platform
# This script uses credentials from .env file

# Load environment variables
source .env

DOCKER_USERNAME=${DOCKER_USERNAME:-440930}
VERSION="v1.0"
LATEST="latest"

echo "🔐 Logging in to Docker Hub as ${DOCKER_USERNAME}..."
echo "${DOCKER_TOKEN}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed. Please check your credentials."
    exit 1
fi

echo "🏷️  Tagging images for Docker Hub..."
# Tag orchestrator
docker tag training-agentic-ai/orchestrator:latest ${DOCKER_USERNAME}/training-agentic-ai:orchestrator-${VERSION}
docker tag training-agentic-ai/orchestrator:latest ${DOCKER_USERNAME}/training-agentic-ai:orchestrator

# Tag legal-document-review
docker tag training-agentic-ai/legal-document-review:latest ${DOCKER_USERNAME}/training-agentic-ai:legal-${VERSION}
docker tag training-agentic-ai/legal-document-review:latest ${DOCKER_USERNAME}/training-agentic-ai:legal

# Tag customer-support-agent
docker tag training-agentic-ai/customer-support-agent:latest ${DOCKER_USERNAME}/training-agentic-ai:support-${VERSION}
docker tag training-agentic-ai/customer-support-agent:latest ${DOCKER_USERNAME}/training-agentic-ai:support

echo "📤 Pushing images to Docker Hub..."
# Push orchestrator
echo "Pushing orchestrator..."
docker push ${DOCKER_USERNAME}/training-agentic-ai:orchestrator-${VERSION}
docker push ${DOCKER_USERNAME}/training-agentic-ai:orchestrator

# Push legal-document-review
echo "Pushing legal document review..."
docker push ${DOCKER_USERNAME}/training-agentic-ai:legal-${VERSION}
docker push ${DOCKER_USERNAME}/training-agentic-ai:legal

# Push customer-support-agent
echo "Pushing customer support..."
docker push ${DOCKER_USERNAME}/training-agentic-ai:support-${VERSION}
docker push ${DOCKER_USERNAME}/training-agentic-ai:support

echo "✅ All images pushed successfully!"
echo ""
echo "📋 Your images are now available at Docker Hub:"
echo "  - ${DOCKER_USERNAME}/training-agentic-ai:orchestrator"
echo "  - ${DOCKER_USERNAME}/training-agentic-ai:legal"
echo "  - ${DOCKER_USERNAME}/training-agentic-ai:support"
echo ""
echo "🚀 To pull and run the complete stack:"
echo "  docker pull ${DOCKER_USERNAME}/training-agentic-ai:orchestrator"
echo "  docker pull ${DOCKER_USERNAME}/training-agentic-ai:legal"
echo "  docker pull ${DOCKER_USERNAME}/training-agentic-ai:support"