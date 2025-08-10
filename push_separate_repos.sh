#!/bin/bash

# Push script with separate repositories for each service
source .env

DOCKER_USERNAME="440930"
VERSION="v1.0"

echo "🔐 Logging in to Docker Hub..."
echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed"
    exit 1
fi

echo "🏷️  Tagging and pushing orchestrator..."
docker tag training-agentic-ai/orchestrator:latest ${DOCKER_USERNAME}/agentic-ai-orchestrator:${VERSION}
docker tag training-agentic-ai/orchestrator:latest ${DOCKER_USERNAME}/agentic-ai-orchestrator:latest
docker push ${DOCKER_USERNAME}/agentic-ai-orchestrator:${VERSION}
docker push ${DOCKER_USERNAME}/agentic-ai-orchestrator:latest

echo "🏷️  Tagging and pushing legal document review..."
docker tag training-agentic-ai/legal-document-review:latest ${DOCKER_USERNAME}/agentic-ai-legal:${VERSION}
docker tag training-agentic-ai/legal-document-review:latest ${DOCKER_USERNAME}/agentic-ai-legal:latest
docker push ${DOCKER_USERNAME}/agentic-ai-legal:${VERSION}
docker push ${DOCKER_USERNAME}/agentic-ai-legal:latest

echo "🏷️  Tagging and pushing customer support..."
docker tag training-agentic-ai/customer-support-agent:latest ${DOCKER_USERNAME}/agentic-ai-support:${VERSION}
docker tag training-agentic-ai/customer-support-agent:latest ${DOCKER_USERNAME}/agentic-ai-support:latest
docker push ${DOCKER_USERNAME}/agentic-ai-support:${VERSION}
docker push ${DOCKER_USERNAME}/agentic-ai-support:latest

echo "✅ All images pushed successfully!"
echo ""
echo "📋 Your Docker Hub repositories:"
echo "  - ${DOCKER_USERNAME}/agentic-ai-orchestrator"
echo "  - ${DOCKER_USERNAME}/agentic-ai-legal"
echo "  - ${DOCKER_USERNAME}/agentic-ai-support"