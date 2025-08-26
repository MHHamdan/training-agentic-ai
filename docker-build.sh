#!/bin/bash

# Training-Agentic-AI Docker Build and Push Script
# This script builds and pushes the Docker image to Docker Hub

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Docker Hub configuration
DOCKER_USERNAME="440930"
IMAGE_NAME="training-agentic-ai"
IMAGE_TAG="${1:-latest}"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}🐳 Training-Agentic-AI Docker Build Script${NC}"
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    echo -e "${YELLOW}📋 Loading environment variables...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

# Login to Docker Hub
echo -e "${YELLOW}🔐 Logging into Docker Hub...${NC}"
echo "${DOCKER_TOKEN}" | docker login -u "${DOCKER_USERNAME}" --password-stdin

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker login failed. Please check your credentials.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Successfully logged into Docker Hub${NC}"

# Build the Docker image
echo -e "${YELLOW}🔨 Building Docker image: ${FULL_IMAGE_NAME}${NC}"
docker build -t ${FULL_IMAGE_NAME} .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Tag image as latest if not already
if [ "${IMAGE_TAG}" != "latest" ]; then
    echo -e "${YELLOW}🏷️  Tagging image as latest...${NC}"
    docker tag ${FULL_IMAGE_NAME} ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
fi

# Push the image to Docker Hub
echo -e "${YELLOW}📤 Pushing image to Docker Hub...${NC}"
docker push ${FULL_IMAGE_NAME}

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker push failed.${NC}"
    exit 1
fi

# Push latest tag if applicable
if [ "${IMAGE_TAG}" != "latest" ]; then
    docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest
fi

echo -e "${GREEN}✅ Successfully pushed image to Docker Hub${NC}"

# Display summary
echo ""
echo "=================================="
echo -e "${GREEN}🎉 Build and Push Complete!${NC}"
echo ""
echo "Image: ${FULL_IMAGE_NAME}"
echo "Docker Hub: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
echo ""
echo "To run locally with Docker Compose:"
echo "  docker-compose up -d"
echo ""
echo "To pull and run the image:"
echo "  docker pull ${FULL_IMAGE_NAME}"
echo "  docker run -p 8500:8500 --env-file .env ${FULL_IMAGE_NAME}"
echo "=================================="