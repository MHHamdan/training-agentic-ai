#!/bin/bash

# Extended Stock Analysis System - Deployment Script

set -e

echo "🚀 Starting Extended Stock Analysis System Deployment"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Please create a .env file with your API keys"
    echo "You can use .env.example as a template"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required commands
if ! command_exists docker-compose; then
    echo "❌ Error: docker-compose is not installed"
    echo "Please install docker-compose and try again"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data reports logs deployment

# Load environment variables
echo "🔧 Loading environment variables..."
export $(cat .env | grep -v '^#' | xargs)

# Validate required API keys
echo "🔑 Validating API keys..."
required_keys=("OPENAI_API_KEY" "GOOGLE_API_KEY")
missing_keys=()

for key in "${required_keys[@]}"; do
    if [ -z "${!key}" ]; then
        missing_keys+=("$key")
    fi
done

if [ ${#missing_keys[@]} -ne 0 ]; then
    echo "⚠️  Warning: Missing API keys: ${missing_keys[*]}"
    echo "The system will still work but with limited functionality"
fi

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check main application
if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo "✅ Stock Analysis App is healthy"
else
    echo "❌ Stock Analysis App is not responding"
fi

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL is healthy"
else
    echo "❌ PostgreSQL is not responding"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

echo ""
echo "🎉 Deployment completed!"
echo ""
echo "📊 Access your applications:"
echo "   Stock Analysis: http://localhost:8501"
echo "   Grafana:        http://localhost:3000 (admin/admin)"
echo "   Prometheus:     http://localhost:9090"
echo ""
echo "📋 View logs with:"
echo "   docker-compose logs -f stock-analysis"
echo ""
echo "🛑 Stop services with:"
echo "   docker-compose down"
echo ""

# Show running containers
echo "🐳 Running containers:"
docker-compose ps