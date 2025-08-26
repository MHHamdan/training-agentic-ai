#!/bin/bash

# Training-Agentic-AI Docker Run Script
# This script runs the multi-agent platform using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Training-Agentic-AI Docker Run Script${NC}"
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env file not found. Please create one with your API keys.${NC}"
    exit 1
fi

# Parse command
COMMAND="${1:-up}"

case ${COMMAND} in
    up|start)
        echo -e "${YELLOW}🔄 Starting all agents...${NC}"
        docker-compose up -d
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ All agents started successfully${NC}"
            echo ""
            echo -e "${BLUE}📊 Access your agents at:${NC}"
            echo "  Main Dashboard:         http://localhost:8500"
            echo "  Legal Document Review:  http://localhost:8501"
            echo "  Customer Support:       http://localhost:8502"
            echo "  Finance Advisor:        http://localhost:8503"
            echo "  Competitive Intel:      http://localhost:8504"
            echo "  Insights Explorer:      http://localhost:8505"
            echo "  Support Triage:         http://localhost:8506"
            echo ""
            echo -e "${YELLOW}📋 View logs:${NC} docker-compose logs -f [service-name]"
            echo -e "${YELLOW}🛑 Stop all:${NC} ./docker-run.sh stop"
        else
            echo -e "${RED}❌ Failed to start agents${NC}"
            exit 1
        fi
        ;;
        
    down|stop)
        echo -e "${YELLOW}🛑 Stopping all agents...${NC}"
        docker-compose down
        echo -e "${GREEN}✅ All agents stopped${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}🔄 Restarting all agents...${NC}"
        docker-compose restart
        echo -e "${GREEN}✅ All agents restarted${NC}"
        ;;
        
    logs)
        SERVICE="${2:-}"
        if [ -z "${SERVICE}" ]; then
            echo -e "${YELLOW}📋 Showing logs for all services...${NC}"
            docker-compose logs -f
        else
            echo -e "${YELLOW}📋 Showing logs for ${SERVICE}...${NC}"
            docker-compose logs -f ${SERVICE}
        fi
        ;;
        
    status|ps)
        echo -e "${YELLOW}📊 Agent Status:${NC}"
        docker-compose ps
        ;;
        
    pull)
        echo -e "${YELLOW}⬇️  Pulling latest images...${NC}"
        docker-compose pull
        echo -e "${GREEN}✅ Images updated${NC}"
        ;;
        
    build)
        echo -e "${YELLOW}🔨 Building images locally...${NC}"
        docker-compose build
        echo -e "${GREEN}✅ Build complete${NC}"
        ;;
        
    clean)
        echo -e "${YELLOW}🧹 Cleaning up volumes and networks...${NC}"
        docker-compose down -v
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
        
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        echo ""
        echo "Usage: ./docker-run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  up|start    - Start all agents"
        echo "  down|stop   - Stop all agents"
        echo "  restart     - Restart all agents"
        echo "  logs [svc]  - View logs (optionally for specific service)"
        echo "  status|ps   - Show agent status"
        echo "  pull        - Pull latest images"
        echo "  build       - Build images locally"
        echo "  clean       - Remove volumes and networks"
        exit 1
        ;;
esac