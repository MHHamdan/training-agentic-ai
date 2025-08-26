#!/bin/bash

# Multi-Agent AI Platform Startup Script
# This script starts all agents on their designated ports

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "🚀 Starting Multi-Agent AI Platform..."
echo "=================================="

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $port is already in use"
        return 1
    else
        return 0
    fi
}

# Function to start agent
start_agent() {
    local name=$1
    local script_path=$2
    local port=$3
    
    echo "Starting $name on port $port..."
    
    if check_port $port; then
        streamlit run "$script_path" --server.port $port --server.headless true --server.runOnSave false &
        local pid=$!
        echo "✅ $name started (PID: $pid)"
        return 0
    else
        echo "❌ Failed to start $name - port $port is busy"
        return 1
    fi
}

# Start main dashboard (orchestrator)
echo ""
echo "1. Starting Main Dashboard..."
start_agent "Main Dashboard" "app.py" "8500"

# Start legal document review agent
echo ""
echo "2. Starting Legal Document Review Agent..."
start_agent "Legal Document Review" "agents/legal-document-review/app.py" "8501"

# Start customer support agent
echo ""
echo "3. Starting Customer Support Agent..."
start_agent "Customer Support Agent" "agents/customer-support-agent/src/ui/app.py" "8502"

# Start finance advisor agent
echo ""
echo "4. Starting Finance Advisor Agent..."
start_agent "Finance Advisor Agent" "agents/Finance-Advaisor-Agent/app.py" "8503"

# Start competitive intel agent
echo ""
echo "5. Starting Competitive Intel Agent..."
start_agent "Competitive Intel Agent" "agents/competitive-intel-agent/app.py" "8504"

# Start insights explorer agent
echo ""
echo "6. Starting Insights Explorer Agent..."
start_agent "Insights Explorer Agent" "agents/insights-explorer-agent/app.py" "8505"

# Start customer support triage agent
echo ""
echo "7. Starting Customer Support Triage Agent..."
start_agent "Customer Support Triage Agent" "agents/Customer-Support-Triage/app.py" "8506"

# Start extended stock analysis agent
echo ""
echo "8. Starting Extended Stock Analysis Agent..."
start_agent "Extended Stock Analysis Agent" "agents/stock-analysis-extended/app.py" "8507"

# Start multi-agent financial analysis system
echo ""
echo "9. Starting Multi-Agent Financial Analysis System..."
start_agent "Multi-Agent Financial Analysis System" "agents/multi-agent-financial-analysis/app.py" "8508"

# Start AI content creation system
echo ""
echo "10. Starting AI Content Creation System..."
start_agent "AI Content Creation System" "agents/ai-content-creation-system/app.py" "8509"

# Start ARIA (Autogen Research Intelligence Agent)
echo ""
echo "11. Starting ARIA (Autogen Research Intelligence Agent)..."
start_agent "ARIA Research Agent" "agents/autogen-research-intelligence-agent/app.py" "8510"

# Start MARIA (Medical Research Intelligence Agent)
echo ""
echo "12. Starting MARIA (Medical Research Intelligence Agent)..."
start_agent "MARIA Medical Research Agent" "agents/medical-research-intelligence-agent/app.py" "8511"

# Start Resume Screening Agent
echo ""
echo "13. Starting Resume Screening Agent..."
start_agent "Resume Screening Agent" "agents/resume-screening/run.py" "8512"

# Start Stock Analysis Agent
echo ""
echo "14. Starting Stock Analysis Agent..."
start_agent "Stock Analysis Agent" "agents/stock-analysis/app.py" "8513"

# Start Research Agent V2
echo ""
echo "15. Starting Research Agent V2..."
start_agent "Research Agent V2" "agents/research-agent/app.py" "8514"

echo ""
echo "=================================="
echo "🎉 All agents have been started!"
echo ""
echo "Access your agents at:"
echo "📊 Main Dashboard:              http://localhost:8500"
echo "⚖️  Legal Document Review:       http://localhost:8501"
echo "🎧 Customer Support:            http://localhost:8502"
echo "💰 Finance Advisor:             http://localhost:8503"
echo "🔍 Competitive Intel:           http://localhost:8504"
echo "📊 Insights Explorer:           http://localhost:8505"
echo "🎫 Support Triage:              http://localhost:8506"
echo "📈 Stock Analysis Extended:     http://localhost:8507"
echo "💹 Multi-Agent Financial:       http://localhost:8508"
echo "✍️  AI Content Creation:         http://localhost:8509"
echo "🔬 ARIA Research Agent:         http://localhost:8510"
echo "🏥 MARIA Medical Research:      http://localhost:8511"
echo "📄 Resume Screening:            http://localhost:8512"
echo "📈 Stock Analysis:              http://localhost:8513"
echo "🔬 Research Agent V2:           http://localhost:8514"
echo ""
echo "To stop all agents, run: ./stop_all_agents.sh"
echo "To view logs, use: ps aux | grep streamlit"
echo ""
echo "Press Ctrl+C to stop this script (agents will continue running)"
echo "=================================="

# Keep script running to show it's active
trap 'echo "Script interrupted. Agents are still running in background."' INT
while true; do
    sleep 60
done