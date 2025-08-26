#!/bin/bash

# Multi-Agent AI Platform Stop Script
# This script stops all running streamlit agents

echo "🛑 Stopping Multi-Agent AI Platform..."
echo "=================================="

# Function to stop processes on specific ports
stop_port() {
    local port=$1
    local name=$2
    
    echo "Stopping $name (port $port)..."
    
    # Find and kill processes using the port
    local pids=$(lsof -ti :$port)
    
    if [ -z "$pids" ]; then
        echo "❌ No process found on port $port"
    else
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Stopped process $pid on port $port"
            else
                echo "⚠️  Could not stop process $pid"
            fi
        done
    fi
}

# Alternative method: Kill all streamlit processes
stop_all_streamlit() {
    echo ""
    echo "Stopping all Streamlit processes..."
    
    local pids=$(pgrep -f streamlit)
    
    if [ -z "$pids" ]; then
        echo "❌ No Streamlit processes found"
    else
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo "✅ Stopped Streamlit process $pid"
            else
                echo "⚠️  Could not stop process $pid"
            fi
        done
    fi
}

# Stop agents by port
stop_port "8500" "Main Dashboard"
stop_port "8501" "Legal Document Review"
stop_port "8502" "Customer Support Agent"
stop_port "8503" "Finance Advisor Agent"
stop_port "8504" "Competitive Intel Agent"
stop_port "8505" "Insights Explorer Agent"
stop_port "8506" "Customer Support Triage Agent"
stop_port "8507" "Extended Stock Analysis Agent"
stop_port "8508" "Multi-Agent Financial Analysis"
stop_port "8509" "AI Content Creation System"
stop_port "8510" "ARIA Research Agent"
stop_port "8511" "MARIA Medical Research Agent"
stop_port "8512" "Resume Screening Agent"
stop_port "8513" "Stock Analysis Agent"
stop_port "8514" "Research Agent V2"

# Fallback: stop all streamlit processes
stop_all_streamlit

echo ""
echo "=================================="
echo "🎉 All agents have been stopped!"
echo "=================================="