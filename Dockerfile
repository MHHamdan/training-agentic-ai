# Multi-stage build for Training-Agentic-AI Platform
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create volume for persistent data
VOLUME ["/app/data", "/app/logs"]

# Expose all agent ports
EXPOSE 8500 8501 8502 8503 8504 8505 8506 8507

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8500
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check for main dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8500/_stcore/health || exit 1

# Default command to run the main dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8500", "--server.address=0.0.0.0"]