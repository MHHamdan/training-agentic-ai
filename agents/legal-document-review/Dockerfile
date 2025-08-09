FROM python:3.11-slim

WORKDIR /app

# Copy root requirements (build context is the repo root)
COPY requirements.txt .

# Install Python dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PROGRESS_BAR=off \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt --progress-bar off

# Copy application code from the subdirectory
COPY agents/legal-document-review/ .

# Create non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

# Set environment variables to reduce resource usage
ENV STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200 \
    STREAMLIT_SERVER_ENABLE_STATIC_SERVING=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]

