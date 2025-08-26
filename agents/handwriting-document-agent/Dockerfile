# Dockerfile for Handwriting & Historical Document Analysis Agent
# Supports multilingual OCR including Arabic script processing

FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set locale and encoding for proper UTF-8 support
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# Update system and install core dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    pkg-config \
    # OCR and image processing
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    libtesseract-dev \
    libleptonica-dev \
    # Image processing libraries
    libopencv-dev \
    python3-opencv \
    libpoppler-cpp-dev \
    poppler-utils \
    # PDF processing
    poppler-utils \
    # System utilities
    curl \
    wget \
    git \
    # Graphics and fonts for proper text rendering
    fonts-liberation \
    fonts-dejavu-core \
    fontconfig \
    # Additional dependencies
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Verify Tesseract installation and languages
RUN tesseract --version && \
    tesseract --list-langs && \
    echo "Available languages:" && \
    ls -la /usr/share/tesseract-ocr/*/tessdata/ || true

# Set up working directory
WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python3.11 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
ENV PATH="/app/venv/bin:$PATH"

# Create directories for data persistence
RUN mkdir -p /app/data/uploads /app/data/models /app/data/chroma_db /app/logs

# Expose Streamlit port
EXPOSE 8516

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8516/_stcore/health || exit 1

# Default command
CMD ["/app/venv/bin/python", "-m", "streamlit", "run", "app.py", "--server.port=8516", "--server.address=0.0.0.0", "--server.headless=true"]