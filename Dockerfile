# ============================================================================
# ChestVision AI — Hugging Face Spaces Docker Deployment
# Optimized for CPU-only PyTorch inference
# ============================================================================
FROM python:3.11-slim

# System dependencies for Pillow (image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (Docker layer caching)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code and assets
COPY app.py .
COPY model.py .
COPY chest_disease_model.pth .
COPY templates/ templates/
COPY static/ static/

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Launch FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
