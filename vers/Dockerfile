# ===========================================================
# VERS v3.0 — Multi-stage Docker Build
# ===========================================================
# Build:   docker build -t vers:3.0 .
# Run:     docker run -p 8501:8501 -p 8000:8000 vers:3.0
# ===========================================================

FROM python:3.9-slim AS builder

WORKDIR /app

# System dependencies for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi 'uvicorn[standard]' python-dotenv

# ---------------------------------------------------------------------------
FROM python:3.9-slim AS runtime

WORKDIR /app

# Copy system libs from builder
COPY --from=builder /usr/lib/ /usr/lib/
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p logs data models

# Expose ports: Streamlit (8501) + FastAPI (8000)
EXPOSE 8501 8000

# Environment
ENV PYTHONUNBUFFERED=1
ENV VERS_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch both services using a simple supervisor script
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & python -m streamlit run src/vers_dashboard.py --server.port=8501 --server.address=0.0.0.0"]
