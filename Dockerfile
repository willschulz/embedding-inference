FROM python:3.11-slim AS base

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY

WORKDIR /app

# System deps (minimal — CUDA comes via pip torch + nvidia-container-toolkit at runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.4 support
RUN pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download model weights into the image for instant cold starts
ARG MODEL_NAME=BAAI/bge-m3
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Copy application code
COPY app.py .

ENV MODEL_NAME=${MODEL_NAME}
EXPOSE 8001

CMD ["python", "app.py"]
