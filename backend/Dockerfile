FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone ComfyUI (Pinned to v0.5.1 stable)
RUN git clone --branch v0.5.1 --single-branch https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /app/ComfyUI

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ComfyUI requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ComfyUI-GGUF custom node for quantized model support
WORKDIR /app/ComfyUI/custom_nodes
RUN git clone https://github.com/city96/ComfyUI-GGUF.git
RUN pip3 install --no-cache-dir -r ComfyUI-GGUF/requirements.txt

# Return to ComfyUI root
WORKDIR /app/ComfyUI

# Expose port
EXPOSE 8188

# Health check endpoint
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=30 \
    CMD curl -f http://localhost:8188/system_stats || exit 1

# Run ComfyUI in headless mode
# Note: --gpu-only causes OOM on 12GB, using default memory management
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
