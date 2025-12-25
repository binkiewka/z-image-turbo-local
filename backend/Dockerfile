FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including ffmpeg for video encoding
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
    ffmpeg \
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

# ============================================
# WAN 2.2 Video Generation Custom Nodes
# ============================================

# Video Helper Suite (VHS_VideoCombine for MP4 output)
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip3 install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

# Frame Interpolation (RIFE VFI for smoother video) - optional, no requirements.txt
RUN git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git

# ComfyUI-WanVideoWrapper (CRITICAL for 12GB VRAM - handles WAN 2.2 T2V/I2V with built-in block swap)
RUN git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    pip3 install --no-cache-dir -r ComfyUI-WanVideoWrapper/requirements.txt

# KJNodes (utility nodes including PathchSageAttentionKJ)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip3 install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt

# rgthree-comfy (Any Switch nodes for model routing)
RUN git clone https://github.com/rgthree/rgthree-comfy.git && \
    if [ -f rgthree-comfy/requirements.txt ]; then pip3 install --no-cache-dir -r rgthree-comfy/requirements.txt; fi

# ComfyUI-Various (JW nodes for image resize, math operations)
RUN git clone https://github.com/jamesWalker55/comfyui-various.git && \
    pip3 install --no-cache-dir soundfile

# ComfyUI-Custom-Scripts (MathExpression node)
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git

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
