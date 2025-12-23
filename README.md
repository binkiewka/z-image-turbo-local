# Z-Image-Turbo Local Deployment

A Dockerized AI image generation system running [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) locally on consumer hardware (RTX 3060 12GB).

![Z-Image-Turbo](https://img.shields.io/badge/Model-Z--Image--Turbo-blue)
![GPU](https://img.shields.io/badge/GPU-RTX%203060%2012GB-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

## Features

- **Fast Generation**: ~3 seconds per image with 8-step distilled model
- **AI Prompt Enhancement**: Optional LLM-powered prompt expansion using Qwen2.5-1.5B (runs on CPU)
- **Local Deployment**: Complete privacy, runs entirely on your hardware
- **User-Friendly UI**: Clean Gradio interface with real-time progress
- **GGUF Quantization**: Optimized to fit in 12GB VRAM
- **Advanced Controls**: Negative prompts, adjustable steps (4-12), aspect ratios
- **Image History**: Gallery of last 12 generated images
- **One-Click Download**: Save images with metadata filenames

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Gradio)                                      │
│  Port: 7860                                             │
│  - User interface                                       │
│  - WebSocket client for progress tracking               │
│  - Prompt Enhancement (Qwen2.5-1.5B on CPU)             │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP + WebSocket
┌─────────────────────▼───────────────────────────────────┐
│  Backend (ComfyUI)                                      │
│  Port: 8188                                             │
│  - Headless inference engine                            │
│  - GGUF model support via ComfyUI-GGUF                  │
│  - CUDA 12.1 + PyTorch                                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│  Models (GGUF Quantized)                                │
│  - Z-Image-Turbo Q8_0 (7.2GB)                           │
│  - Qwen 3 4B Text Encoder IQ4_XS (2.3GB)                │
│  - Flux VAE (335MB)                                     │
│  - Qwen2.5-1.5B-Instruct Q4 (~1GB, auto-downloaded)     │
└─────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware

- **GPU**: NVIDIA RTX 3060 12GB (or similar with 12GB+ VRAM)
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 15GB+ SSD space for models and Docker images
- **OS**: Linux (tested on Linux Mint 21.3 / Ubuntu 22.04) or Windows with WSL2 (see [WSL2 Setup Guide](docs/setup-guides/Windows-WSL2.md))

### Software

- Docker Engine (v24.0+)
- Docker Compose (v2.20+)
- NVIDIA Driver 550+ (CUDA 12.1 compatible)
- NVIDIA Container Toolkit

## Installation

### 1. Install NVIDIA Drivers

```bash
# Check current driver
nvidia-smi

# If needed, install driver 550+
sudo apt install nvidia-driver-550
sudo reboot
```

### 2. Install Docker

```bash
# Add Docker's official GPG key
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3. Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Clone Repository

```bash
git clone https://github.com/binkiewka/z-image-turbo-local.git
cd z-image-turbo-local
```

### 5. Download Models

Create model directories and download the required files:

```bash
# Create model directories
mkdir -p models/diffusion_models models/text_encoders models/vae

# Z-Image-Turbo diffusion model (7.2GB)
wget -O models/diffusion_models/z_image_turbo-Q8_0.gguf \
  "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf"

# Qwen 3 4B text encoder (2.3GB)
wget -O models/text_encoders/Qwen_3_4b-IQ4_XS.gguf \
  "https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-IQ4_XS.gguf"
```

**Flux VAE (335MB) - Manual Download Required:**

The VAE model requires accepting a license on HuggingFace:

1. Visit [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
2. Log in to HuggingFace and click "Agree and access repository"
3. Download [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors) (click the download ↓ button)
4. Move it to the models folder:

   ```bash
   mv ~/Downloads/ae.safetensors models/vae/
   ```

**Verify downloads (important!):**

```bash
ls -lh models/diffusion_models/  # Should show ~7.2GB
ls -lh models/text_encoders/      # Should show ~2.3GB
ls -lh models/vae/                 # Should show ~335MB (NOT 0 bytes!)
```

> ⚠️ If any file shows **0 bytes**, delete it and re-download. This is a common issue that causes startup failures.

### 6. Build and Run

```bash
# Build containers
docker compose build --no-cache

# Start services
docker compose up -d

# View logs
docker compose logs -f
```

## Platform Support

This project officially supports:

- **Linux** (Ubuntu 22.04, Linux Mint 21.3, or similar): Follow the [Installation](#installation) section above
- **Windows with WSL2**: Follow the comprehensive [Windows WSL2 Setup Guide](docs/setup-guides/Windows-WSL2.md)
  - Includes step-by-step WSL2 installation
  - NVIDIA GPU passthrough configuration
  - Docker setup (both Docker CE and Docker Desktop options)
  - WSL2-specific troubleshooting and performance optimization

## Usage

1. **Access the UI**: Open `http://localhost:7860` in your browser
2. **Enter a prompt**: Describe the image you want (e.g., "a cat wearing a wizard hat")
3. **Optional settings**:
   - **✨ Enhance Prompt**: Enable AI-powered prompt expansion (~5-10s, downloads model on first use)
   - **Negative Prompt**: Things to avoid (e.g., "blurry, ugly, distorted")
   - **Seed**: Use -1 for random, or a specific number for reproducibility
   - **Steps**: 4-12 (default 8) - more steps = better quality but slower
   - **Aspect Ratio**: Choose from 1:1, 3:4, 4:3, 16:9, 9:16
4. **Generate**: Click the button and wait ~3 seconds (or ~10-15s with enhancement)
5. **Download**: Click the download link to save the image

## Configuration

### Ports

- **Frontend**: `7860` (Gradio UI)
- **Backend**: `8188` (ComfyUI API)

To change ports, edit `docker-compose.yml`:

```yaml
ports:
  - "0.0.0.0:7860:7860"  # Change first 7860 to desired port
```

### VRAM Management

The system uses ~11.4GB VRAM with default settings. If you experience OOM errors:

1. **Reduce resolution**: Use lower aspect ratios (720p instead of 1024p)
2. **Use fewer steps**: Try 4-6 steps instead of 8
3. **Use lighter text encoder**: Download `Qwen_3_4b-Q3_K_M.gguf` (2GB instead of 2.3GB)

## Troubleshooting

### "CUDA Out of Memory"

- **Cause**: Resolution too high or other GPU processes running
- **Fix**: Close other GPU apps, use lower resolution, reduce steps to 4-6

### "Models not loading"

- **Cause**: Model files not downloaded or in wrong folder
- **Fix**: Verify files exist in `models/diffusion_models/`, `models/text_encoders/`, `models/vae/`

### "WebSocket connection failed"

- **Cause**: Backend still starting up
- **Fix**: Wait 30-60 seconds, check logs with `docker compose logs backend`

### "Generation is slow (30s+)"

- **Cause**: Models loading from disk on each request
- **Fix**: After first generation, subsequent ones should be ~3s. If not, check if GPU is being used:

  ```bash
  docker exec z-image-backend nvidia-smi
  ```

### "Invalid prompt validation error"

- **Cause**: Model files not found or incompatible versions
- **Fix**: Verify exact filenames match:
  - `z_image_turbo-Q8_0.gguf`
  - `Qwen_3_4b-IQ4_XS.gguf`
  - `ae.safetensors`

## Development

### Rebuild after changes

```bash
# Rebuild everything
docker compose down -v && docker compose build --no-cache && docker compose up

# Rebuild only frontend
docker compose build --no-cache frontend && docker compose up -d

# Rebuild only backend
docker compose build --no-cache comfyui && docker compose up -d
```

### View logs

```bash
docker compose logs -f           # All services
docker compose logs -f frontend  # Frontend only
docker compose logs -f comfyui   # Backend only
```

## Performance

| Metric | Value |
|--------|-------|
| First generation | ~10-30s (model loading) |
| Subsequent generations | ~3s |
| VRAM usage | ~11.4GB / 12GB |
| Resolution | Up to 1024x1024 |
| Steps | 4-12 (default 8) |

## Credits

- **Z-Image-Turbo**: [Tongyi-MAI/Alibaba](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)
- **ComfyUI-GGUF**: [city96](https://github.com/city96/ComfyUI-GGUF)
- **Flux VAE**: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- **Qwen 3**: [Qwen Team](https://github.com/QwenLM/Qwen)

## License

This project is for personal use. Model licenses:

- Z-Image-Turbo: Apache 2.0
- FLUX.1-schnell: Apache 2.0
- ComfyUI: GPL-3.0

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Review Docker logs: `docker compose logs -f`
3. Verify GPU access: `docker exec z-image-backend nvidia-smi`
