#!/bin/bash

# Z-Image-Turbo Interactive Installer
# Usage: ./setup.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   Z-Image-Turbo & WAN 2.2 Setup Script      ${NC}"
echo -e "${BLUE}==============================================${NC}"

# Function to check command existence
check_cmd() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    else
        return 0
    fi
}

# 1. Prerequisites Check
echo -e "\n${YELLOW}[1/5] Checking Prerequisites...${NC}"

# Check for NVIDIA Drivers
if check_cmd nvidia-smi; then
    echo -e "${GREEN}✓ NVIDIA Drivers found${NC}"
else
    echo -e "${RED}✗ NVIDIA Drivers not found!${NC}"
    echo -e "  Please install NVIDIA drivers (version 550+) manually."
    read -p "  Continue anyway? (y/N) " continue_drivers
    if [[ ! "$continue_drivers" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Docker
if check_cmd docker; then
    echo -e "${GREEN}✓ Docker found${NC}"
else
    echo -e "${YELLOW}! Docker not found.${NC}"
    read -p "  Install Docker (requires sudo)? (y/N) " install_docker
    if [[ "$install_docker" =~ ^[Yy]$ ]]; then
        echo "Installing Docker..."
        # Add Docker's official GPG key
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg
        sudo install -m 0755 -d /etc/apt/keyrings
        if [ ! -f /etc/apt/keyrings/docker.gpg ]; then
             curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
             sudo chmod a+r /etc/apt/keyrings/docker.gpg
        fi

        # Add repository (assuming Ubuntu/Debian based, general enough for Mint/Ubuntu)
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
        echo -e "${GREEN}✓ Docker installed${NC}"
    else
         echo -e "${RED}Docker is required. Exiting.${NC}"
         exit 1
    fi
fi

# Check for NVIDIA Container Toolkit
if dpkg -l | grep -q nvidia-container-toolkit; then
     echo -e "${GREEN}✓ NVIDIA Container Toolkit found${NC}"
else
     echo -e "${YELLOW}! NVIDIA Container Toolkit not found.${NC}"
     read -p "  Install NVIDIA Container Toolkit (requires sudo)? (y/N) " install_ctk
     if [[ "$install_ctk" =~ ^[Yy]$ ]]; then
        echo "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
     else
        echo -e "${RED}NVIDIA Container Toolkit is required for GPU support. Exiting.${NC}"
        exit 1
     fi
fi

# 2. Directory Setup
echo -e "\n${YELLOW}[2/5] Creating Directory Structure...${NC}"
mkdir -p models/diffusion_models models/text_encoders models/vae models/loras models/upscaler models/vfi
echo -e "${GREEN}✓ Directories created${NC}"

# 3. Model Downloads
echo -e "\n${YELLOW}[3/5] Download Models...${NC}"

download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        if [ ! -s "$dest" ]; then
             echo "File $dest exists but is 0 bytes. Re-downloading..."
             rm "$dest"
        else
             echo -e "${GREEN}✓ $desc already exists${NC}"
             return
        fi
    fi

    echo "Downloading $desc..."
    wget -O "$dest" "$url"
}

# Image Gen Models (Required)
echo "--- Image Generation Models ---"
download_file "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf" "models/diffusion_models/z_image_turbo-Q8_0.gguf" "Z-Image-Turbo Model"
download_file "https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-IQ4_XS.gguf" "models/text_encoders/Qwen_3_4b-IQ4_XS.gguf" "Qwen Text Encoder"

# Flux VAE check
if [ ! -f "models/vae/ae.safetensors" ]; then
    echo -e "\n${YELLOW}![IMPORTANT] Flux VAE (ae.safetensors) is missing.${NC}"
    echo "This file requires a manual download from HuggingFace due to licensing."
    echo "1. Go to: https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors"
    echo "2. Download the file."
    echo "3. Place it in: $(pwd)/models/vae/"
    
    while true; do
        read -p "Have you placed 'ae.safetensors' in the models/vae directory? (y/n) " vae_confirm
        if [[ "$vae_confirm" =~ ^[Yy]$ ]]; then
            if [ -f "models/vae/ae.safetensors" ]; then
                echo -e "${GREEN}✓ Flux VAE found${NC}"
                break
            else
                echo -e "${RED}File still not found in models/vae/ae.safetensors${NC}"
            fi
        else
            echo "Skipping validation. Please remember to add it before running."
            break
        fi
    done
else
    echo -e "${GREEN}✓ Flux VAE already exists${NC}"
fi

# Video Gen Models (Optional)
echo -e "\n--- Video Generation Models (WAN 2.2) ---"
read -p "Do you want to download video generation models? (~17GB) (y/N) " download_video
if [[ "$download_video" =~ ^[Yy]$ ]]; then
    download_file "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf" "models/diffusion_models/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf" "WAN 2.2 High Noise"
    download_file "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/resolve/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf" "models/diffusion_models/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf" "WAN 2.2 Low Noise"
    download_file "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" "UMT5 Text Encoder"
    download_file "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" "models/vae/wan_2.1_vae.safetensors" "WAN VAE"
    # LoRAs
    download_file "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors" "High Noise LoRA"
    download_file "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" "models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors" "Low Noise LoRA"
fi

# Upscaler/VFI (Optional)
read -p "Do you want to download Upscaler and Frame Interpolation models? (~300MB) (y/N) " download_extras
if [[ "$download_extras" =~ ^[Yy]$ ]]; then
    download_file "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" "models/upscaler/RealESRGAN_x2.pth" "RealESRGAN x2"
    download_file "https://huggingface.co/hfmaster/models-moved/resolve/main/rife/rife49.pth" "models/vfi/rife49.pth" "RIFE 4.9"
fi

# 4. Docker Build & Run
echo -e "\n${YELLOW}[4/5] Building Docker Containers...${NC}"
read -p "Build and start Docker containers now? (y/N) " start_docker
if [[ "$start_docker" =~ ^[Yy]$ ]]; then
    echo "Building... (this may take a while)"
    if sudo docker compose build --no-cache; then
        echo -e "${GREEN}✓ Build successful${NC}"
        echo "Starting services..."
        sudo docker compose up -d
        echo -e "${GREEN}✓ Services started${NC}"
        echo -e "\n${GREEN}Setup Complete!${NC}"
        echo -e "Access the UI at: ${BLUE}http://localhost:7860${NC}"
    else
        echo -e "${RED}Docker build failed. Check logs.${NC}"
        exit 1
    fi
else
    echo "Skipping Docker build. You can run 'docker compose up -d --build' later."
fi

echo -e "\n${BLUE}Done.${NC}"
