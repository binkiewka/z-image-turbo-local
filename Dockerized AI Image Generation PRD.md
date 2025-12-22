# **Product Requirements Document: Local Deployment of Z-Image-Turbo Generative AI System on Docker**

## **1\. Executive Summary**

### **1.1 Project Overview**

This Product Requirements Document (PRD) establishes the comprehensive technical framework for the deployment of a local, high-performance Artificial Intelligence image generation system. The initiative focuses on the implementation of **Z-Image-Turbo**, a state-of-the-art diffusion model developed by Tongyi-MAI (Alibaba), which utilizes a distilled Scalable Single-Stream Diffusion Transformer (S3-DiT) architecture to achieve rapid inference speeds.

The project is constrained by specific hardware and software parameters: the host environment is a **Linux Mint** workstation powered by an **NVIDIA GeForce RTX 3060** with **12GB of VRAM**. The deployment strategy mandates the use of **Docker** containerization to ensure environment isolation, reproducibility, and modularity. Furthermore, the system requires a decoupled architecture wherein a dedicated, user-friendly web interface interacts with the backend inference engine to facilitate prompt entry and image visualization, abstracting the complexity of the underlying model graph.

### **1.2 Strategic Objectives**

The primary objective is to engineer a system that balances the competing demands of high-fidelity model performance and restricted hardware resources. While Z-Image-Turbo is optimized for speed (utilizing an 8-step adversarial distillation process), its parameter count (6 billion) and reliance on a large language model (LLM) text encoder (Qwen-3 4B) present significant memory management challenges for a 12GB VRAM environment.

Therefore, the strategic goals are defined as follows:

* **Operational Stability:** Eliminate Out-Of-Memory (OOM) failures through the strategic application of GGUF quantization and rigorous memory management protocols within the ComfyUI backend.  
* **Inference Latency:** Achieve near real-time generation speeds (targeting \<4 seconds per image) by leveraging the "Turbo" distillation capabilities and ensuring the GPU remains the primary compute device without fallback to system RAM.  
* **User Experience (UX):** Deliver a streamlined, "black-box" interface for the end-user that hides the node-based complexity of the inference engine while retaining the powerful prompting capabilities of the Qwen encoder.  
* **Deployability:** Provide a fully containerized docker-compose stack that allows for "one-click" deployment on the target Linux Mint OS, handling all CUDA runtime dependencies transparently.

### **1.3 Scope of Work**

This document covers the entire lifecycle of the deployment, including:

* **Host OS Preparation:** Configuration of Linux Mint kernel and NVIDIA proprietary drivers.  
* **Container Orchestration:** Design of the Docker Compose stack utilizing the NVIDIA Container Toolkit.  
* **Model Engineering:** Selection and configuration of quantized model weights (GGUF format) to fit the hardware budget.  
* **Backend Configuration:** Setup of ComfyUI, installation of custom nodes, and API exposure.  
* **Frontend Development:** Construction of a Python-based web client (Gradio) to interface with the backend API.  
* **Performance Tuning:** Optimization of sampler settings and batch sizes for the RTX 3060\.

## ---

**2\. Background and Technology Landscape**

To understand the specific requirements of deploying Z-Image-Turbo, it is necessary to contextualize it within the broader landscape of generative AI and the evolution of diffusion architectures. This context informs the architectural decisions made later in this document regarding hardware utilization and software stack selection.

### **2.1 Evolution of Diffusion Architectures**

The field of text-to-image generation has evolved rapidly from the early days of Generative Adversarial Networks (GANs) to Latent Diffusion Models (LDMs).

#### **2.1.1 The UNet Era (Stable Diffusion 1.5/XL)**

Earlier models like Stable Diffusion 1.5 and SDXL relied on a **UNet** backbone. These architectures process the noised latent image through downsampling and upsampling blocks, injecting text conditioning via cross-attention layers. While effective, UNets often struggle with complex spatial relationships and dense text comprehension. They typically operate as a "dual-stream" system where the text encoder (CLIP) and the image generation model (UNet) operate somewhat independently, meeting only at specific attention mechanisms.

#### **2.1.2 The Transformer Era (DiT)**

The industry has shifted toward **Diffusion Transformers (DiT)**, pioneered by models like Sora and consolidated in image generation by Flux and Z-Image. DiT architectures replace the UNet with a standard Transformer backbone, treating image patches as tokens similar to words in a sentence. This allows for significantly better scaling laws—adding more parameters consistently improves performance.

**Z-Image-Turbo** utilizes a specific variant called **S3-DiT (Scalable Single-Stream Diffusion Transformer)**. In this architecture, text tokens (from the prompt) and visual tokens (from the noised image) are concatenated into a *single* sequence and processed by the same Transformer blocks.1 This "single-stream" approach allows for dense, bidirectional interaction between text and image information at every layer of the network, resulting in superior prompt adherence and the ability to render legible text within images—a key differentiator for Z-Image.2

### **2.2 The Significance of "Turbo" and Distillation**

Standard diffusion models typically require 20 to 50 denoising steps to resolve a coherent image from random noise. This results in generation times of 10–30 seconds on consumer hardware.

"Turbo" models, including Z-Image-Turbo, employ **Adversarial Diffusion Distillation (ADD)** or similar distillation techniques (like Decoupled-DMD in Z-Image's case 2). This process involves training a "student" model to mimic the output of a larger "teacher" model in significantly fewer steps. Z-Image-Turbo is optimized for just **8 steps**.2

For the user with an RTX 3060, this is critical. It effectively quadruples the throughput compared to a standard 30-step model. However, it imposes strict constraints on the Sampler and Scheduler settings in the backend workflow, which must be hardcoded to prevent the user from selecting incompatible parameters that would degrade image quality.

### **2.3 Hardware Context: The RTX 3060 12GB**

The NVIDIA GeForce RTX 3060 12GB is a unique card in the deep learning market.

* **VRAM Advantage:** It possesses 12GB of GDDR6 memory, which is more than the RTX 3070 (8GB) and RTX 3080 (10GB). This makes it surprisingly capable for running larger models that simply crash on more powerful cards due to memory limits.  
* **Bandwidth Bottleneck:** However, it utilizes a 192-bit memory bus, limiting its memory bandwidth compared to the 256-bit or 384-bit buses of higher-tier cards.  
* **Implication for PRD:** The system must prioritize **memory capacity** over raw compute power. We can fit the model, but moving large tensors in and out of VRAM will be slow. Therefore, the design must ensure the entire model pipeline stays resident in VRAM. Swapping to system RAM (offloading) will hit the bandwidth bottleneck hard, increasing generation time from \~3 seconds to \~30+ seconds. This necessitates the use of quantization (GGUF) to shrink the model footprint below the 12GB physical limit.

## ---

**3\. Technical Requirements & Hardware Constraints**

This section defines the rigid technical boundaries within which the system must operate.

### **3.1 Host System Specifications**

* **Operating System:** Linux Mint 21.3 (Code name: Virginia).  
  * *Basis:* Based on Ubuntu 22.04 LTS (Jammy Jellyfish).  
  * *Kernel:* Linux 5.15 (standard) or 6.5 (Edge). Both are compatible with the required NVIDIA drivers.  
* **CPU:** Architecture x86\_64 (Assume modern multi-core, e.g., Ryzen 5 or Intel i5+).  
* **RAM:** Minimum 16GB System RAM (32GB Recommended).  
  * *Relevance:* Docker containers and model loading require system RAM buffering before transferring to GPU.  
* **Storage:** SSD required. Model weights (even quantized) are large (10GB+ total). Mechanical HDDs will introduce unacceptable latency during container startup and model switching.

### **3.2 GPU Specifications & Constraints**

* **Model:** NVIDIA GeForce RTX 3060\.  
* **VRAM:** 12,288 MB (12 GB) GDDR6.  
* **Compute Capability:** 8.6 (Ampere Architecture).  
* **Driver Requirement:** Proprietary NVIDIA Driver version **535.x** or higher (Recommended: **550.x**).  
  * *Reasoning:* CUDA 12.1 support requires driver version \>= 530\. The ComfyUI backend will utilize PyTorch with CUDA 12.1.  
* **Constraint Analysis:**  
  * **Z-Image-Turbo (BF16):** \~11.5 GB.  
  * **Qwen-3 4B Text Encoder (FP16):** \~7.8 GB.  
  * **VAE (FP16):** \~335 MB.  
  * **Runtime Overhead (CUDA Context \+ Activations):** \~2-3 GB.  
  * **Total (Unoptimized):** \~21 GB.  
  * **Conclusion:** Running the native BF16 model is **impossible** on this hardware without severe offloading. **GGUF Quantization is a mandatory requirement.**

### **3.3 Software Stack Requirements**

* **Container Engine:** Docker Engine Community Edition (v24.0 or newer).  
* **Orchestration:** Docker Compose (v2.20 or newer).  
* **GPU Runtime:** NVIDIA Container Toolkit (v1.14 or newer).  
  * *Function:* Allows Docker containers to access /dev/nvidia\* devices and injects driver libraries at runtime.  
* **Backend Framework:** ComfyUI (Latest release).  
  * *Justification:* Only ComfyUI currently supports the flexibility required to load GGUF versions of Z-Image via custom nodes (ComfyUI-GGUF).6  
* **Frontend Framework:** Python 3.10+ with Gradio.  
  * *Justification:* Gradio provides pre-built components for image display and text input, minimizing development time and Docker image size.

## ---

**4\. Model Architecture & Data Strategy**

To satisfy the VRAM constraints, we must employ a specific data strategy involving model quantization. This section details the exact artifacts required.

### **4.1 The Z-Image-Turbo Model (6B)**

The core generation model is a 6-billion parameter transformer. In its native bfloat16 format, it offers the highest fidelity but is too large. We will utilize the **GGUF (GPT-Generated Unified Format)** version. GGUF was originally designed for LLMs (like Llama) but has been adapted for Diffusion models to allow efficient packing of weights into 8-bit, 5-bit, or 4-bit integers.

* **Selected Format:** GGUF.  
* **Target Quantization:** Q8\_0 (8-bit quantization).  
  * *Size:* Approx. **6.4 GB**.7  
  * *Quality:* Near-lossless compared to BF16.  
  * *Alternative:* Q5\_K\_M (Approx 4.3 GB) if the user wishes to run background tasks, but Q8\_0 is preferred for quality.

### **4.2 The Qwen-3 4B Text Encoder**

Z-Image departs from the standard CLIP or T5 encoders used in Stable Diffusion. It uses **Qwen-3**, a highly capable Large Language Model, to interpret prompts.8 This is the source of its strong bilingual (English/Chinese) performance and complex instruction following.

* **Constraint:** A 4B LLM in FP16 is \~8GB.  
* **Selected Format:** GGUF.  
* **Target Quantization:** Q4\_K\_M (4-bit quantization).  
  * *Size:* Approx. **2.6 GB**.  
  * *Reasoning:* LLMs are highly resilient to 4-bit quantization. The perplexity loss is negligible for the purpose of image conditioning.

### **4.3 The Variational Autoencoder (VAE)**

The VAE is responsible for compressing the pixel-space image into latent space and decoding it back. Z-Image uses the standard Flux VAE.

* **Selected Format:** Safetensors (Standard).  
* **Size:** **335 MB**.  
* **Reasoning:** The VAE is small enough that quantization risks introducing visual artifacts (pixelation, color shifting) for minimal memory gain.

### **4.4 Total VRAM Budget Calculation (Optimized)**

With the selected GGUF strategy, the VRAM usage is projected as follows:

* **Model (Q8\_0):** 6.4 GB  
* **Encoder (Q4\_K\_M):** 2.6 GB  
* **VAE (FP16):** 0.4 GB  
* **CUDA Context:** 0.8 GB  
* **Inference Buffer (1024x1024):** \~1.5 GB  
* **Total Estimated Load:** **\~11.7 GB**  
* **Result:** Fits within the 12GB physical limit of the RTX 3060, ensuring high-speed GPU-resident inference.

## ---

**5\. System Architecture Design**

The system is designed as a **Microservices Architecture** utilizing Docker Compose. This ensures that the frontend and backend are loosely coupled, allowing independent updates and failure isolation.

### **5.1 Service 1: The Inference Engine (backend)**

This container is the heavy lifter. It encapsulates the ComfyUI runtime and the model weights.

* **Base Image:** nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04.  
  * *Why:* Provides the necessary CUDA libraries compatible with PyTorch 2.x.  
* **Application:** ComfyUI (Headless Mode).  
* **Custom Nodes:**  
  * ComfyUI-GGUF: Essential for loading the .gguf model files.7  
  * ComfyUI-Z-Image-Utilities (Optional): For advanced prompt enhancement features.9  
* **Volumes:**  
  * /app/ComfyUI/models: Mapped to host storage to persist large model files.  
  * /app/ComfyUI/output: Mapped to host to save generated images.  
* **Networking:** Exposes port 8188 to the Docker internal network (and optionally to localhost for debugging).

### **5.2 Service 2: The User Interface (frontend)**

This container provides the "simple webpage" requested. It acts as an API client to the backend.

* **Base Image:** python:3.10-slim.  
  * *Why:* Lightweight, minimal attack surface.  
* **Framework:** Gradio.  
* **Logic:**  
  * Loads a specific JSON workflow template (exported from ComfyUI API).  
  * Accepts user input (Prompt, Seed, Aspect Ratio).  
  * Injects input into the JSON template.  
  * Sends JSON to http://backend:8188/prompt.  
  * Establishes WebSocket connection to ws://backend:8188/ws to listen for execution status.  
  * Fetches the resulting image via HTTP GET.  
* **Networking:** Exposes port 7860 to the host machine for user access via browser.

## ---

**6\. Implementation Specifications: Backend**

This section details the specific configuration of the ComfyUI backend.

### **6.1 Dockerfile Specification**

The backend Dockerfile must handle the installation of system dependencies (git, pip), the cloning of the ComfyUI repository, and the installation of the specific custom nodes required for Z-Image GGUF support.

**Crucial Implementation Detail:** The ComfyUI-GGUF node repo must be cloned into the custom\_nodes directory during the build process to ensure the container is ready upon first startup.

Dockerfile

\# /backend/Dockerfile  
FROM nvidia/cuda:12.1.1\-cudnn8-runtime-ubuntu22.04

ENV DEBIAN\_FRONTEND=noninteractive  
ENV PYTHONUNBUFFERED=1

\# Install System Dependencies  
RUN apt-get update && apt-get install \-y \\  
    python3 python3-pip git wget \\  
    libgl1-mesa-glx libglib2.0-0 \\  
    && rm \-rf /var/lib/apt/lists/\*

WORKDIR /app

\# Clone ComfyUI  
RUN git clone https://github.com/comfyanonymous/ComfyUI.git  
WORKDIR /app/ComfyUI

\# Install Python Dependencies (PyTorch with CUDA 12.1)  
RUN pip3 install \--no-cache-dir torch torchvision torchaudio \--index-url https://download.pytorch.org/whl/cu121  
RUN pip3 install \--no-cache-dir \-r requirements.txt

\# Install Custom Nodes: ComfyUI-GGUF  
WORKDIR /app/ComfyUI/custom\_nodes  
RUN git clone https://github.com/city96/ComfyUI-GGUF.git  
RUN pip3 install \--no-cache-dir \-r ComfyUI-GGUF/requirements.txt

\# Return to root  
WORKDIR /app/ComfyUI

\# Entrypoint  
CMD \["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"\]

### **6.2 Workflow Design (The "API Graph")**

ComfyUI executes directed acyclic graphs (DAGs). The frontend needs a template of this graph to submit jobs. The developer must manually create this graph once in ComfyUI, export it, and save it as workflow\_api.json.

**Node Configuration for Z-Image-Turbo GGUF:**

1. **Node: UnetLoaderGGUF**  
   * *Input:* unet\_name \-\> z\_image\_turbo-Q8\_0.gguf  
   * *Output:* MODEL  
2. **Node: DualCLIPLoader (or CLIPLoaderGGUF)**  
   * *Input:* clip\_name \-\> Qwen3-4B-Instruct-Q4\_K\_M.gguf  
   * *Output:* CLIP  
   * *Note:* Z-Image uses the S3-DiT architecture where text and image tokens are processed together. The CLIP loader must be compatible with the specific tokenization required by Qwen.  
3. **Node: CLIPTextEncode**  
   * *Input:* Text (User Prompt), CLIP (from Loader).  
   * *Output:* CONDITIONING (Positive/Negative).  
   * *Note:* Z-Image-Turbo is a distilled model and generally does not use negative prompts (or uses empty ones). The prompt text is the primary input.  
4. **Node: KSampler**  
   * *Steps:* **8** (Fixed).  
   * *CFG:* **1.0** (Fixed).  
   * *Sampler:* euler.  
   * *Scheduler:* simple or sgm\_uniform.  
   * *Denoiser:* 1.0.  
5. **Node: VAEDecode**  
   * *Input:* Samples (from KSampler), VAE (from VAELoader).  
   * *Output:* IMAGE.

The frontend will perform string replacement on this JSON structure, specifically targeting the inputs.text field of the CLIPTextEncode node and the inputs.seed field of the KSampler node.

## ---

**7\. Implementation Specifications: Frontend**

This section details the web application logic.

### **7.1 Framework Selection: Gradio**

Gradio is the industry standard for rapid prototyping of ML interfaces. It handles the UI rendering, event loop, and state management with minimal code. For this project, Gradio serves as the presentation layer, completely abstracting the WebSocket communication required by ComfyUI.

### **7.2 Application Logic (app.py)**

The Python script must handle the lifecycle of a generation request:

1. **Connection:** Establish a WebSocket connection to the backend service (hostname: comfyui).  
2. **Payload Construction:** Load the workflow\_api.json template. Replace the placeholder prompt text with the user's input. Generate a random seed (or use user input).  
3. **Submission:** HTTP POST the JSON payload to /prompt. Receive a prompt\_id.  
4. **Monitoring:** Listen on the WebSocket. ComfyUI broadcasts status updates. The client filters for execution\_success messages matching the prompt\_id.  
5. **Retrieval:** The success message contains the filename of the generated image. The client performs an HTTP GET to /view to download the image byte stream.  
6. **Display:** The image bytes are returned to the Gradio interface for rendering.

### **7.3 User Interface Design**

The interface should be kept intentionally simple to meet the "webpage to prompt and view" requirement.

* **Title:** "Z-Image-Turbo (Local)"  
* **Input Area:**  
  * **Prompt:** Text Area (Lines: 4). Placeholder: "Describe the image..."  
  * **Seed:** Number Input (Default: \-1 for random).  
  * **Generate Button:** Primary Action.  
* **Output Area:**  
  * **Image Display:** Large viewport.  
  * **Status Bar:** Shows "Queueing", "Processing", "Done".

## ---

**8\. Installation and Configuration Guide (Linux Mint)**

This section provides the step-by-step procedure to deploy the system on the target machine.

### **8.1 Phase 1: Host Preparation**

Linux Mint 21 does not include the NVIDIA Container Toolkit by default.

Step 1: Install NVIDIA Drivers  
Use the Driver Manager to install nvidia-driver-550. Verify with:

Bash

nvidia-smi

Step 2: Install Docker Engine  
Follow the Ubuntu Jammy installation path.

Bash

\# Add Docker's official GPG key:  
sudo apt-get update  
sudo apt-get install ca-certificates curl gnupg  
sudo install \-m 0755 \-d /etc/apt/keyrings  
curl \-fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg \--dearmor \-o /etc/apt/keyrings/docker.gpg  
sudo chmod a+r /etc/apt/keyrings/docker.gpg

\# Add the repository to Apt sources:  
echo \\  
  "deb \[arch="$(dpkg \--print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg\] https://download.docker.com/linux/ubuntu \\  
  jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list \> /dev/null  
sudo apt-get update  
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

Step 3: Install NVIDIA Container Toolkit  
This enables the runtime: nvidia capability in Docker Compose.

Bash

curl \-fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg \--dearmor \-o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \\  
  && curl \-s \-L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\  
  sed 's\#deb https://\#deb \[signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg\] https://\#g' | \\  
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update  
sudo apt-get install \-y nvidia-container-toolkit  
sudo nvidia-ctk runtime configure \--runtime=docker  
sudo systemctl restart docker

### **8.2 Phase 2: Project Setup & Model Acquisition**

Create the directory structure.

Bash

mkdir \-p z-image-local/models/{unet,clip,vae}  
mkdir \-p z-image-local/frontend  
mkdir \-p z-image-local/backend  
mkdir \-p z-image-local/output

Model Downloads (Manual Step):  
The developer must download the models to z-image-local/models/:

1. **UNET:** Download z\_image\_turbo-Q8\_0.gguf from HuggingFace (jayn7/Z-Image-Turbo-GGUF) \-\> place in models/unet/.  
2. **CLIP:** Download Qwen3-4B-Instruct-Q4\_K\_M.gguf \-\> place in models/clip/.  
3. **VAE:** Download ae.safetensors \-\> place in models/vae/.

### **8.3 Phase 3: Docker Composition**

Create docker-compose.yml in the root.

YAML

version: '3.8'

services:  
  comfyui:  
    build:./backend  
    container\_name: z-image-backend  
    restart: unless-stopped  
    runtime: nvidia  
    environment:  
      \- NVIDIA\_VISIBLE\_DEVICES=all  
      \- NVIDIA\_DRIVER\_CAPABILITIES=compute,utility  
      \- CLI\_ARGS=--listen 0.0.0.0 \--port 8188 \--preview-method auto  
    volumes:  
      \# Map model directories individually to the correct ComfyUI paths  
      \-./models/unet:/app/ComfyUI/models/unet  
      \-./models/clip:/app/ComfyUI/models/clip  
      \-./models/vae:/app/ComfyUI/models/vae  
      \-./output:/app/ComfyUI/output  
    ports:  
      \- "8188:8188"  
    deploy:  
      resources:  
        reservations:  
          devices:  
            \- driver: nvidia  
              count: 1  
              capabilities: \[gpu\]

  frontend:  
    build:./frontend  
    container\_name: z-image-frontend  
    restart: unless-stopped  
    ports:  
      \- "7860:7860"  
    environment:  
      \- COMFYUI\_HOST=comfyui  
      \- COMFYUI\_PORT=8188  
    depends\_on:  
      \- comfyui

### **8.4 Phase 4: Initialization**

1. **Build the Stack:** docker-compose up \--build \-d  
2. **Verification:** Check logs with docker-compose logs \-f.  
   * *Success Indicator:* ComfyUI logs should show "Device: cuda:0 NVIDIA GeForce RTX 3060" and "Loaded custom nodes: ComfyUI-GGUF".  
3. **Usage:** Open a web browser on the Linux host and navigate to http://localhost:7860.

## ---

**9\. Performance Analysis & Benchmarking**

### **9.1 Theoretical Throughput**

The system is architected to optimize the 8-step inference of Z-Image-Turbo.

* **Model Loading:** The first generation will incur a delay of 5-10 seconds as the GGUF weights are loaded from disk into VRAM.  
* **Inference (Warm):**  
  * The RTX 3060 has 3584 CUDA cores.  
  * At 8 steps, inference time is calculated as $T\_{total} \= T\_{encode} \+ (8 \\times T\_{step}) \+ T\_{decode}$.  
  * Estimations:  
    * Text Encode (Qwen 4B Q4): \~0.5s.  
    * Diffusion (8 steps Q8): \~2.4s (approx 0.3s/step).  
    * Decode (VAE): \~0.2s.  
  * **Total Latency:** \~3.1 seconds per image.

This meets the requirement for a "Turbo" experience.

### **9.2 VRAM Profiling**

With the GGUF strategy:

* **Idle (Desktop):** \~0.5 GB used by Linux Mint GUI.  
* **Model Loaded (Static):** \~9.5 GB used.  
* **Peak during Inference:** \~11.0 \- 11.5 GB.  
* **Margin:** \~0.5 GB free.  
* *Observation:* This is a tight fit. The user should avoid running other VRAM-heavy applications (like high-res YouTube videos or gaming) simultaneously with the Docker stack.

## ---

**10\. Operational Considerations and Troubleshooting**

### **10.1 Updates and Maintenance**

Since the backend is built from the master branch of ComfyUI in the Dockerfile, updates are handled by rebuilding the container.

* **Update Command:** docker-compose build \--no-cache comfyui && docker-compose up \-d  
* **Frequency:** Recommend updating monthly, or when new custom node features are released.

### **10.2 Common Failure Modes**

* **"CUDA Out of Memory"**:  
  * *Cause:* User inputs an excessively high resolution (e.g., \>1024x1024) or batch size \> 1\.  
  * *Fix:* The Frontend logic limits default resolution to 1024x1024. If the user overrides this, the backend will fail.  
* **"WebSocket Error" / "Connection Refused"**:  
  * *Cause:* ComfyUI container hasn't finished booting before the Frontend tries to connect.  
  * *Fix:* The depends\_on directive in Docker Compose handles startup order, but application-level retries in app.py are necessary for robustness.  
* **Slow Generation (Minutes instead of Seconds)**:  
  * *Cause:* Docker is not using the GPU (fallback to CPU).  
  * *Fix:* Verify nvidia-smi inside the container. Reinstall NVIDIA Container Toolkit on the host.

### **10.3 Security**

The system is designed for **local use only**.

* **Ports:** Port 8188 and 7860 are exposed. If the Linux machine is directly connected to the internet without a firewall, these interfaces are public.  
* **Recommendation:** Use ufw (Uncomplicated Firewall) on Linux Mint to restrict access to 127.0.0.1 if the machine is on a shared network.  
  Bash  
  sudo ufw allow from 127.0.0.1 to any port 7860  
  sudo ufw enable

## ---

**11\. Conclusion**

This Product Requirements Document outlines a fully realized, high-performance, and resource-optimized solution for running Z-Image-Turbo locally. By recognizing the constraints of the RTX 3060 12GB—specifically the memory bandwidth and capacity limits—and applying a strict GGUF quantization strategy, the system avoids the common pitfalls of deploying large S3-DiT models on consumer hardware.

The architecture leverages Docker for reproducibility, ensuring that the complex interplay of CUDA libraries, Python dependencies, and custom nodes works identically on deployment as it does in development. The decoupled Frontend/Backend design satisfies the user's request for a simple "prompt and view" experience while maintaining the raw power of ComfyUI under the hood. This system represents the optimal balance of speed, quality, and usability for the specified hardware profile.

## ---

**12\. Appendices: Data Tables**

### **12.1 Hardware Compatibility Matrix**

| Hardware Component | Specification | Compatibility Status | Notes |
| :---- | :---- | :---- | :---- |
| GPU | RTX 3060 12GB | **Verified** | Requires GGUF Quantization (Q8/Q4) to fit VRAM. |
| OS | Linux Mint 21.3 | **Verified** | Requires proprietary NVIDIA driver 550+ and Container Toolkit. |
| RAM | 16GB+ System | **Required** | Model loading buffers through system RAM. |
| Storage | SSD | **Critical** | HDD will cause timeouts during model loading. |

### **12.2 Model Configuration Table**

| Model Component | Filename | Format | Quantization | Size (Approx) |
| :---- | :---- | :---- | :---- | :---- |
| **Diffusion Model** | z\_image\_turbo-Q8\_0.gguf | GGUF | Q8\_0 (8-bit) | 6.4 GB |
| **Text Encoder** | Qwen3-4B-Instruct-Q4\_K\_M.gguf | GGUF | Q4\_K\_M (4-bit) | 2.6 GB |
| **VAE** | ae.safetensors | Safetensors | FP16 | 0.3 GB |
| **Total** |  |  |  | **\~9.3 GB** |

### **12.3 API Endpoint Reference (Backend)**

| Method | Endpoint | Payload | Description |
| :---- | :---- | :---- | :---- |
| POST | /prompt | {"prompt": \<json\_graph\>, "client\_id": \<uuid\>} | Queues a workflow execution. Returns prompt\_id. |
| WS | /ws | ?clientId=\<uuid\> | WebSocket stream for status updates (executing, progress). |
| GET | /view | ?filename=\<name\>\&type=output | Retrieves the generated image binary. |
| POST | /upload/image | Multipart Form Data | Uploads input images (for img2img workflows). |

### **12.4 Frontend Feature List**

| Feature | Description | Priority |
| :---- | :---- | :---- |
| **Prompt Input** | Multi-line text area for entering descriptive prompts. | P0 (Critical) |
| **Image Viewer** | High-resolution display of the generated result. | P0 (Critical) |
| **Seed Control** | Option to randomize or fix the seed for reproducibility. | P1 (High) |
| **Aspect Ratio** | Dropdown to select 1:1, 3:4, or 16:9 ratios. | P1 (High) |
| **Progress Bar** | Visual indicator of generation status (Queue/Processing). | P2 (Medium) |
| **History Gallery** | Thumbnails of the last 5 generated images. | P2 (Medium) |

#### **Works cited**

1. Reviewing the Best New Image Generator Models \- DigitalOcean, accessed on December 22, 2025, [https://www.digitalocean.com/community/tutorials/image-generation-model-review](https://www.digitalocean.com/community/tutorials/image-generation-model-review)  
2. Z-Image Turbo vs Z-Image: A Comprehensive Comparison \- Fal.ai, accessed on December 22, 2025, [https://fal.ai/learn/devs/z-image-turbo-vs-z-image-comparison](https://fal.ai/learn/devs/z-image-turbo-vs-z-image-comparison)  
3. Z-Image-Turbo AI Image Generator \- Runware, accessed on December 22, 2025, [https://runware.ai/models/z-image-turbo](https://runware.ai/models/z-image-turbo)  
4. Tongyi-MAI/Z-Image-Turbo \- Hugging Face, accessed on December 22, 2025, [https://huggingface.co/Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)  
5. Z-Image-Turbo | Text-to-Image by Tongyi-MAI | WaveSpeedAI, accessed on December 22, 2025, [https://wavespeed.ai/models/wavespeed-ai/z-image/turbo](https://wavespeed.ai/models/wavespeed-ai/z-image/turbo)  
6. Z-Image-Turbo-GGUF Free Image Generate Online, Click to Use\!, accessed on December 22, 2025, [https://skywork.ai/blog/models/z-image-turbo-gguf-free-image-generate-online/](https://skywork.ai/blog/models/z-image-turbo-gguf-free-image-generate-online/)  
7. example\_workflow.json · jayn7/Z-Image-Turbo-GGUF at main, accessed on December 22, 2025, [https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/example\_workflow.json](https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/blob/main/example_workflow.json)  
8. Z-Image Turbo: Fast and Functional Photorealism | Diffusion Doodles, accessed on December 22, 2025, [https://medium.com/diffusion-doodles/z-image-turbo-fast-and-functional-photorealism-eb5ba351ba52](https://medium.com/diffusion-doodles/z-image-turbo-fast-and-functional-photorealism-eb5ba351ba52)  
9. Koko-boya/Comfyui-Z-Image-Utilities: ComfyUI utility nodes ... \- GitHub, accessed on December 22, 2025, [https://github.com/Koko-boya/Comfyui-Z-Image-Utilities](https://github.com/Koko-boya/Comfyui-Z-Image-Utilities)