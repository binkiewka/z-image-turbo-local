# Running Z-Image-Turbo on Windows with WSL2

This guide walks you through setting up the Z-Image-Turbo AI image generation system on Windows using Windows Subsystem for Linux 2 (WSL2) with full NVIDIA GPU acceleration.

## What is WSL2?

Windows Subsystem for Linux 2 (WSL2) is a compatibility layer that allows you to run a full Linux environment directly on Windows without the overhead of a traditional virtual machine. For GPU-accelerated applications like Z-Image-Turbo, WSL2 provides native CUDA support through GPU passthrough, enabling near-native performance.

**Expected Outcome:** After completing this guide, you'll have a fully functional GPU-accelerated AI image generation system running on Windows, capable of generating 1024x1024 images in ~3 seconds.

**Official Documentation:** [Microsoft WSL Documentation](https://learn.microsoft.com/en-us/windows/wsl/)

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Install WSL2 with Ubuntu 22.04](#step-1-install-wsl2-with-ubuntu-2204)
3. [Step 2: Install Windows NVIDIA GPU Driver](#step-2-install-windows-nvidia-gpu-driver)
4. [Step 3A: Install Docker CE inside WSL2 (Recommended)](#step-3a-install-docker-ce-inside-wsl2-recommended)
5. [Step 3B: Install NVIDIA Container Toolkit](#step-3b-install-nvidia-container-toolkit)
6. [Step 3C: Alternative - Docker Desktop for Windows](#step-3c-alternative---docker-desktop-for-windows)
7. [Step 4: Clone Repository and Setup](#step-4-clone-repository-and-setup)
8. [Step 5: Download Models](#step-5-download-models)
9. [Step 6: Build and Run](#step-6-build-and-run)
10. [Step 7: Access from Windows](#step-7-access-from-windows)
11. [WSL2-Specific Troubleshooting](#wsl2-specific-troubleshooting)
12. [Performance Optimization](#performance-optimization)
13. [Additional Resources](#additional-resources)

---

## Prerequisites

### System Requirements

| Component | Requirement |
|-----------|-------------|
| **Windows Version** | Windows 10 version 21H2 or higher, OR Windows 11 |
| **GPU** | NVIDIA RTX 3060 12GB (or similar with 12GB+ VRAM) |
| **RAM** | 16GB minimum (32GB recommended) |
| **Storage** | 20GB+ free space (15GB for models + Docker images) |
| **Virtualization** | Enabled in BIOS/UEFI |
| **Administrator** | Administrator access to Windows |
| **Ports** | 7860 and 8188 available (not 8081/8082) |

### Verify Windows Version

Open PowerShell and run:

```powershell
winver
```

You should see version 21H2 or higher (or Windows 11).

### Verify Virtualization is Enabled

Open Task Manager (Ctrl+Shift+Esc) → Performance tab → CPU. Check that "Virtualization" shows as "Enabled".

If disabled, you'll need to enable it in your BIOS/UEFI settings (restart your computer and press F2/F10/Del during boot).

---

## Step 1: Install WSL2 with Ubuntu 22.04

WSL2 installation has been greatly simplified in recent Windows versions. A single command handles everything.

### 1.1 Open PowerShell as Administrator

Right-click the Start button and select "Windows PowerShell (Admin)" or "Terminal (Admin)".

### 1.2 Install Ubuntu 22.04

Run the following command:

```powershell
wsl --install Ubuntu-22.04
```

This command will:

- Enable the required Windows features (WSL and Virtual Machine Platform)
- Download and install WSL2
- Download and install Ubuntu 22.04 from the Microsoft Store
- Set WSL2 as the default version

### 1.3 Restart Your Computer

When prompted, restart your computer.

### 1.4 Complete Ubuntu Setup

After restart, Ubuntu will launch automatically. You'll be prompted to:

1. Wait for installation to complete (1-5 minutes)
2. Create a UNIX username (lowercase, no spaces)
3. Create a password (won't show while typing)
4. Confirm password

**Example:**

```
Enter new UNIX username: binkiewka
New password: ********
Retype new password: ********
```

### 1.5 Verify Installation

In the Ubuntu terminal, run:

```bash
wsl -l -v
```

Expected output:

```
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

The `VERSION` column must show `2` (not `1`).

### 1.6 Update Ubuntu

Update all packages to the latest versions:

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Step 2: Install Windows NVIDIA GPU Driver

**⚠️ CRITICAL: Install the Windows GPU driver ONLY. Do NOT install any Linux NVIDIA drivers inside WSL2.**

WSL2 uses a special architecture where the Windows NVIDIA driver is automatically made available inside WSL2 as a stub (`libcuda.so`). Installing a Linux driver will break GPU passthrough.

### 2.1 Download Windows NVIDIA Driver

1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/download/index.aspx)
2. Select your GPU model (e.g., GeForce RTX 3060)
3. Select "Windows 10 64-bit" or "Windows 11"
4. Download the driver (version **550 or higher** recommended for CUDA 12.1 support)

**Minimum driver version:** 470.xx
**Recommended:** 550+ (latest Game Ready or Studio driver)

### 2.2 Install the Driver on Windows

1. Run the downloaded `.exe` installer
2. Choose "Express Installation" or "Custom Installation"
3. Follow the installation wizard
4. Restart Windows if prompted

### 2.3 Verify GPU Access from WSL2

After driver installation, open Ubuntu from the Start menu and run:

```bash
nvidia-smi
```

**Expected output:**

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.1   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 170W |    500MiB / 12288MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

If you see this output, GPU passthrough is working correctly!

**If you get "nvidia-smi: command not found":** See the [Troubleshooting](#wsl2-specific-troubleshooting) section.

---

## Step 3A: Install Docker CE inside WSL2 (Recommended)

**Why this approach is recommended:**

- Completely free for all users (no commercial licensing restrictions)
- Native Linux experience, aligns with the main project README
- Full control over Docker configuration
- Matches the CLI-based workflow used by this project

### 3.1 Add Docker's Official GPG Key and Repository

Run these commands in your Ubuntu terminal:

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

### 3.2 Add Docker Repository

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  jammy stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 3.3 Install Docker Engine

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 3.4 Add Your User to the Docker Group

This allows you to run Docker commands without `sudo`:

```bash
sudo usermod -aG docker $USER
```

**Important:** Log out and log back into Ubuntu for group changes to take effect.

To log out and back in:

```bash
exit
```

Then reopen Ubuntu from the Start menu.

### 3.5 Verify Docker Installation

```bash
docker run hello-world
```

**Expected output:**

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
...
```

### 3.6 Enable Docker to Start Automatically

Ubuntu 22.04 on WSL2 supports systemd, which allows Docker to start automatically:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

Verify Docker is running:

```bash
sudo systemctl status docker
```

You should see `active (running)` in green.

---

## Step 3B: Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker containers to access your NVIDIA GPU.

### 3.7 Add NVIDIA Container Toolkit Repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3.8 Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 3.9 Configure Docker Runtime

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

This command modifies `/etc/docker/daemon.json` to add the NVIDIA runtime.

### 3.10 Restart Docker

```bash
sudo systemctl restart docker
```

### 3.11 Test GPU Access in Docker

Run this test to verify that containers can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

**Expected output:** Same `nvidia-smi` output as before, but running inside a Docker container.

If you see GPU information, **GPU passthrough is working!** You can proceed to [Step 4](#step-4-clone-repository-and-setup).

---

## Step 3C: Alternative - Docker Desktop for Windows

If you prefer a GUI-based Docker experience or are new to Docker, you can use Docker Desktop for Windows with the WSL2 backend instead of Docker CE.

**When to choose Docker Desktop:**

- You prefer a graphical interface for managing containers
- You're new to Docker and want an easier setup
- You're okay with commercial licensing restrictions

**⚠️ Licensing Note:** Docker Desktop is free for:

- Personal use
- Education
- Non-commercial open source projects
- Small businesses (fewer than 250 employees AND less than $10M revenue)

For larger companies, a [paid subscription](https://www.docker.com/pricing/) is required.

### 3C.1 Download Docker Desktop

Visit [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/) and download the installer.

### 3C.2 Install Docker Desktop

1. Run the installer
2. Ensure "Use WSL 2 instead of Hyper-V" is checked
3. Complete installation
4. Restart Windows if prompted

### 3C.3 Configure WSL2 Backend

1. Open Docker Desktop
2. Go to Settings (gear icon) → General
3. Ensure "Use the WSL 2 based engine" is checked
4. Go to Settings → Resources → WSL Integration
5. Enable integration with "Ubuntu-22.04"
6. Click "Apply & Restart"

### 3C.4 Verify GPU Support

Docker Desktop has built-in GPU support for WSL2. Test it with:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

If you see GPU information, you're ready to proceed to [Step 4](#step-4-clone-repository-and-setup).

**Note:** With Docker Desktop, you don't need to manually install the NVIDIA Container Toolkit.

---

## Step 4: Clone Repository and Setup

### 4.1 Filesystem Choice - Important for Performance

**⚠️ Use WSL2's native filesystem, NOT the Windows filesystem mounted at `/mnt/c`.**

**Why?**

- WSL2 native filesystem (ext4): Fast, optimized for Linux
- Windows filesystem via `/mnt/c`: 2-5x slower due to 9P protocol overhead
- Model files are large (10GB+), and performance is critical

**Recommended:** Work in your WSL2 home directory (`~/` or `/home/<your-wsl-username>/`).

### 4.2 Navigate to Home Directory

```bash
cd ~/
```

### 4.3 Clone the Repository

If you have the repository on GitHub:

```bash
git clone https://github.com/binkiewka/z-image-turbo-local.git
cd z-image-turbo-local
```

If you're copying files from Windows:

```bash
# Copy from Windows to WSL2 (one-time operation)
cp -r /mnt/c/Users/YourName/Desktop/z-image-turbo-local ~/z-image-turbo-local
cd ~/z-image-turbo-local
```

**After copying, always work from `~/z-image-turbo-local`, NOT `/mnt/c/...`**

### 4.4 Create Model Directories

```bash
mkdir -p models/diffusion_models models/text_encoders models/vae output
```

Verify structure:

```bash
ls -la models/
```

Expected output:

```
drwxr-xr-x  2 user user 4096 Dec 23 10:00 diffusion_models
drwxr-xr-x  2 user user 4096 Dec 23 10:00 text_encoders
drwxr-xr-x  2 user user 4096 Dec 23 10:00 vae
```

---

## Step 5: Download Models

You'll need to download three model files totaling ~10GB.

### 5.1 Download via wget

From your `image-gen` directory, download the two public models:

**Z-Image-Turbo diffusion model (7.2GB):**

```bash
wget -O models/diffusion_models/z_image_turbo-Q8_0.gguf \
  "https://huggingface.co/jayn7/Z-Image-Turbo-GGUF/resolve/main/z_image_turbo-Q8_0.gguf"
```

**Qwen 3 4B text encoder (2.3GB):**

```bash
wget -O models/text_encoders/Qwen_3_4b-IQ4_XS.gguf \
  "https://huggingface.co/worstplayer/Z-Image_Qwen_3_4b_text_encoder_GGUF/resolve/main/Qwen_3_4b-IQ4_XS.gguf"
```

### 5.2 Flux VAE - Manual Download Required (335MB)

> [!IMPORTANT]
> This model is gated and requires HuggingFace authentication. You must download it manually through your web browser.

1. Visit [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
2. Log in to HuggingFace (create an account if needed)
3. Click "Agree and access repository"
4. Navigate to: [ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors)
5. Click the "download" button (↓ icon) - saves to Windows Downloads folder
6. Copy from Windows to WSL2:

```bash
# Copy from Windows Downloads to WSL2 (adjust username)
cp /mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/ae.safetensors ~/z-image-turbo-local/models/vae/
```

> [!WARNING]
> **Common mistake:** If the file shows `0 bytes`, the download failed. The actual file should be approximately **335MB**.

### 5.3 Verify Downloads (CRITICAL!)

> [!CAUTION]
> **Do NOT proceed to the next step until all files pass verification.** A 0-byte or missing VAE file is the #1 cause of startup failures.

Run this verification command:

```bash
echo "=== Model Verification ===" && \
for f in models/diffusion_models/z_image_turbo-Q8_0.gguf models/text_encoders/Qwen_3_4b-IQ4_XS.gguf models/vae/ae.safetensors; do \
  if [ ! -f "$f" ]; then echo "❌ MISSING: $f"; \
  elif [ ! -s "$f" ]; then echo "❌ EMPTY (0 bytes): $f - DELETE AND RE-DOWNLOAD!"; \
  elif [ -L "$f" ]; then echo "❌ SYMLINK (broken): $f - DELETE AND RE-DOWNLOAD!"; \
  else echo "✅ OK: $f ($(du -h "$f" | cut -f1))"; fi; \
done
```

**Expected output (all files should show ✅):**

```
=== Model Verification ===
✅ OK: models/diffusion_models/z_image_turbo-Q8_0.gguf (7.2G)
✅ OK: models/text_encoders/Qwen_3_4b-IQ4_XS.gguf (2.3G)
✅ OK: models/vae/ae.safetensors (335M)
```

**If any file shows ❌:**

1. Delete the broken file: `rm <path-to-file>`
2. Re-download following the instructions above
3. For the VAE file, you MUST download via browser and copy from Windows Downloads

---

## Step 6: Build and Run

### 6.1 Build Docker Containers

From the `image-gen` directory:

```bash
docker compose build --no-cache
```

This will take 5-15 minutes depending on your internet speed.

### 6.2 Start Services

```bash
docker compose up -d
```

The `-d` flag runs containers in detached mode (background).

### 6.3 View Logs

Monitor the startup process:

```bash
docker compose logs -f
```

Press `Ctrl+C` to stop following logs (containers will keep running).

### 6.4 Wait for Healthcheck

The backend has a healthcheck that can take 30-60 seconds. You'll see:

```
z-image-backend | Healthcheck passed
z-image-frontend | Connected to ComfyUI backend
```

Once you see these messages, the system is ready!

---

## Step 7: Access from Windows

### 7.1 Open Browser

On your Windows machine, open any web browser and navigate to:

```
http://localhost:7860
```

**WSL2 automatically forwards `localhost` ports to the host Windows system.**

### 7.2 Expected UI

You should see a web interface with:

- Prompt text box
- Negative prompt text box
- Seed input (-1 for random)
- Steps slider (4-12, default 8)
- Aspect ratio dropdown (1:1, 3:4, 4:3, 16:9, 9:16)
- Generate button
- Image display area
- Download link

### 7.3 Test Generation

Try this sample prompt:

```
a cat wearing a wizard hat, detailed, high quality
```

Click "Generate" and wait ~3 seconds (first generation may take 10-30s due to model loading).

### 7.4 Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| First generation | 10-30 seconds (model loading) |
| Subsequent generations | ~3 seconds |
| VRAM usage | ~11.4GB / 12GB |
| Maximum resolution | 1024x1024 |

---

## WSL2-Specific Troubleshooting

### Issue: "nvidia-smi: command not found" in WSL2

**Possible Causes:**

1. Windows NVIDIA driver not installed
2. WSL2 version is 1 instead of 2
3. Windows version too old (need 21H2+)

**Solutions:**

1. **Verify Windows driver is installed:**
   - On Windows, open Command Prompt and run: `nvidia-smi`
   - If not found, reinstall Windows NVIDIA driver (Step 2)

2. **Verify WSL version:**

   ```bash
   wsl -l -v
   ```

   Must show `VERSION 2`, not `1`. If showing `1`, convert to version 2:

   ```powershell
   wsl --set-version Ubuntu-22.04 2
   ```

3. **Check Windows version:**
   - Run `winver` in PowerShell
   - Need Windows 10 21H2+ or Windows 11
   - Update Windows if necessary

### Issue: "Docker daemon not starting"

**Possible Causes:**

1. Docker service not running
2. systemd not enabled in WSL2

**Solutions:**

1. **Start Docker manually:**

   ```bash
   sudo service docker start
   ```

2. **Check Docker status:**

   ```bash
   sudo systemctl status docker
   ```

3. **Enable systemd in WSL2 (Ubuntu 22.04):**
   Edit `/etc/wsl.conf`:

   ```bash
   sudo nano /etc/wsl.conf
   ```

   Add these lines:

   ```ini
   [boot]
   systemd=true
   ```

   Save (Ctrl+O, Enter, Ctrl+X), then restart WSL2:

   ```powershell
   wsl --shutdown
   ```

   Reopen Ubuntu from Start menu.

### Issue: "Cannot connect to Docker daemon"

**Error message:**

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

**Possible Causes:**

1. User not in `docker` group
2. Logout/login required after `usermod`
3. Docker service not running

**Solutions:**

1. **Verify group membership:**

   ```bash
   groups
   ```

   Should include `docker`. If not:

   ```bash
   sudo usermod -aG docker $USER
   ```

2. **Logout and login:**

   ```bash
   exit
   ```

   Reopen Ubuntu from Start menu.

3. **Restart Docker:**

   ```bash
   sudo systemctl restart docker
   ```

### Issue: "No NVIDIA GPU devices found"

**Error in container logs:**

```
RuntimeError: No CUDA GPUs are available
```

**Possible Causes:**

1. Windows NVIDIA driver too old (<470)
2. nvidia-container-toolkit not installed
3. Docker runtime not configured

**Solutions:**

1. **Verify driver version:**

   ```bash
   nvidia-smi
   ```

   Check "Driver Version" in output. Must be 470+, recommend 550+.

2. **Test GPU in Docker:**

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
   ```

   If this fails, nvidia-container-toolkit is not properly configured.

3. **Reinstall nvidia-container-toolkit:**

   ```bash
   sudo apt-get install --reinstall -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

### Issue: "Models loading slow or from /mnt/c"

**Symptoms:**

- First image generation takes >60 seconds
- Subsequent images also slow (>10 seconds)

**Cause:** Models are stored on Windows filesystem (`/mnt/c`) instead of WSL2 native filesystem.

**Solution:**

1. **Check current location:**

   ```bash
   pwd
   ```

   If shows `/mnt/c/...`, you're in the wrong place.

2. **Move to WSL2 filesystem:**

   ```bash
   cd ~/
   cp -r /mnt/c/Users/YourName/Desktop/z-image-turbo-local ~/z-image-turbo-local
   cd ~/z-image-turbo-local
   ```

3. **Rebuild containers:**

   ```bash
   docker compose down -v
   docker compose build --no-cache
   docker compose up -d
   ```

**Performance comparison:**

- `/mnt/c` (Windows filesystem): 2-5x slower
- `~/` (WSL2 ext4): Optimal performance

### Issue: "Can't access localhost:7860 from Windows"

**Symptoms:**

- Browser shows "This site can't be reached"
- Connection refused or timeout

**Possible Causes:**

1. Containers not running
2. Windows Firewall blocking WSL2
3. WSL2 networking issue

**Solutions:**

1. **Verify containers are running:**

   ```bash
   docker compose ps
   ```

   Both `z-image-backend` and `z-image-frontend` should show "Up".

2. **Check WSL2 IP address:**

   ```bash
   ip addr show eth0 | grep inet
   ```

   Try accessing via WSL2 IP directly: `http://<wsl-ip>:7860`

3. **Windows Firewall:**
   - Open Windows Defender Firewall
   - Allow an app through firewall
   - Ensure "vEthernet (WSL)" is allowed for private networks

4. **Port forwarding:**
   WSL2 should auto-forward ports, but you can manually forward:

   ```powershell
   # In PowerShell (Admin)
   netsh interface portproxy add v4tov4 listenport=7860 listenaddress=0.0.0.0 connectport=7860 connectaddress=<wsl-ip>
   ```

### Issue: "Port already in use"

**Error message:**

```
Error starting userland proxy: listen tcp4 0.0.0.0:7860: bind: address already in use
```

**Solutions:**

1. **Find process using the port:**

   ```bash
   sudo lsof -i :7860
   sudo lsof -i :8188
   ```

2. **Kill the process:**

   ```bash
   sudo kill -9 <PID>
   ```

3. **Change ports in docker-compose.yml:**
   Edit `docker-compose.yml`:

   ```yaml
   ports:
     - "0.0.0.0:7861:7860"  # Changed from 7860 to 7861
   ```

   **Note:** Avoid using ports 8081 and 8082 (reserved for other services).

### Issue: "Model not found" / VAE Symlink Error

**Error message:**

```
WARNING path /app/ComfyUI/models/vae/ae.safetensors exists but doesn't link anywhere, skipping.
FileNotFoundError: Model in folder 'vae' with filename 'ae.safetensors' not found.
```

**Causes:**

1. **0-byte placeholder file** - Most common! The file exists but is empty (failed or incomplete download)
2. **Broken symlink** - The file is a symlink that points nowhere (can happen with certain download tools)
3. **Incomplete download** - Browser download was interrupted

**Diagnosis:**

First, check if the file is real or empty:

```bash
ls -lh models/vae/ae.safetensors
```

- ✅ **Good:** Shows `335M` or similar size
- ❌ **Bad:** Shows `0` bytes - the file is empty/broken!
- ❌ **Bad:** Shows `->` arrow (symlink) pointing to another path

**Solution (if file is 0 bytes or broken):**

1. **Delete the broken file:**

   ```bash
   rm models/vae/ae.safetensors
   ```

2. **Manually download the VAE model:**

   Since this model is gated (requires HuggingFace authentication), you must download manually:

   a. Visit [ae.safetensors on HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-schnell/blob/main/ae.safetensors)
   b. Log in and accept the license if prompted
   c. Click the download button (↓)
   d. Copy from Windows Downloads to WSL2:

   ```bash
   cp /mnt/c/Users/YOUR_USERNAME/Downloads/ae.safetensors ~/z-image-turbo-local/models/vae/
   ```

3. **Verify the file is real (~335MB):**

   ```bash
   ls -lh models/vae/ae.safetensors
   # Should show: -rw-rw-r-- 1 user user 335M Dec 23 12:00 ae.safetensors
   ```

4. **Restart containers:**

   ```bash
   docker compose restart
   ```

---

### Issue: "CUDA Out of Memory" in WSL2

**Error in logs:**

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Close other GPU applications on Windows:**
   - Check Task Manager → Performance → GPU
   - Close games, video editors, or other GPU-heavy apps

2. **Reduce resolution:**
   Use lower aspect ratios (e.g., 3:4 instead of 1:1)

3. **Reduce steps:**
   Set steps to 4-6 instead of 8

4. **Allocate more VRAM to WSL2:**
   Edit `C:\Users\<username>\.wslconfig`:

   ```ini
   [wsl2]
   memory=16GB
   ```

   Restart WSL2:

   ```powershell
   wsl --shutdown
   ```

---

## Performance Optimization

### WSL2 Resource Configuration

Create or edit `C:\Users\<YourUsername>\.wslconfig` on Windows:

```ini
[wsl2]
# RAM allocation (use half of total RAM for 32GB systems)
memory=16GB

# CPU cores (leave some for Windows)
processors=8

# Swap space
swap=8GB

# Enable localhost forwarding
localhostForwarding=true

# Network mode (default is NAT)
# networkingMode=mirrored  # Windows 11 22H2+ only
```

**After editing, restart WSL2:**

Open PowerShell (Admin):

```powershell
wsl --shutdown
```

Then reopen Ubuntu from Start menu.

### Filesystem Performance Tips

1. **Always use WSL2 native filesystem:**
   - ✅ Good: `/home/username/image-gen`
   - ❌ Bad: `/mnt/c/Users/username/Desktop/image-gen`

2. **Why WSL2 filesystem is faster:**
   - Native ext4 filesystem (Linux-native)
   - Direct disk access
   - No 9P protocol overhead
   - **2-5x faster** for large files

3. **When to use /mnt/c:**
   - Sharing small files with Windows
   - Temporary file transfers
   - Never for active development or large model files

### Docker Performance

1. **Use BuildKit for faster builds:**
   Already enabled with `docker-buildx-plugin`.

2. **Prune unused resources:**

   ```bash
   docker system prune -a
   ```

3. **Monitor resource usage:**

   ```bash
   docker stats
   ```

### GPU Performance

1. **Monitor GPU usage:**

   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Check VRAM during generation:**

   ```bash
   docker exec z-image-backend nvidia-smi
   ```

3. **Expected VRAM usage:** ~11.4GB / 12GB during generation

---

## Additional Resources

### Official Documentation

- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [Microsoft Enable NVIDIA CUDA on WSL 2](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
- [Microsoft WSL Installation Guide](https://learn.microsoft.com/en-us/windows/wsl/install)
- [Ubuntu on WSL Documentation](https://documentation.ubuntu.com/wsl/stable/)
- [Docker Desktop WSL 2 Backend](https://docs.docker.com/desktop/features/wsl/)
- [Docker GPU Support](https://docs.docker.com/desktop/features/gpu/)
- [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Project Documentation

- [Main README](../../README.md) - General usage, configuration, and troubleshooting
- [Architecture PRD](../../Dockerized%20AI%20Image%20Generation%20PRD.md) - Technical deep dive

### Community Resources

- [WSL2 GitHub Issues](https://github.com/microsoft/WSL/issues)
- [Docker WSL2 Backend Issues](https://github.com/docker/for-win/issues)
- [NVIDIA Container Toolkit Discussions](https://github.com/NVIDIA/nvidia-docker/discussions)

---

## Need Help?

1. **Check this troubleshooting section** for WSL2-specific issues
2. **Review main project README** for general application issues
3. **Check Docker logs:**

   ```bash
   docker compose logs -f
   ```

4. **Verify GPU access:**

   ```bash
   docker exec z-image-backend nvidia-smi
   ```

5. **Check WSL2 logs:**

   ```bash
   dmesg | grep -i nvidia
   ```

---

**Ready to generate images?** Head to `http://localhost:7860` and start creating!
