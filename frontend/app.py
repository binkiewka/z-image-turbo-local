import os
import json
import uuid
import random
import asyncio
import time
import shutil
import gc
from io import BytesIO
from pathlib import Path

import gradio as gr
import httpx
import websockets
from PIL import Image
from prompt_enhancer import enhance_prompt, get_enhancer

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
COMFYUI_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_WS = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"

ASPECT_RATIOS = {
    "1:1 (1024x1024)": (1024, 1024),
    "3:4 (896x1152)": (896, 1152),
    "4:3 (1152x896)": (1152, 896),
    "16:9 (1280x720)": (1280, 720),
    "9:16 (720x1280)": (720, 1280),
}

# Ensure download directory exists
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp/zimage_downloads"))
DOWNLOAD_DIR.mkdir(exist_ok=True)

# Video output directory (mounted from backend)
VIDEO_OUTPUT_DIR = Path("/app/output")

with open("workflow_api.json", "r") as f:
    WORKFLOW_TEMPLATE = json.load(f)

# Load video workflows
with open("workflow_video_t2v.json", "r") as f:
    WORKFLOW_VIDEO_T2V = json.load(f)

with open("workflow_video_i2v.json", "r") as f:
    WORKFLOW_VIDEO_I2V = json.load(f)

# Global gallery state
gallery_history = []

# Video resolution options
VIDEO_RESOLUTIONS = {
    "480p (848x480)": (848, 480),
    "720p (1280x720)": (1280, 720),
}

# Video configuration defaults
VIDEO_CONFIG = {
    "fps": 16,
    "steps": 4,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "sgm_uniform",
    "blocks_to_swap": 40,
}


def scan_video_models():
    """Scan for available WAN 2.2 video models in ComfyUI models folder."""
    # Models are mounted at /app/ComfyUI/models/diffusion_models in backend
    # Frontend accesses via API, so we query ComfyUI for available models
    models = {
        "t2v_high": [],
        "t2v_low": [],
        "i2v_high": [],
        "i2v_low": [],
        "loras_high": [],
        "loras_low": [],
    }

    try:
        import httpx
        response = httpx.get(f"{COMFYUI_URL}/object_info/UnetLoaderGGUF", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if "UnetLoaderGGUF" in data:
                all_models = data["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0]
                for model in all_models:
                    model_lower = model.lower()
                    if "wan" in model_lower or "wan2" in model_lower:
                        if "t2v" in model_lower:
                            if "high" in model_lower:
                                models["t2v_high"].append(model)
                            elif "low" in model_lower:
                                models["t2v_low"].append(model)
                        elif "i2v" in model_lower:
                            if "high" in model_lower:
                                models["i2v_high"].append(model)
                            elif "low" in model_lower:
                                models["i2v_low"].append(model)

        # Scan for LoRAs
        response = httpx.get(f"{COMFYUI_URL}/object_info/LoraLoaderModelOnly", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if "LoraLoaderModelOnly" in data:
                all_loras = data["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0]
                for lora in all_loras:
                    lora_lower = lora.lower()
                    if "wan" in lora_lower and "lightx2v" in lora_lower:
                        if "high" in lora_lower:
                            models["loras_high"].append(lora)
                        elif "low" in lora_lower:
                            models["loras_low"].append(lora)
    except Exception as e:
        print(f"Error scanning models: {e}")

    return models


def prepare_workflow(prompt: str, seed: int, steps: int, width: int, height: int, num_images: int = 1) -> tuple[dict, int]:
    """Prepare the workflow with user inputs."""
    workflow = json.loads(json.dumps(WORKFLOW_TEMPLATE))

    # Positive prompt
    workflow["8"]["inputs"]["text"] = prompt

    # Negative prompt (not used by Z-Image-Turbo, but keep empty for workflow compatibility)
    workflow["3"]["inputs"]["text"] = ""

    # Seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    workflow["10"]["inputs"]["seed"] = seed

    # Steps
    workflow["10"]["inputs"]["steps"] = steps

    # Resolution and batch size
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height
    workflow["5"]["inputs"]["batch_size"] = num_images

    return workflow, seed


async def submit_and_wait_with_progress(workflow: dict, client_id: str, progress_callback=None) -> tuple[str, str]:
    """Submit workflow to ComfyUI and wait for completion via WebSocket with progress updates."""
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        response = await http_client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        )
        response.raise_for_status()
        result = response.json()
        prompt_id = result["prompt_id"]

    retries = 3
    for attempt in range(retries):
        try:
            async with websockets.connect(
                f"{COMFYUI_WS}?clientId={client_id}",
                ping_interval=20,
                ping_timeout=60
            ) as ws:
                async for message in ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type")

                        # Progress update
                        if msg_type == "progress" and progress_callback:
                            prog_data = data.get("data", {})
                            value = prog_data.get("value", 0)
                            max_val = prog_data.get("max", 8)
                            progress_callback(value, max_val)

                        # Execution complete
                        if msg_type == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                if exec_data.get("node") is None:
                                    output_data = await get_history(prompt_id)
                                    if output_data:
                                        return output_data

                        elif msg_type == "execution_error":
                            error_data = data.get("data", {})
                            if error_data.get("prompt_id") == prompt_id:
                                raise Exception(f"Execution error: {error_data}")

        except websockets.exceptions.ConnectionClosed:
            if attempt < retries - 1:
                await asyncio.sleep(2)
                continue
            raise

    raise Exception("Failed to connect to ComfyUI WebSocket after retries")


async def get_history(prompt_id: str) -> list[tuple[str, str]] | None:
    """Get the output image info from history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    # Return all images from batch
                    images_info = []
                    for img in node_output["images"]:
                        images_info.append((img["filename"], img.get("subfolder", "")))
                    return images_info if images_info else None
    return None


async def fetch_image(filename: str, subfolder: str) -> Image.Image:
    """Fetch the generated image from ComfyUI."""
    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{COMFYUI_URL}/view", params=params)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


def generate_image(prompt: str, seed: int, steps: int, aspect_ratio: str, num_images: int, current_gallery: list, progress=gr.Progress()):
    """Main generation function with progress tracking."""
    global gallery_history

    if not prompt or not prompt.strip():
        return None, "Please enter a prompt", current_gallery

    actual_prompt = prompt.strip()
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    num_images = int(num_images)
    workflow, actual_seed = prepare_workflow(
        actual_prompt,
        int(seed),
        int(steps),
        width,
        height,
        num_images
    )
    client_id = str(uuid.uuid4())

    # Progress tracking
    current_step = [0]
    max_steps = [int(steps)]

    def update_progress(value, max_val):
        current_step[0] = value
        max_steps[0] = max_val
        progress((value / max_val), desc=f"Step {value}/{max_val}")

    async def run_generation():
        return await submit_and_wait_with_progress(workflow, client_id, update_progress)

    try:
        progress(0, desc="Starting generation...")
        result = asyncio.run(run_generation())

        if result:
            images = []
            for i, (filename, subfolder) in enumerate(result):
                image = asyncio.run(fetch_image(filename, subfolder))
                
                # Save for download
                download_path = DOWNLOAD_DIR / f"zimage_{actual_seed}_{i}_{int(time.time())}.png"
                image.save(download_path)
                
                images.append(image)
                
                # Update gallery
                gallery_entry = (image, f"Seed: {actual_seed} ({i+1}/{num_images})")
                gallery_history.insert(0, gallery_entry)
            
            gallery_history = gallery_history[:12]  # Keep last 12
            
            # Return first image as main display
            return images[0], f"Done! Seed: {actual_seed} ({num_images} image(s))", gallery_history
        else:
            return None, "No image generated", current_gallery

    except Exception as e:
        return None, f"Error: {str(e)}", current_gallery


def enhance_only(prompt, progress=gr.Progress()):
    """Enhance prompt without generating image."""
    if not prompt or not prompt.strip():
        return "", "Please enter a prompt first", gr.update(visible=False)

    try:
        # Define progress callback for status updates
        def progress_callback(status_msg):
            progress(0.5, desc=status_msg)

        progress(0, desc="Loading enhancer model...")
        enhanced = enhance_prompt(prompt.strip(), progress_callback=progress_callback)
        progress(1.0, desc="Done!")
        return enhanced, "✨ Prompt enhanced! Click 'Generate with Enhanced' or 'Copy to Prompt'", gr.update(visible=True)
    except Exception as e:
        return "", f"Enhancement failed: {e}", gr.update(visible=False)


# ===========================
# VIDEO GENERATION FUNCTIONS
# ===========================

def prepare_video_workflow(
    mode: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    model_high: str,
    model_low: str,
    lora_high: str,
    lora_low: str,
    input_image: str = None
) -> tuple[dict, int]:
    """Prepare the video workflow with user inputs."""
    if mode == "Text to Video":
        workflow = json.loads(json.dumps(WORKFLOW_VIDEO_T2V))
    else:
        workflow = json.loads(json.dumps(WORKFLOW_VIDEO_I2V))

    # Seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Update model names
    workflow["2"]["inputs"]["unet_name"] = model_high
    workflow["3"]["inputs"]["unet_name"] = model_low

    # Update LoRA names
    workflow["4"]["inputs"]["lora_name"] = lora_high
    workflow["5"]["inputs"]["lora_name"] = lora_low

    # Update prompt
    workflow["8"]["inputs"]["text"] = prompt

    # Update resolution, frame count, and seed based on mode
    if mode == "Text to Video":
        # T2V: EmptyHunyuanLatentVideo is node 10, KSamplers are 11 and 12
        workflow["10"]["inputs"]["width"] = width
        workflow["10"]["inputs"]["height"] = height
        workflow["10"]["inputs"]["length"] = frames
        workflow["11"]["inputs"]["noise_seed"] = seed
        workflow["12"]["inputs"]["noise_seed"] = seed
    else:
        # I2V: WanImageToVideo is node 12, KSamplers are 13 and 14
        workflow["12"]["inputs"]["width"] = width
        workflow["12"]["inputs"]["height"] = height
        workflow["12"]["inputs"]["length"] = frames
        workflow["13"]["inputs"]["noise_seed"] = seed
        workflow["14"]["inputs"]["noise_seed"] = seed
        # Set input image
        if input_image:
            workflow["10"]["inputs"]["image"] = input_image

    return workflow, seed


async def submit_and_wait_video(workflow: dict, client_id: str, progress_callback=None) -> tuple[str, str] | None:
    """Submit video workflow and wait for completion with extended timeout."""
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        )
        response.raise_for_status()
        result = response.json()
        prompt_id = result["prompt_id"]

    retries = 3
    for attempt in range(retries):
        try:
            async with websockets.connect(
                f"{COMFYUI_WS}?clientId={client_id}",
                ping_interval=30,
                ping_timeout=120,
                close_timeout=30
            ) as ws:
                start_time = time.time()
                max_wait = 1200  # 20 minutes max for video

                async for message in ws:
                    # Timeout check
                    if time.time() - start_time > max_wait:
                        raise TimeoutError("Video generation timed out (20 min)")

                    if isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type")

                        # Progress update
                        if msg_type == "progress" and progress_callback:
                            prog_data = data.get("data", {})
                            value = prog_data.get("value", 0)
                            max_val = prog_data.get("max", 100)
                            progress_callback(value, max_val)

                        # Execution complete
                        if msg_type == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                if exec_data.get("node") is None:
                                    # Get video output info
                                    output_data = await get_video_history(prompt_id)
                                    if output_data:
                                        return output_data

                        elif msg_type == "execution_error":
                            error_data = data.get("data", {})
                            if error_data.get("prompt_id") == prompt_id:
                                raise Exception(f"Execution error: {error_data}")

        except websockets.exceptions.ConnectionClosed:
            if attempt < retries - 1:
                await asyncio.sleep(5)
                continue
            raise

    raise Exception("Failed to connect to ComfyUI WebSocket after retries")


async def get_video_history(prompt_id: str) -> tuple[str, str] | None:
    """Get the output video info from history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                # VHS_VideoCombine outputs gifs/videos
                if "gifs" in node_output:
                    for vid in node_output["gifs"]:
                        return (vid["filename"], vid.get("subfolder", ""))
                # Also check for images (fallback)
                if "images" in node_output:
                    for img in node_output["images"]:
                        if img["filename"].endswith((".mp4", ".webm", ".gif")):
                            return (img["filename"], img.get("subfolder", ""))
    return None


async def fetch_video(filename: str, subfolder: str) -> bytes:
    """Fetch the generated video from ComfyUI."""
    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{COMFYUI_URL}/view", params=params)
        response.raise_for_status()
        return response.content


async def free_comfyui_memory():
    """Call ComfyUI's /free API to release GPU/RAM memory after generation.
    
    This is critical for video generation which loads large models that
    would otherwise stay in memory and cause OOM on subsequent generations.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Call the /free endpoint to unload models and free memory
            response = await client.post(
                f"{COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True}
            )
            if response.status_code == 200:
                print("ComfyUI memory freed successfully")
            else:
                print(f"Warning: /free returned status {response.status_code}")
    except Exception as e:
        print(f"Warning: Failed to free ComfyUI memory: {e}")
    
    # Also run Python garbage collection
    gc.collect()


def upload_image_to_comfyui(image_path: str) -> str:
    """Upload an image to ComfyUI input folder and return filename."""
    import httpx

    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/png")}
        response = httpx.post(f"{COMFYUI_URL}/upload/image", files=files, timeout=30.0)
        response.raise_for_status()
        result = response.json()
        return result.get("name", os.path.basename(image_path))


def generate_video(
    mode: str,
    prompt: str,
    seed: int,
    resolution: str,
    frames: int,
    model_high: str,
    model_low: str,
    lora_high: str,
    lora_low: str,
    input_image=None,
    progress=gr.Progress()
):
    """Main video generation function."""
    if not prompt or not prompt.strip():
        return None, "Please enter a prompt"

    if not model_high or not model_low:
        return None, "Please select both High Noise and Low Noise models"

    if not lora_high or not lora_low:
        return None, "Please select both High Noise and Low Noise LoRAs"

    if mode == "Image to Video" and input_image is None:
        return None, "Please upload an input image for Image-to-Video mode"

    actual_prompt = prompt.strip()
    width, height = VIDEO_RESOLUTIONS.get(resolution, (848, 480))

    # Upload image if I2V mode
    uploaded_image_name = None
    if mode == "Image to Video" and input_image is not None:
        try:
            progress(0.05, desc="Uploading input image...")
            uploaded_image_name = upload_image_to_comfyui(input_image)
        except Exception as e:
            return None, f"Failed to upload image: {e}"

    workflow, actual_seed = prepare_video_workflow(
        mode=mode,
        prompt=actual_prompt,
        seed=int(seed),
        width=width,
        height=height,
        frames=int(frames),
        model_high=model_high,
        model_low=model_low,
        lora_high=lora_high,
        lora_low=lora_low,
        input_image=uploaded_image_name
    )
    client_id = str(uuid.uuid4())

    # Progress tracking
    current_step = [0]
    max_steps = [100]

    def update_progress(value, max_val):
        current_step[0] = value
        max_steps[0] = max_val
        # Video progress is slower, show estimated time
        pct = value / max_val if max_val > 0 else 0
        progress(pct, desc=f"Generating video... Frame {value}/{max_val}")

    async def run_video_generation():
        return await submit_and_wait_video(workflow, client_id, update_progress)

    try:
        progress(0.1, desc="Starting video generation (this will take 15-20 minutes)...")
        result = asyncio.run(run_video_generation())

        if result:
            filename, subfolder = result
            progress(0.95, desc="Downloading video...")

            # Fetch and save video
            video_bytes = asyncio.run(fetch_video(filename, subfolder))
            video_path = DOWNLOAD_DIR / f"wan2_video_{actual_seed}_{int(time.time())}.mp4"
            with open(video_path, "wb") as f:
                f.write(video_bytes)

            progress(1.0, desc="Done!")
            return str(video_path), f"Video generated! Seed: {actual_seed}"
        else:
            return None, "No video generated"

    except TimeoutError as e:
        return None, f"Timeout: {str(e)}"
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # CRITICAL: Free memory after video generation to prevent RAM buildup
        # Video models are very large and will cause OOM if not unloaded
        print("Cleaning up video generation memory...")
        asyncio.run(free_comfyui_memory())


def copy_to_prompt(enhanced_text):
    """Copy enhanced prompt to main prompt field."""
    return enhanced_text, gr.update(visible=False), ""


def warmup_models():
    """Pre-load models by running a dummy generation at startup."""
    print("Warming up models... (this may take 10-30 seconds on first run)")
    try:
        result = generate_image("warmup test", 42, 8, "1:1 (1024x1024)", 1, [], gr.Progress())
        if result[0] is not None:
            print("Models loaded successfully!")
            # Clear the warmup image from gallery
            global gallery_history
            gallery_history = []
        else:
            print(f"Warmup completed with message: {result[1]}")
    except Exception as e:
        print(f"Warmup failed (models will load on first request): {e}")


# Warmup models before starting Gradio
warmup_models()

# Enhancer model loads lazily on first use to save RAM
print("Prompt enhancer will load on first use (saves ~1.5GB RAM)")


# Custom CSS - keep structure, remove blue input fills
custom_css = """
/* Dark background */
.gradio-container {
    background: #0f1419 !important;
}

/* Keep panel borders and structure - darker transparent background */
.container, .panel, .block, .form, .wrap {
    background: rgba(20, 30, 40, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
}

/* Input fields - dark transparent, NO blue fill */
textarea, input[type="text"], input[type="number"] {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 6px !important;
    color: #e0e0e0 !important;
}

textarea:focus, input:focus {
    border-color: #0ea5e9 !important;
    outline: none !important;
}

/* Dropdowns - dark transparent */
select, .gr-dropdown {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #e0e0e0 !important;
}

/* Slider */
input[type="range"] {
    accent-color: #0ea5e9 !important;
}

/* Primary button - cyan */
button.primary {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 6px !important;
}

button.primary:hover {
    opacity: 0.9 !important;
}

/* Secondary button */
button.secondary {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 6px !important;
}

button.secondary:hover {
    background: rgba(255, 255, 255, 0.12) !important;
}

/* Image container */
.image-container, .gr-image {
    background: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
}

/* Gallery - no internal scroll, flows with page */
.gallery, .gr-gallery {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    overflow: visible !important;
}

/* Gallery grid - consistent thumbnail sizing */
.gr-gallery .grid-wrap {
    display: grid !important;
    grid-template-columns: repeat(6, 1fr) !important;
    gap: 8px !important;
    overflow: visible !important;
}

/* Gallery thumbnail containers */
.gr-gallery .grid-wrap > div,
.gr-gallery .thumbnail-item {
    aspect-ratio: 1 !important;
    height: auto !important;
    max-height: 120px !important;
    overflow: hidden !important;
}

/* Gallery images - consistent sizing */
.gr-gallery .grid-wrap img {
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
}

/* Headers - cyan */
h1, h2, h3 {
    color: #0ea5e9 !important;
    background: none !important;
    -webkit-text-fill-color: #0ea5e9 !important;
}

/* Labels */
label {
    color: #e0e0e0 !important;
}
"""

def refresh_video_models():
    """Refresh video model lists from ComfyUI - combines T2V and I2V models for user choice."""
    models = scan_video_models()
    # Combine T2V and I2V models into single lists for user to choose any model
    all_high_models = sorted(set(models["t2v_high"] + models["i2v_high"]))
    all_low_models = sorted(set(models["t2v_low"] + models["i2v_low"]))
    all_high_loras = sorted(set(models["loras_high"]))
    all_low_loras = sorted(set(models["loras_low"]))

    return (
        gr.update(choices=all_high_models, value=all_high_models[0] if all_high_models else None),
        gr.update(choices=all_low_models, value=all_low_models[0] if all_low_models else None),
        gr.update(choices=all_high_loras, value=all_high_loras[0] if all_high_loras else None),
        gr.update(choices=all_low_loras, value=all_low_loras[0] if all_low_loras else None),
    )


def update_video_mode(mode):
    """Update UI based on video mode selection."""
    if mode == "Image to Video":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


with gr.Blocks(
    title="Z-Image-Turbo & WAN 2.2 Video (Local)",
    theme=gr.themes.Glass(
        primary_hue="cyan",
        secondary_hue="sky",
        neutral_hue="slate",
    ),
    css=custom_css
) as demo:
    gr.Markdown("# ✨ Z-Image-Turbo & WAN 2.2 Video")
    gr.Markdown("*Generate stunning images and videos locally with AI-powered turbo mode*")

    with gr.Tabs():
        # ==================
        # IMAGE TAB
        # ==================
        with gr.TabItem("🖼️ Image Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=3,
                        max_lines=6
                    )

                    enhanced_prompt_output = gr.Textbox(
                        label="✨ Enhanced Prompt",
                        interactive=False,
                        visible=True,
                        lines=3,
                        max_lines=6,
                        placeholder="Click 'Enhance' to get an AI-expanded prompt..."
                    )

                    with gr.Row(visible=False) as enhanced_actions:
                        generate_enhanced_btn = gr.Button("🎨 Generate with Enhanced", variant="primary")
                        copy_prompt_btn = gr.Button("📋 Copy to Prompt", variant="secondary")

                    with gr.Row():
                        generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                        enhance_btn = gr.Button("✨ Enhance", variant="secondary", size="lg")

                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="Use -1 for random"
                        )
                        aspect_ratio = gr.Dropdown(
                            label="Aspect Ratio",
                            choices=list(ASPECT_RATIOS.keys()),
                            value="1:1 (1024x1024)"
                        )

                    with gr.Row():
                        steps = gr.Slider(
                            label="Steps",
                            minimum=4,
                            maximum=12,
                            value=8,
                            step=1,
                            info="More steps = better quality, slower"
                        )
                        num_images = gr.Dropdown(
                            label="Number of Images",
                            choices=[1, 2, 3, 4],
                            value=1,
                            info="Generate up to 4 variants"
                        )

                with gr.Column(scale=1):
                    output_image = gr.Image(
                        label="Generated Image",
                        type="pil",
                        height=512
                    )

                    status = gr.Textbox(label="Status", interactive=False)

            # Gallery section
            gr.Markdown("### 🖼️ Recent Generations")
            gallery = gr.Gallery(
                label="Recent Generations",
                show_label=False,
                columns=6,
                object_fit="cover",
                allow_preview=True
            )

            # Hidden state for gallery
            gallery_state = gr.State([])

            # Generate button click - uses original prompt
            generate_btn.click(
                fn=generate_image,
                inputs=[prompt, seed, steps, aspect_ratio, num_images, gallery_state],
                outputs=[output_image, status, gallery]
            )

            # Enter key in prompt - uses original prompt
            prompt.submit(
                fn=generate_image,
                inputs=[prompt, seed, steps, aspect_ratio, num_images, gallery_state],
                outputs=[output_image, status, gallery]
            )

            # Enhance button click - only enhances, no generation
            enhance_btn.click(
                fn=enhance_only,
                inputs=[prompt],
                outputs=[enhanced_prompt_output, status, enhanced_actions]
            )

            # Generate with enhanced prompt
            generate_enhanced_btn.click(
                fn=generate_image,
                inputs=[enhanced_prompt_output, seed, steps, aspect_ratio, num_images, gallery_state],
                outputs=[output_image, status, gallery]
            )

            # Copy enhanced prompt to main prompt field
            copy_prompt_btn.click(
                fn=copy_to_prompt,
                inputs=[enhanced_prompt_output],
                outputs=[prompt, enhanced_actions, enhanced_prompt_output]
            )

        # ==================
        # VIDEO TAB
        # ==================
        with gr.TabItem("🎬 Video Generation (WAN 2.2)"):
            gr.Markdown("*Generate videos with WAN 2.2 - Takes ~15-20 minutes for 480p*")

            with gr.Row():
                with gr.Column(scale=1):
                    video_mode = gr.Radio(
                        label="Mode",
                        choices=["Text to Video", "Image to Video"],
                        value="Text to Video"
                    )

                    video_prompt = gr.Textbox(
                        label="Video Prompt",
                        placeholder="Describe the video you want to generate...",
                        lines=3,
                        max_lines=6
                    )

                    with gr.Group(visible=False) as input_image_group:
                        video_input_image = gr.Image(
                            label="Input Image (for Image-to-Video)",
                            type="filepath",
                            height=200
                        )

                    with gr.Accordion("Model Selection", open=True):
                        refresh_models_btn = gr.Button("🔄 Refresh Model List", variant="secondary", size="sm")

                        with gr.Row():
                            video_model_high = gr.Dropdown(
                                label="High Noise Model",
                                choices=[],
                                value=None,
                                info="First pass model"
                            )
                            video_model_low = gr.Dropdown(
                                label="Low Noise Model",
                                choices=[],
                                value=None,
                                info="Second pass model"
                            )

                        with gr.Row():
                            video_lora_high = gr.Dropdown(
                                label="High Noise LoRA",
                                choices=[],
                                value=None,
                                info="Lightning LoRA for speed"
                            )
                            video_lora_low = gr.Dropdown(
                                label="Low Noise LoRA",
                                choices=[],
                                value=None,
                                info="Lightning LoRA for speed"
                            )

                    with gr.Row():
                        video_seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="Use -1 for random"
                        )
                        video_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=list(VIDEO_RESOLUTIONS.keys()),
                            value="480p (848x480)",
                            info="480p recommended for 12GB VRAM"
                        )

                    video_frames = gr.Slider(
                        label="Frames",
                        minimum=49,
                        maximum=121,
                        value=81,
                        step=16,
                        info="81 frames = ~5 seconds at 16fps"
                    )

                    generate_video_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")

                with gr.Column(scale=1):
                    output_video = gr.Video(
                        label="Generated Video",
                        height=400
                    )

                    video_status = gr.Textbox(label="Status", interactive=False)

            # Video mode change handler
            video_mode.change(
                fn=update_video_mode,
                inputs=[video_mode],
                outputs=[input_image_group]
            )

            # Refresh models button
            refresh_models_btn.click(
                fn=refresh_video_models,
                inputs=[],
                outputs=[video_model_high, video_model_low, video_lora_high, video_lora_low]
            )

            # Generate video button
            generate_video_btn.click(
                fn=generate_video,
                inputs=[
                    video_mode,
                    video_prompt,
                    video_seed,
                    video_resolution,
                    video_frames,
                    video_model_high,
                    video_model_low,
                    video_lora_high,
                    video_lora_low,
                    video_input_image
                ],
                outputs=[output_video, video_status]
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
