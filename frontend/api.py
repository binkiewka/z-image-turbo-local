"""
FastAPI Backend for Z-Image-Turbo & WAN 2.2 Video

REST API + WebSocket backend for real-time generation progress.
"""

import os
import json
import uuid
import random
import asyncio
import time
import gc
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
import httpx
import websockets
from PIL import Image

from prompt_enhancer import enhance_prompt, enhance_video_prompt, get_enhancer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ComfyUI connection settings
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
COMFYUI_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_WS = f"ws://{COMFYUI_HOST}:{COMFYUI_PORT}/ws"

# Aspect ratio presets
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}

# Video resolution presets
VIDEO_RESOLUTIONS = {
    "480p": (848, 480),
    "720p": (1280, 720),
}

# Directories
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp/zimage_downloads"))
DOWNLOAD_DIR.mkdir(exist_ok=True)
VIDEO_OUTPUT_DIR = Path("/app/output")

# Load workflow templates
with open("workflow_api.json", "r") as f:
    WORKFLOW_TEMPLATE = json.load(f)

with open("workflow_video_i2v.json", "r") as f:
    WORKFLOW_VIDEO_I2V = json.load(f)

# In-memory gallery storage (last 12 images)
gallery_history = []


def load_gallery_from_disk():
    """Load recent images from disk into gallery history."""
    global gallery_history
    try:
        # Find all PNG files in download dir
        images = []
        for file_path in DOWNLOAD_DIR.glob("*.png"):
            if file_path.name.startswith("zimage_") or file_path.name.startswith("z_image_"):
                images.append(file_path)
        
        # Sort by modification time (newest first)
        images.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Load last 12
        for img_path in images[:12]:
            try:
                gallery_history.append({
                    "url": f"/api/download/{img_path.name}",
                    "filename": str(img_path)
                })
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")
                
        logger.info(f"Loaded {len(gallery_history)} images from disk")
    except Exception as e:
        logger.error(f"Failed to load gallery from disk: {e}")


# =============================================================================
# Pydantic Models
# =============================================================================

class LoraConfig(BaseModel):
    name: str
    strength: float = 1.0


class ImageGenerateRequest(BaseModel):
    prompt: str
    seed: int = -1
    steps: int = 8
    aspect_ratio: str = "1:1"
    num_images: int = 1
    loras: list[LoraConfig] = []
    upscale_enabled: bool = False
    upscale_model: Optional[str] = None


class VideoGenerateRequest(BaseModel):
    mode: str = "i2v"  # I2V only (T2V removed)
    prompt: str
    seed: int = -1
    resolution: str = "480p"
    frames: int = 81
    model_high: str
    model_low: str
    lora_high: str
    lora_low: str
    input_image: Optional[str] = None
    upscale_enabled: bool = False
    upscale_model: Optional[str] = None
    interpolate_enabled: bool = False
    interpolate_model: Optional[str] = None
    interpolate_multiplier: int = 2


class EnhanceRequest(BaseModel):
    prompt: str


# =============================================================================
# Helper Functions
# =============================================================================

def prepare_workflow(prompt: str, seed: int, steps: int, width: int, height: int, num_images: int = 1, loras: list[LoraConfig] = [], upscale_enabled: bool = False, upscale_model: str = None) -> tuple[dict, int]:
    """Prepare the image workflow with user inputs."""
    workflow = json.loads(json.dumps(WORKFLOW_TEMPLATE))

    workflow["8"]["inputs"]["text"] = prompt
    workflow["3"]["inputs"]["text"] = ""  # Negative prompt (unused)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    workflow["10"]["inputs"]["seed"] = seed
    workflow["10"]["inputs"]["steps"] = steps
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height
    workflow["5"]["inputs"]["batch_size"] = num_images

    # --- LoRA Injection ---
    current_model = ["11", 0] # Default from ModelSamplingAuraFlow
    current_clip = ["7", 0]
    
    # Checkpoint -> (Lora) -> ModelSampling
    # The template has: 
    # 9 (Unet) -> 11 (ModelSampling) -> 10 (KSampler) [model]
    # 7 (Clip) -> 8/3 (TextEncode) -> 10 (KSampler) [positive/negative]
    
    # If LoRA:
    # 9 (Unet) -> Lora (model) -> 11 (ModelSampling)
    # 7 (Clip) -> Lora (clip) -> 8/3 (TextEncode)
    
    # If LoRA:
    # 9 (Unet) -> Lora (model) -> 11 (ModelSampling)
    # 7 (Clip) -> Lora (clip) -> 8/3 (TextEncode)
    
    if loras:
        last_model_node = ["9", 0] # Start with Unet Loader
        last_clip_node = ["7", 0]  # Start with Clip Loader
        
        for i, lora_config in enumerate(loras):
            if not lora_config.name: continue
            
            node_id = f"100_{i}"
            workflow[node_id] = {
                "inputs": {
                    "lora_name": lora_config.name,
                    "strength_model": lora_config.strength,
                    "strength_clip": lora_config.strength,
                    "model": last_model_node,
                    "clip": last_clip_node
                },
                "class_type": "LoraLoader",
                "_meta": {"title": f"Load LoRA {i+1}"}
            }
            # Update pointers for next iteration
            last_model_node = [node_id, 0]
            last_clip_node = [node_id, 1]
            
        # Update final connections
        workflow["11"]["inputs"]["model"] = last_model_node 
        workflow["8"]["inputs"]["clip"] = last_clip_node
        workflow["3"]["inputs"]["clip"] = last_clip_node

    # --- Upscaling ---
    current_image_node = ["6", 0] # VAE Decode output

    if upscale_enabled and upscale_model:
        workflow["101"] = {
            "inputs": {"model_name": upscale_model},
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"}
        }
        workflow["102"] = {
            "inputs": {
                "upscale_model": ["101", 0],
                "image": current_image_node
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "Upscale Image"}
        }
        current_image_node = ["102", 0]
        
    # Final Save
    workflow["18"]["inputs"]["images"] = current_image_node

    return workflow, seed


def prepare_video_workflow(
    prompt: str,
    seed: int,
    width: int,
    height: int,
    frames: int,
    model_high: str,
    model_low: str,
    lora_high: str,
    lora_low: str,
    input_image: str,
    upscale_enabled: bool = False,
    upscale_model: str = None,
    interpolate_enabled: bool = False,
    interpolate_model: str = None,
    interpolate_multiplier: int = 2
) -> tuple[dict, int]:
    """Prepare the I2V video workflow with user inputs."""
    workflow = json.loads(json.dumps(WORKFLOW_VIDEO_I2V))

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Update models
    workflow["2"]["inputs"]["unet_name"] = model_high
    workflow["3"]["inputs"]["unet_name"] = model_low
    workflow["4"]["inputs"]["lora_name"] = lora_high
    workflow["5"]["inputs"]["lora_name"] = lora_low
    workflow["8"]["inputs"]["text"] = prompt

    # Update resolution/frames/seed (I2V node IDs)
    workflow["12"]["inputs"]["width"] = width
    workflow["12"]["inputs"]["height"] = height
    workflow["12"]["inputs"]["length"] = frames
    workflow["13"]["inputs"]["noise_seed"] = seed
    workflow["14"]["inputs"]["noise_seed"] = seed
    workflow["10"]["inputs"]["image"] = input_image

    # Enhancement nodes (I2V: VAE decode is node 15)
    current_image_node = ["15", 0]

    if upscale_enabled and upscale_model:
        workflow["100"] = {
            "inputs": {"model_name": upscale_model},
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"}
        }
        workflow["101"] = {
            "inputs": {
                "upscale_model": ["100", 0],
                "image": current_image_node
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "Upscale Image"}
        }
        current_image_node = ["101", 0]

    if interpolate_enabled and interpolate_model:
        workflow["102"] = {
            "inputs": {
                "ckpt_name": interpolate_model,
                "clear_cache_after_n_frames": 10,
                "multiplier": int(interpolate_multiplier),
                "fast_mode": True,
                "ensemble": True,
                "scale_factor": 1.0,
                "frames": current_image_node
            },
            "class_type": "RIFE VFI",
            "_meta": {"title": "RIFE Frame Interpolation"}
        }
        current_image_node = ["102", 0]

    combine_node_id = "16"  # I2V video combine node

    if interpolate_enabled and interpolate_model:
        base_fps = 16
        new_fps = base_fps * interpolate_multiplier
        workflow[combine_node_id]["inputs"]["frame_rate"] = new_fps

    workflow[combine_node_id]["inputs"]["images"] = current_image_node

    # Extract last frame
    workflow["103"] = {
        "inputs": {
            "images": current_image_node,
            "start_index": -1,
            "num_frames": 1
        },
        "class_type": "GetImageRangeFromBatch",
        "_meta": {"title": "Get Last Frame"}
    }
    workflow["104"] = {
        "inputs": {
            "filename_prefix": "wan2_lastframe",
            "images": ["103", 0],
            "save_output": True
        },
        "class_type": "SaveImage",
        "_meta": {"title": "Save Last Frame"}
    }

    return workflow, seed


async def get_history(prompt_id: str) -> list[tuple[str, str]] | None:
    """Get output image info from ComfyUI history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    images_info = []
                    for img in node_output["images"]:
                        images_info.append((img["filename"], img.get("subfolder", "")))
                    return images_info if images_info else None
    return None


async def get_video_history(prompt_id: str) -> tuple[str, str] | None:
    """Get output video info from ComfyUI history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                if "gifs" in node_output:
                    for vid in node_output["gifs"]:
                        return (vid["filename"], vid.get("subfolder", ""))
                if "images" in node_output:
                    for img in node_output["images"]:
                        if img["filename"].endswith((".mp4", ".webm", ".gif")):
                            return (img["filename"], img.get("subfolder", ""))
    return None


async def fetch_image(filename: str, subfolder: str) -> Image.Image:
    """Fetch generated image from ComfyUI."""
    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{COMFYUI_URL}/view", params=params)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))


async def fetch_video(filename: str, subfolder: str) -> bytes:
    """Fetch generated video from ComfyUI."""
    params = {"filename": filename, "type": "output"}
    if subfolder:
        params["subfolder"] = subfolder

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(f"{COMFYUI_URL}/view", params=params)
        response.raise_for_status()
        return response.content


async def free_comfyui_memory():
    """Free ComfyUI GPU/RAM memory."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True}
            )
            if response.status_code == 200:
                logger.info("ComfyUI memory freed successfully")
    except Exception as e:
        logger.warning(f"Failed to free ComfyUI memory: {e}")
    gc.collect()


def scan_video_models() -> dict:
    """Scan for available WAN 2.2 I2V video models."""
    models = {
        "i2v_high": [], "i2v_low": [],
        "loras_high": [], "loras_low": [],
    }

    try:
        response = httpx.get(f"{COMFYUI_URL}/object_info/UnetLoaderGGUF", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if "UnetLoaderGGUF" in data:
                all_models = data["UnetLoaderGGUF"]["input"]["required"]["unet_name"][0]
                for model in all_models:
                    model_lower = model.lower()
                    if ("wan" in model_lower or "wan2" in model_lower) and "i2v" in model_lower:
                        if "high" in model_lower:
                            models["i2v_high"].append(model)
                        elif "low" in model_lower:
                            models["i2v_low"].append(model)

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
        logger.error(f"Error scanning models: {e}")

    return models


def scan_image_models() -> dict:
    """Scan for Image LoRAs (in image/ folder)."""
    models = {
        "loras": []
    }
    try:
        response = httpx.get(f"{COMFYUI_URL}/object_info/LoraLoader", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            if "LoraLoader" in data:
                all_loras = data["LoraLoader"]["input"]["required"]["lora_name"][0]
                for lora in all_loras:
                    # Filter for LoRAs in 'image/' directory or standard/root ones if needed.
                    # For now only taking those explicitly in image/ folder as requested.
                    if "image/" in lora.lower() or "image\\" in lora.lower():
                         models["loras"].append(lora)
    except Exception as e:
         logger.error(f"Error scanning image models: {e}")
         
    return models


def scan_enhancement_models() -> dict:
    """Scan for RIFE VFI and Upscale models."""
    models = {"vfi": [], "upscale": []}

    try:
        # Scan RIFE VFI models
        response = httpx.get(f"{COMFYUI_URL}/object_info/RIFE VFI", timeout=5.0)
        # logger.debug(f"RIFE VFI response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "RIFE VFI" in data:
                input_req = data["RIFE VFI"]["input"]["required"]
                if "ckpt_name" in input_req:
                    ckpt_list = input_req["ckpt_name"][0]
                    # logger.debug(f"VFI ckpt_list type: {type(ckpt_list)}, value: {ckpt_list}")
                    if isinstance(ckpt_list, list):
                        models["vfi"] = ckpt_list
                    elif isinstance(ckpt_list, str):
                        models["vfi"] = [ckpt_list]

        # Scan Upscale models
        response = httpx.get(f"{COMFYUI_URL}/object_info/UpscaleModelLoader", timeout=5.0)
        # logger.debug(f"UpscaleModelLoader response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            # logger.debug(f"UpscaleModelLoader full data: {data}")
            if "UpscaleModelLoader" in data:
                input_req = data["UpscaleModelLoader"]["input"]["required"]
                # logger.debug(f"UpscaleModelLoader input_req: {input_req}")
                if "model_name" in input_req:
                    model_info = input_req["model_name"]
                    # logger.debug(f"model_name info: {model_info}")
                    # Handle COMBO format: ['COMBO', {'multiselect': False, 'options': [...]}]
                    if isinstance(model_info, list) and len(model_info) > 0:
                        if model_info[0] == 'COMBO' and len(model_info) > 1 and isinstance(model_info[1], dict):
                            # New ComfyUI format with COMBO type indicator
                            options = model_info[1].get('options', [])
                            if isinstance(options, list):
                                models["upscale"] = options
                            # logger.debug(f"Extracted from COMBO options: {models['upscale']}")
                        elif isinstance(model_info[0], list):
                            # Old format: direct list of models
                            models["upscale"] = model_info[0]
                        elif isinstance(model_info[0], str) and model_info[0] != 'COMBO':
                            # Single model as string
                            models["upscale"] = [model_info[0]]
                    elif isinstance(model_info, str):
                        models["upscale"] = [model_info]

        logger.info(f"Final enhancement models: upscale={models['upscale']}, vfi={models['vfi']}")
    except Exception as e:
        logger.error(f"Error scanning enhancement models: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return models


# =============================================================================
# Lifespan for startup/shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Z-Image-Turbo API...")
    
    # Load gallery from disk
    load_gallery_from_disk()
    
    logger.info("Warming up models...")

    # Warmup by running a test generation
    # Warmup by running a test generation
    try:
        # Use small size and 1 step for fast warmup
        workflow, _ = prepare_workflow("warmup test", 42, 1, 512, 512, 1)
        
        # Remove SaveImage node to prevent writing to disk
        if "18" in workflow:
            del workflow["18"]
            
        # Add PreviewImage node to ensure execution (ID 19)
        workflow["19"] = {
            "inputs": {"images": ["6", 0]},
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        }
        client_id = str(uuid.uuid4())

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            response = await http_client.post(
                f"{COMFYUI_URL}/prompt",
                json={"prompt": workflow, "client_id": client_id}
            )
            if response.status_code == 200:
                prompt_id = response.json()["prompt_id"]
                # Wait for completion
                for _ in range(60):
                    await asyncio.sleep(1)
                    history = await get_history(prompt_id)
                    if history:
                        logger.info("Models loaded successfully!")
                        break
    except Exception as e:
        logger.warning(f"Warmup failed (models will load on first request): {e}")

    logger.info("Prompt enhancer will load on first use")
    logger.info("API ready!")

    yield

    logger.info("Shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Z-Image-Turbo API",
    description="AI Image & Video Generation API",
    version="2.0.0",
    lifespan=lifespan
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "comfyui_url": COMFYUI_URL}


@app.get("/api/models")
async def get_models():
    """Get all available models."""
    video_models = scan_video_models()
    enhancement_models = scan_enhancement_models()
    image_models = scan_image_models()

    all_high = sorted(video_models["i2v_high"])
    all_low = sorted(video_models["i2v_low"])

    return {
        "video": {
            "high": all_high,
            "low": all_low,
            "loras_high": sorted(video_models["loras_high"]),
            "loras_low": sorted(video_models["loras_low"]),
        },
        "enhancement": {
            "upscale": sorted(enhancement_models["upscale"]),
            "vfi": sorted(enhancement_models["vfi"]),
        },
        "image": {
            "aspect_ratios": list(ASPECT_RATIOS.keys()),
            "loras": sorted(image_models["loras"]),
        }
    }


@app.post("/api/enhance")
async def enhance_prompt_endpoint(request: EnhanceRequest):
    """Enhance a prompt using the LLM."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        loop = asyncio.get_running_loop()
        enhanced = await loop.run_in_executor(None, enhance_prompt, request.prompt.strip())
        return {"original": request.prompt, "enhanced": enhanced}
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/api/enhance-video")
async def enhance_video_prompt_endpoint(request: EnhanceRequest):
    """Enhance a video prompt for WAN 2.2 using the LLM."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        loop = asyncio.get_running_loop()
        enhanced = await loop.run_in_executor(None, enhance_video_prompt, request.prompt.strip())
        return {"original": request.prompt, "enhanced": enhanced}
    except Exception as e:
        logger.error(f"Video enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video enhancement failed: {str(e)}")


@app.post("/api/generate")
async def generate_image_endpoint(request: ImageGenerateRequest):
    """
    Start image generation. Returns a client_id to connect via WebSocket for progress.
    """
    global gallery_history

    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    width, height = ASPECT_RATIOS.get(request.aspect_ratio, (1024, 1024))
    workflow, actual_seed = prepare_workflow(
        request.prompt.strip(),
        request.seed,
        request.steps,
        width,
        height,

        request.num_images,
        request.loras,
        request.upscale_enabled,
        request.upscale_model
    )
    client_id = str(uuid.uuid4())

    # Submit to ComfyUI
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        response = await http_client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        )
        response.raise_for_status()
        result = response.json()
        prompt_id = result["prompt_id"]

    return {
        "client_id": client_id,
        "prompt_id": prompt_id,
        "seed": actual_seed,
        "width": width,
        "height": height
    }


@app.post("/api/generate-video")
async def generate_video_endpoint(request: VideoGenerateRequest):
    """Start video generation. Returns client_id for WebSocket progress."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    if not request.model_high or not request.model_low:
        raise HTTPException(status_code=400, detail="Both high and low noise models required")

    if not request.lora_high or not request.lora_low:
        raise HTTPException(status_code=400, detail="Both high and low noise LoRAs required")

    if not request.input_image:
        raise HTTPException(status_code=400, detail="Input image required for video generation")

    width, height = VIDEO_RESOLUTIONS.get(request.resolution, (848, 480))

    workflow, actual_seed = prepare_video_workflow(
        prompt=request.prompt.strip(),
        seed=request.seed,
        width=width,
        height=height,
        frames=request.frames,
        model_high=request.model_high,
        model_low=request.model_low,
        lora_high=request.lora_high,
        lora_low=request.lora_low,
        input_image=request.input_image,
        upscale_enabled=request.upscale_enabled,
        upscale_model=request.upscale_model,
        interpolate_enabled=request.interpolate_enabled,
        interpolate_model=request.interpolate_model,
        interpolate_multiplier=request.interpolate_multiplier
    )
    client_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        response = await http_client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id}
        )
        response.raise_for_status()
        result = response.json()
        prompt_id = result["prompt_id"]

    return {
        "client_id": client_id,
        "prompt_id": prompt_id,
        "seed": actual_seed,
        "width": width,
        "height": height
    }


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to ComfyUI for I2V generation."""
    try:
        content = await file.read()
        files = {"image": (file.filename, content, file.content_type or "image/png")}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{COMFYUI_URL}/upload/image", files=files)
            response.raise_for_status()
            result = response.json()
            return {"filename": result.get("name", file.filename)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/result/{prompt_id}")
async def get_result(prompt_id: str, type: str = "image"):
    """Get generation result (image or video)."""
    global gallery_history

    try:
        if type == "image":
            result = await get_history(prompt_id)
            if result:
                images = []
                for i, (filename, subfolder) in enumerate(result):
                    # Check if file already exists in output (shared volume)
                    download_path = DOWNLOAD_DIR / filename
                    
                    if not download_path.exists():
                        # Fetch and save if not found (e.g. remote ComfyUI)
                        image = await fetch_image(filename, subfolder)
                        
                        # Use original filename if possible, else timestamp
                        # But here we try to match what ComfyUI produced
                        image.save(download_path)
                    
                    img_url = f"/api/download/{download_path.name}"

                    images.append({
                        "url": img_url,
                        "filename": str(download_path),
                        "index": i
                    })

                    # Add to gallery
                    gallery_history.insert(0, {
                        "url": img_url,
                        "filename": str(download_path)
                    })

                gallery_history = gallery_history[:12]
                return {"images": images, "count": len(images)}
        else:
            result = await get_video_history(prompt_id)
            if result:
                filename, subfolder = result
                video_path = DOWNLOAD_DIR / filename

                if video_path.exists():
                    # File already exists (shared volume), read it
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                else:
                    # Fetch from ComfyUI and save with ORIGINAL filename
                    video_bytes = await fetch_video(filename, subfolder)
                    with open(video_path, "wb") as f:
                        f.write(video_bytes)

                # Free memory after video
                await free_comfyui_memory()

                import base64
                video_base64 = base64.b64encode(video_bytes).decode()
                return {"video": video_base64, "filename": str(video_path)}

        return {"error": "No result found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/gallery")
async def get_gallery():
    """Get recent generations gallery."""
    # Ensure all items have a valid URL
    for item in gallery_history:
        if "url" not in item:
            filename = Path(item["filename"]).name
            item["url"] = f"/api/download/{filename}"
    return {"images": gallery_history[:12]}


@app.delete("/api/gallery/{filename}")
async def delete_image(filename: str):
    """Delete an image from gallery and disk."""
    global gallery_history
    
    # Sanitize filename (basic check)
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = DOWNLOAD_DIR / filename
    
    # Remove from disk
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file")

    # Remove from memory
    gallery_history = [img for img in gallery_history if Path(img["filename"]).name != filename]
    
    return {"status": "deleted", "filename": filename}


@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    """Download a generated file."""
    # Security check: ensure filename is just a name, not a path
    if "/" in filename or ".." in filename:
         # unless it's a relative path from download dir? 
         # The 'path' type param allows slashes. 
         # But we want to restrict to DOWNLOAD_DIR.
         # So we should be careful. 
         # Ideally we just take the basename if we flat structure.
         pass
         
    file_path = DOWNLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# =============================================================================
# WebSocket for Progress Updates
# =============================================================================

# Mapping of Node IDs to user-friendly status messages
NODE_ID_MAPPINGS = {
    "1": "Loading CLIP model...",
    "2": "Loading High Noise Model...",
    "3": "Loading Low Noise Model...",
    "4": "Loading High Noise LoRA...",
    "5": "Loading Low Noise LoRA...",
    "8": "Encoding prompts...",
    "9": "Encoding prompts...",
    "10": "Processing input image...",
    "11": "Loading VAE...",
    "12": "Preparing video latents...",
    "13": "Generating video (High Noise Pass)...",
    "14": "Refining video (Low Noise Pass)...",
    "15": "Decoding video frames...",
    "16": "Saving video...",
    "100": "Loading Upscale Model...",
    "101": "Upscaling frames...",
    "102": "Interpolating frames (RIFE)...",
    "103": "Extracting last frame...",
    "104": "Saving last frame..."
}

@app.websocket("/ws/{client_id}")
async def websocket_progress(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint that proxies ComfyUI progress updates.
    Client connects here after starting generation to receive progress.
    """
    await websocket.accept()

    retries = 3
    for attempt in range(retries):
        try:
            async with websockets.connect(
                f"{COMFYUI_WS}?clientId={client_id}",
                ping_interval=20,
                ping_timeout=60
            ) as comfy_ws:
                async for message in comfy_ws:
                    if isinstance(message, str):
                        data = json.loads(message)
                        msg_type = data.get("type")

                        # Forward progress updates
                        if msg_type == "progress":
                            prog_data = data.get("data", {})
                            await websocket.send_json({
                                "type": "progress",
                                "value": prog_data.get("value", 0),
                                "max": prog_data.get("max", 100)
                            })

                        # Forward completion or status updates
                        elif msg_type == "executing":
                            exec_data = data.get("data", {})
                            node_id = exec_data.get("node")

                            if node_id is None:
                                # Execution finished
                                await websocket.send_json({
                                    "type": "complete",
                                    "prompt_id": exec_data.get("prompt_id")
                                })
                                return
                            else:
                                # New node started executing
                                status_msg = NODE_ID_MAPPINGS.get(str(node_id))
                                if status_msg:
                                    await websocket.send_json({
                                        "type": "status",
                                        "message": status_msg,
                                        "node": node_id
                                    })

                        # Forward errors
                        elif msg_type == "execution_error":
                            await websocket.send_json({
                                "type": "error",
                                "data": data.get("data", {})
                            })
                            return

        except websockets.exceptions.ConnectionClosed:
            if attempt < retries - 1:
                await asyncio.sleep(2)
                continue
            await websocket.send_json({"type": "error", "message": "Connection lost"})
        except WebSocketDisconnect:
            return
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})
            return


# =============================================================================
# Static Files (Serve frontend)
# =============================================================================

# Mount static files LAST so API routes take precedence
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
