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

from prompt_enhancer import enhance_prompt, get_enhancer

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

with open("workflow_video_t2v.json", "r") as f:
    WORKFLOW_VIDEO_T2V = json.load(f)

with open("workflow_video_i2v.json", "r") as f:
    WORKFLOW_VIDEO_I2V = json.load(f)

# In-memory gallery storage (last 12 images)
gallery_history = []


# =============================================================================
# Pydantic Models
# =============================================================================

class ImageGenerateRequest(BaseModel):
    prompt: str
    seed: int = -1
    steps: int = 8
    aspect_ratio: str = "1:1"
    num_images: int = 1


class VideoGenerateRequest(BaseModel):
    mode: str = "t2v"  # "t2v" or "i2v"
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

def prepare_workflow(prompt: str, seed: int, steps: int, width: int, height: int, num_images: int = 1) -> tuple[dict, int]:
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

    return workflow, seed


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
    input_image: str = None,
    upscale_enabled: bool = False,
    upscale_model: str = None,
    interpolate_enabled: bool = False,
    interpolate_model: str = None,
    interpolate_multiplier: int = 2
) -> tuple[dict, int]:
    """Prepare the video workflow with user inputs."""
    if mode == "t2v":
        workflow = json.loads(json.dumps(WORKFLOW_VIDEO_T2V))
    else:
        workflow = json.loads(json.dumps(WORKFLOW_VIDEO_I2V))

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Update models
    workflow["2"]["inputs"]["unet_name"] = model_high
    workflow["3"]["inputs"]["unet_name"] = model_low
    workflow["4"]["inputs"]["lora_name"] = lora_high
    workflow["5"]["inputs"]["lora_name"] = lora_low
    workflow["8"]["inputs"]["text"] = prompt

    # Update resolution/frames/seed
    if mode == "t2v":
        workflow["10"]["inputs"]["width"] = width
        workflow["10"]["inputs"]["height"] = height
        workflow["10"]["inputs"]["length"] = frames
        workflow["11"]["inputs"]["noise_seed"] = seed
        workflow["12"]["inputs"]["noise_seed"] = seed
    else:
        workflow["12"]["inputs"]["width"] = width
        workflow["12"]["inputs"]["height"] = height
        workflow["12"]["inputs"]["length"] = frames
        workflow["13"]["inputs"]["noise_seed"] = seed
        workflow["14"]["inputs"]["noise_seed"] = seed
        if input_image:
            workflow["10"]["inputs"]["image"] = input_image

    # Enhancement nodes
    vae_decode_id = "14" if mode == "t2v" else "15"
    current_image_node = [vae_decode_id, 0]

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

    combine_node_id = "15" if mode == "t2v" else "16"

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
                print("ComfyUI memory freed successfully")
    except Exception as e:
        print(f"Warning: Failed to free ComfyUI memory: {e}")
    gc.collect()


def scan_video_models() -> dict:
    """Scan for available WAN 2.2 video models."""
    models = {
        "t2v_high": [], "t2v_low": [],
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


def scan_enhancement_models() -> dict:
    """Scan for RIFE VFI and Upscale models."""
    models = {"vfi": [], "upscale": []}

    try:
        # Scan RIFE VFI models
        response = httpx.get(f"{COMFYUI_URL}/object_info/RIFE VFI", timeout=5.0)
        print(f"[DEBUG] RIFE VFI response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if "RIFE VFI" in data:
                input_req = data["RIFE VFI"]["input"]["required"]
                if "ckpt_name" in input_req:
                    ckpt_list = input_req["ckpt_name"][0]
                    print(f"[DEBUG] VFI ckpt_list type: {type(ckpt_list)}, value: {ckpt_list}")
                    if isinstance(ckpt_list, list):
                        models["vfi"] = ckpt_list
                    elif isinstance(ckpt_list, str):
                        models["vfi"] = [ckpt_list]

        # Scan Upscale models
        response = httpx.get(f"{COMFYUI_URL}/object_info/UpscaleModelLoader", timeout=5.0)
        print(f"[DEBUG] UpscaleModelLoader response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[DEBUG] UpscaleModelLoader full data: {data}")
            if "UpscaleModelLoader" in data:
                input_req = data["UpscaleModelLoader"]["input"]["required"]
                print(f"[DEBUG] UpscaleModelLoader input_req: {input_req}")
                if "model_name" in input_req:
                    model_info = input_req["model_name"]
                    print(f"[DEBUG] model_name info: {model_info}")
                    # Handle COMBO format: ['COMBO', {'multiselect': False, 'options': [...]}]
                    if isinstance(model_info, list) and len(model_info) > 0:
                        if model_info[0] == 'COMBO' and len(model_info) > 1 and isinstance(model_info[1], dict):
                            # New ComfyUI format with COMBO type indicator
                            options = model_info[1].get('options', [])
                            if isinstance(options, list):
                                models["upscale"] = options
                            print(f"[DEBUG] Extracted from COMBO options: {models['upscale']}")
                        elif isinstance(model_info[0], list):
                            # Old format: direct list of models
                            models["upscale"] = model_info[0]
                        elif isinstance(model_info[0], str) and model_info[0] != 'COMBO':
                            # Single model as string
                            models["upscale"] = [model_info[0]]
                    elif isinstance(model_info, str):
                        models["upscale"] = [model_info]

        print(f"[DEBUG] Final enhancement models: upscale={models['upscale']}, vfi={models['vfi']}")
    except Exception as e:
        print(f"Error scanning enhancement models: {e}")
        import traceback
        traceback.print_exc()

    return models


# =============================================================================
# Lifespan for startup/shutdown
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("Starting Z-Image-Turbo API...")
    print("Warming up models...")

    # Warmup by running a test generation
    try:
        workflow, _ = prepare_workflow("warmup test", 42, 8, 1024, 1024, 1)
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
                        print("Models loaded successfully!")
                        break
    except Exception as e:
        print(f"Warmup failed (models will load on first request): {e}")

    print("Prompt enhancer will load on first use")
    print("API ready!")

    yield

    print("Shutting down...")


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

    all_high = sorted(set(video_models["t2v_high"] + video_models["i2v_high"]))
    all_low = sorted(set(video_models["t2v_low"] + video_models["i2v_low"]))

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
        }
    }


@app.post("/api/enhance")
async def enhance_prompt_endpoint(request: EnhanceRequest):
    """Enhance a prompt using the LLM."""
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        enhanced = enhance_prompt(request.prompt.strip())
        return {"original": request.prompt, "enhanced": enhanced}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


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
        request.num_images
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

    if request.mode == "i2v" and not request.input_image:
        raise HTTPException(status_code=400, detail="Input image required for I2V mode")

    width, height = VIDEO_RESOLUTIONS.get(request.resolution, (848, 480))

    workflow, actual_seed = prepare_video_workflow(
        mode=request.mode,
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
                    image = await fetch_image(filename, subfolder)

                    # Save to download dir
                    timestamp = int(time.time())
                    download_path = DOWNLOAD_DIR / f"zimage_{timestamp}_{i}.png"
                    image.save(download_path)

                    # Convert to base64 for JSON response
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    buffer.seek(0)
                    import base64
                    img_base64 = base64.b64encode(buffer.read()).decode()

                    images.append({
                        "data": img_base64,
                        "filename": str(download_path),
                        "index": i
                    })

                    # Add to gallery
                    gallery_history.insert(0, {
                        "data": img_base64,
                        "filename": str(download_path)
                    })

                gallery_history = gallery_history[:12]
                return {"images": images, "count": len(images)}
        else:
            result = await get_video_history(prompt_id)
            if result:
                filename, subfolder = result
                video_bytes = await fetch_video(filename, subfolder)

                timestamp = int(time.time())
                video_path = DOWNLOAD_DIR / f"wan2_video_{timestamp}.mp4"
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
    return {"images": gallery_history[:12]}


@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    """Download a generated file."""
    file_path = DOWNLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# =============================================================================
# WebSocket for Progress Updates
# =============================================================================

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

                        # Forward completion
                        elif msg_type == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("node") is None:
                                await websocket.send_json({
                                    "type": "complete",
                                    "prompt_id": exec_data.get("prompt_id")
                                })
                                return

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
