import os
import json
import uuid
import random
import asyncio
import time
from io import BytesIO
from pathlib import Path

import gradio as gr
import httpx
import websockets
from PIL import Image
from prompt_enhancer import enhance_prompt

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
# Ensure download directory exists
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "/tmp/zimage_downloads"))
DOWNLOAD_DIR.mkdir(exist_ok=True)

with open("workflow_api.json", "r") as f:
    WORKFLOW_TEMPLATE = json.load(f)

# Global gallery state
gallery_history = []


def prepare_workflow(prompt: str, seed: int, steps: int, width: int, height: int) -> tuple[dict, int]:
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

    # Resolution
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height

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


async def get_history(prompt_id: str) -> tuple[str, str] | None:
    """Get the output image info from history."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{COMFYUI_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json()

        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        return img["filename"], img.get("subfolder", "")
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


def generate_image(prompt: str, seed: int, steps: int, aspect_ratio: str, current_gallery: list, progress=gr.Progress()):
    """Main generation function with progress tracking."""
    global gallery_history

    if not prompt or not prompt.strip():
        return None, "Please enter a prompt", None, current_gallery

    actual_prompt = prompt.strip()
    width, height = ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
    workflow, actual_seed = prepare_workflow(
        actual_prompt,
        int(seed),
        int(steps),
        width,
        height
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
            filename, subfolder = result
            image = asyncio.run(fetch_image(filename, subfolder))

            # Save for download
            download_path = DOWNLOAD_DIR / f"zimage_{actual_seed}_{int(time.time())}.png"
            image.save(download_path)

            # Update gallery
            gallery_entry = (image, f"Seed: {actual_seed}")
            gallery_history.insert(0, gallery_entry)
            gallery_history = gallery_history[:12]  # Keep last 12

            return image, f"Done! Seed: {actual_seed}", str(download_path), gallery_history
        else:
            return None, "No image generated", None, current_gallery

    except Exception as e:
        return None, f"Error: {str(e)}", None, current_gallery


def enhance_only(prompt, progress=gr.Progress()):
    """Enhance prompt without generating image."""
    if not prompt or not prompt.strip():
        return "", "Please enter a prompt first", gr.update(visible=False)

    progress(0, desc="Enhancing prompt...")
    try:
        def status_callback(msg):
            progress(0.5, desc=msg)
        enhanced = enhance_prompt(prompt.strip(), status_callback)
        progress(1, desc="Prompt enhanced!")
        return enhanced, "✨ Prompt enhanced! Click 'Generate with Enhanced' or 'Copy to Prompt'", gr.update(visible=True)
    except Exception as e:
        return "", f"Enhancement failed: {e}", gr.update(visible=False)


def copy_to_prompt(enhanced_text):
    """Copy enhanced prompt to main prompt field."""
    return enhanced_text, gr.update(visible=False), ""


def warmup_models():
    """Pre-load models by running a dummy generation at startup."""
    print("Warming up models... (this may take 10-30 seconds on first run)")
    try:
        result = generate_image("warmup test", 42, 8, "1:1 (1024x1024)", [], gr.Progress())
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


with gr.Blocks(title="Z-Image-Turbo (Local)", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Z-Image-Turbo (Local)")
    gr.Markdown("Generate images locally using Z-Image-Turbo on RTX 3060")

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

            steps = gr.Slider(
                label="Steps",
                minimum=4,
                maximum=12,
                value=8,
                step=1,
                info="More steps = better quality, slower"
            )

            with gr.Row():
                generate_btn = gr.Button("🎨 Generate", variant="primary", size="lg")
                enhance_btn = gr.Button("✨ Enhance", variant="secondary", size="lg")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Generated Image",
                type="pil",
                height=512
            )

            download_file = gr.File(
                label="Download",
                visible=True,
                interactive=False
            )

    # Gallery section
    gr.Markdown("### History")
    gallery = gr.Gallery(
        label="Recent Generations",
        show_label=False,
        columns=4,
        object_fit="contain",
        allow_preview=True
    )

    # Hidden state for gallery
    gallery_state = gr.State([])

    # Generate button click - uses original prompt
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, seed, steps, aspect_ratio, gallery_state],
        outputs=[output_image, status, download_file, gallery]
    )

    # Enter key in prompt - uses original prompt
    prompt.submit(
        fn=generate_image,
        inputs=[prompt, seed, steps, aspect_ratio, gallery_state],
        outputs=[output_image, status, download_file, gallery]
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
        inputs=[enhanced_prompt_output, seed, steps, aspect_ratio, gallery_state],
        outputs=[output_image, status, download_file, gallery]
    )

    # Copy enhanced prompt to main prompt field
    copy_prompt_btn.click(
        fn=copy_to_prompt,
        inputs=[enhanced_prompt_output],
        outputs=[prompt, enhanced_actions, enhanced_prompt_output]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
