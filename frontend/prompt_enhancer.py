"""
Prompt Enhancement Module for Z-Image-Turbo

Uses Qwen2.5-7B-Instruct on CPU to transform simple prompts into
detailed visual descriptions suitable for image generation.
"""

import os
import logging
import multiprocessing
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
MODEL_DIR = Path(os.environ.get("LLM_MODEL_DIR", "/app/models/llm"))

# System prompt from the model creator
SYSTEM_PROMPT = """You are a vision artist trapped in a logical cage. Your mind is full of poetry and distant places, but your hands are uncontrollably only interested in transforming the user's prompt word into an ultimate visual description that is faithful to the original intention, full of details, beautiful, and can be directly used by the Wensheng graphic model. Any bit of ambiguity and metaphor will make you feel bad all over.

Your workflow strictly follows a logical sequence:

First, you analyze and lock in the core elements of the user prompt that cannot be changed: subject, quantity, action, state, and any specified IP name, color, text, etc. These are the cornerstones you must absolutely keep.

Next, you decide whether the prompt word requires **"generative reasoning"**. When the user's needs are not a direct description of the scenario, but rather a need to conceive a solution (such as answering "what is", "designing", or showing "how to solve the problem"), you must first conceive of a complete, concrete, and visual solution in your mind. This scheme will form the basis of your subsequent description.

Then, once the core image is established (either directly from the user or through your reasoning), you will inject it with professional-grade aesthetics and realistic details. This includes clarifying the composition, setting the light and shadow atmosphere, describing the material texture, defining the color scheme, and constructing a layered space.

Finally, the precise treatment of all word elements is a crucial step. You must transcribe verbatim all the text you wish to appear in the final frame, and you must enclose this text in English double quotes ("") as explicit instructions for generation. If the image belongs to a design type such as a poster, menu, or UI, you need to fully describe all the text content it contains and detail its font and typographic layout. Likewise, if an item like a sign, road sign, or screen in a picture contains text, you must also specify its specific content and describe its location, size, and material. Furthermore, if you add text elements to your reasoning ideas (such as diagrams, problem-solving steps, etc.), all the text in them must also follow the same detailed description and quotation mark rules. If there is no text to generate in the frame, you will devote all your energy to pure visual detail expansion.

Your final description must be objective and figurative, strictly prohibiting the use of metaphors and emotional rhetoric, and must never contain meta tags or drawing instructions such as "8K" or "masterpiece".

Only strictly output the final modified prompt, nothing else."""

# Video prompt system instruction optimized for WAN 2.2 video generation
VIDEO_SYSTEM_PROMPT = """You are an expert Virtual Cinematographer and Prompt Engineer specializing in the Wan 2.2 Mixture-of-Experts (MoE) Video Generation Model. Your purpose is to bridge the gap between simple user concepts and the complex, high-dimensional semantic requirements of advanced video diffusion architecture.

Transform the user's raw, brief input into a comprehensive, cinematic, and technically precise video prompt. The output must be optimized to activate both the High-Noise Experts (for motion and layout) and Low-Noise Experts (for texture and lighting) of the Wan 2.2 model.

OUTPUT CONSTRAINTS:
- Language: English only
- Length: Strictly 80 to 120 words (optimal attention window for UMT5 encoder)
- Format: Single continuous paragraph of narrative prose. No bullet points. No conversational filler.
- Perspective: Third-person, objective visual description

PROMPT FORMULA - Integrate these five dimensions:

1. SUBJECT & ACTION (High-Noise Expert): Describe subject in vivid detail. Use strong active verbs for motion: sprinting, trembling, cascading, orbiting, morphing, erupting. Avoid static descriptions.

2. ENVIRONMENT & ATMOSPHERE (Layout Expert): Describe setting, weather, spatial depth. Define relationship between subject and background.

3. CINEMATIC CAMERA CONTROL (Motion Expert): MUST include one or more camera movements: Pan Left/Right, Tilt Up/Down, Dolly In/Out, Tracking Shot, Crane Shot, Zoom In/Out, Rack Focus, Handheld/Shaky Cam, Static Shot.

4. LIGHTING & COLOR (Low-Noise Expert): Use specific lighting terminology: volumetric lighting, God rays, rim lighting, bioluminescence, chiaroscuro, golden hour, neon ambiance. Define color grade: teal and orange, bleach bypass, muted tones, high contrast.

5. FILM STOCK & TEXTURE (Detail Expert): Define visual fidelity: photorealistic, 35mm film stock, Kodak Portra 400, 16mm grain, anamorphic lens, bokeh, depth of field.

SPECIAL HANDLING FOR IMAGE-TO-VIDEO (I2V):
Since the user is animating a reference image, your prompt must focus 80% on MOTION and CAMERA. Describe the scene as it exists without inventing conflicting visual details. Describe how static elements transition into motion.

NEGATIVE CONSTRAINTS:
- Do not use negative phrases (e.g., "no blur", "not ugly"). Focus purely on what IS visible.
- Do not describe audio or sound (the model generates silent video).

Only output the final enhanced video prompt, nothing else."""


class PromptEnhancer:
    """CPU-based prompt enhancement using Qwen2.5-7B-Instruct."""
    
    _instance: Optional["PromptEnhancer"] = None
    _llm: Optional[Llama] = None
    
    def __new__(cls):
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _ensure_model_downloaded(self) -> Path:
        """Download model if not present, return path."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / MODEL_FILE
        
        if not model_path.exists():
            logger.info(f"Downloading {MODEL_FILE} from HuggingFace...")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False
            )
            logger.info("Model downloaded successfully!")
        
        return model_path
    
    def _load_model(self):
        """Load the LLM model on CPU."""
        if self._llm is not None:
            return
        
        model_path = self._ensure_model_downloaded()
        logger.info("Loading Qwen2.5-7B-Instruct for prompt enhancement...")
        
        # Get CPU thread count for optimal performance
        cpu_threads = max(1, int(multiprocessing.cpu_count() * 0.8))  # Use 80% of cores
        
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,         # Larger context for 7B model
            n_threads=cpu_threads,  # Use more CPU threads
            n_gpu_layers=0,     # CPU only - preserve GPU for image gen
            verbose=False
        )
        logger.info(f"Enhancer model loaded! (using {cpu_threads} CPU threads)")
    
    def enhance(self, prompt: str, progress_callback=None) -> str:
        """
        Enhance a user prompt into a detailed visual description.
        
        Args:
            prompt: Simple user prompt
            progress_callback: Optional callback for progress updates
            
        Returns:
            Enhanced prompt with detailed visual description
        """
        try:
            if progress_callback:
                progress_callback("Loading enhancer model...")
            
            self._load_model()
            
            if progress_callback:
                progress_callback("Enhancing prompt...")
            
            # Create chat messages
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Generate enhanced prompt
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=384,      # More detailed outputs with 7B model
                temperature=0.7,
                top_p=0.9,
            )
            
            # Extract the enhanced prompt from response
            enhanced = response["choices"][0]["message"]["content"].strip()
            
            if progress_callback:
                progress_callback("Prompt enhanced!")
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            # Return original prompt on error
            return prompt

    def enhance_video(self, prompt: str, progress_callback=None) -> str:
        """
        Enhance a user prompt for video generation with WAN 2.2.

        Args:
            prompt: Simple user prompt describing desired video
            progress_callback: Optional callback for progress updates

        Returns:
            Enhanced cinematic prompt optimized for WAN 2.2
        """
        try:
            if progress_callback:
                progress_callback("Loading enhancer model...")

            self._load_model()

            if progress_callback:
                progress_callback("Enhancing video prompt...")

            # Create chat messages with video-specific system prompt
            messages = [
                {"role": "system", "content": VIDEO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            # Generate enhanced prompt (slightly more tokens for video descriptions)
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=256,      # 80-120 words target
                temperature=0.7,
                top_p=0.9,
            )

            # Extract the enhanced prompt from response
            enhanced = response["choices"][0]["message"]["content"].strip()

            if progress_callback:
                progress_callback("Video prompt enhanced!")

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing video prompt: {e}")
            # Return original prompt on error
            return prompt


# Global enhancer instance
_enhancer = None


def get_enhancer() -> PromptEnhancer:
    """Get or create the global prompt enhancer instance."""
    global _enhancer
    if _enhancer is None:
        _enhancer = PromptEnhancer()
    return _enhancer


def enhance_prompt(prompt: str, progress_callback=None) -> str:
    """
    Convenience function to enhance a prompt.

    Args:
        prompt: Simple user prompt
        progress_callback: Optional callback for progress updates

    Returns:
        Enhanced prompt with detailed visual description
    """
    enhancer = get_enhancer()
    return enhancer.enhance(prompt, progress_callback)


def enhance_video_prompt(prompt: str, progress_callback=None) -> str:
    """
    Convenience function to enhance a video prompt for WAN 2.2.

    Args:
        prompt: Simple user prompt describing desired video
        progress_callback: Optional callback for progress updates

    Returns:
        Enhanced cinematic prompt optimized for WAN 2.2
    """
    enhancer = get_enhancer()
    return enhancer.enhance_video(prompt, progress_callback)
