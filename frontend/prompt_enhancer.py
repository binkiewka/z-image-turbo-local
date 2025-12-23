"""
Prompt Enhancement Module for Z-Image-Turbo

Uses Qwen2.5-1.5B-Instruct on CPU to transform simple prompts into
detailed visual descriptions suitable for image generation.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Model configuration
MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
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


class PromptEnhancer:
    """CPU-based prompt enhancement using Qwen2.5-1.5B-Instruct."""
    
    _instance = None
    _llm = None
    
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
            print(f"Downloading {MODEL_FILE} from HuggingFace...")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False
            )
            print("Model downloaded successfully!")
        
        return model_path
    
    def _load_model(self):
        """Load the LLM model on CPU."""
        if self._llm is not None:
            return
        
        model_path = self._ensure_model_downloaded()
        print("Loading Qwen2.5-1.5B-Instruct for prompt enhancement...")
        
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,       # Context window
            n_threads=4,      # CPU threads
            n_gpu_layers=0,   # CPU only - preserve GPU for image gen
            verbose=False
        )
        print("Enhancer model loaded!")
    
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
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Extract the enhanced prompt from response
            enhanced = response["choices"][0]["message"]["content"].strip()
            
            if progress_callback:
                progress_callback("Prompt enhanced!")
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing prompt: {e}")
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
