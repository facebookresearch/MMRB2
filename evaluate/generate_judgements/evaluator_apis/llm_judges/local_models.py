"""Local model management for running evaluators on GPU."""

from typing import List, Optional

import torch
import torch._dynamo
from PIL import Image
from transformers import pipeline


class LocalModelManager:
    """Manager for loading and running local VLM models."""
    
    MODEL_CONFIGS = {
        "qwen3-vl-8b": {
            "huggingface_id": "Qwen/Qwen3-VL-8B-Instruct",
        },
    }

    def __init__(
        self,
        model_name: str,
        device_id: Optional[int] = None,
    ):
        """Initialize the local model manager.
        
        Args:
            model_name: Name of the model to load.
            device_id: GPU device ID to use. If None, uses CUDA if available.
        """
        self.model_name = model_name

        if device_id is not None:
            self.device_id = device_id
        else:
            self.device_id = "cuda" if torch.cuda.is_available() else "cpu"

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.model_full_name = self.MODEL_CONFIGS[model_name]["huggingface_id"]
        self.tp = self.MODEL_CONFIGS[model_name].get("tp", False)

        if self.tp:
            print("Big model. Use all GPUs.")
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_full_name,
                device_map="auto",
                dtype=torch.bfloat16,
            )
        else:
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_full_name,
                device=self.device_id,
                dtype=torch.bfloat16,
            )

        # Stabilize Gemma-3 decoding
        if self.model_name.startswith("gemma-3"):
            self.pipe.model.generation_config.cache_implementation = "static"
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

    def generate_response(
        self,
        prompt: List[List[str]],
        max_new_tokens: int = 2000,
        temperature: float = 1.0,
    ) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: List of [type, content] pairs where type is "text" or "image".
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text response.
        """
        system_prompt = None
        messages = []

        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                }
            )

        user_content = []
        for segment in prompt:
            if segment[0] == "text":
                user_content.append({"type": "text", "text": segment[1]})
            elif segment[0] == "image":
                user_content.append(
                    {"type": "image", "image": Image.open(segment[1])}
                )

        messages.append({"role": "user", "content": user_content})

        response = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=True,
            temperature=temperature,
        )

        return response[0]["generated_text"]

