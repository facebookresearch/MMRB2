# Copyright (c) Meta Platforms, Inc. and affiliates.
"""OpenAI GPT API for text generation."""

import datetime
import os
from pathlib import Path
from typing import List

from ...types import Prompt, extract_text_from_prompt
from ..base import APIResponse, BaseAPI
from .utils import OpenAIClient


class OpenAIGPT(BaseAPI):
    """OpenAI GPT API for text generation."""

    AVAILABLE_MODELS = {
        "gpt-4o": "gpt-4o",
    }

    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize OpenAI GPT with specified model.

        Args:
            model_name: Name of the model to use.
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported GPT model: {model_name}. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )
        self.openai_model_name = model_name

        # Initialize client - will read from OPENAI_API_KEY env var
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable:\n"
                "  export OPENAI_API_KEY='your-api-key'"
            )

        self.client = OpenAIClient(model_name=model_name, api_key=api_key)

    @property
    def model_name(self) -> str:
        return self.openai_model_name

    def generate_text(
        self, prompt: Prompt, output_path: Path, n: int = 1
    ) -> List[APIResponse]:
        """Generate text from the model.

        Args:
            prompt: The prompt containing text and/or images.
            output_path: Output path (unused for text generation).
            n: Number of responses to generate.

        Returns:
            List of APIResponse objects.
        """
        # Generate text from input
        content = []
        for seg in prompt.prompt:
            if seg[0] == "text":
                content.append({"type": "text", "text": seg[1]})
            elif seg[0] == "image":
                content.append(self.client.image_to_content(seg[1]))
        messages = [{"role": "user", "content": content}]

        response = self.client.chat_with_retry(messages, n)

        current_time = datetime.datetime.now().isoformat()

        metadata = {
            "input_content": prompt.prompt,
            "input_text": extract_text_from_prompt(prompt),
            "input_images_count": len([c for c in prompt.prompt if c[0] == "image"]),
            "num_images_generated": 0,
            "prompt_source": prompt.source,
            "prompt_metadata": prompt.metadata,
            "model_name": self.model_name,
        }

        outputs = []
        for choice in response.choices:
            text = choice.message.content
            outputs.append(
                APIResponse(
                    response=[["text", text]],
                    model_name=self.model_name,
                    date=current_time,
                    metadata=metadata,
                )
            )

        return outputs
