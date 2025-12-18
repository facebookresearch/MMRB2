"""OpenAI API utility functions."""

import base64
import os
from pathlib import Path
from time import sleep
from typing import Dict, List

from openai import OpenAI


class OpenAIClient:
    """Client for OpenAI API calls."""

    def __init__(self, model_name: str, api_key: str = None):
        """Initialize the OpenAI client.

        Args:
            model_name: Name of the model to use.
            api_key: API key for authentication. If not provided, reads from OPENAI_API_KEY env var.
        """
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
                )

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_to_content(self, image_path: str) -> dict:
        """Convert image to content dict for multimodal input."""
        image_ext = Path(image_path).suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(image_ext, "image/jpeg")

        base64_image = self.encode_image_base64(image_path)

        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
        }

    def chat_with_retry(
        self, messages: List[Dict], n=1, max_retries: int = 3, **kwargs
    ):
        """Retry chat completion with exponential backoff."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, messages=messages, n=n, **kwargs
                )
                return response

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                print(f"Error in attempt {attempt + 1}: {error_type}: {error_msg}")

                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Failed to complete chat after {max_retries} attempts. "
                        f"Last error: {error_type}: {error_msg}"
                    )
                sleep(2**attempt)

    def simple_query(self, prompt, n=1, max_retries=3, **kwargs):
        """Simple query helper method."""
        parts = []
        for seg in prompt:
            if seg[0] == "text":
                parts.append({"type": "text", "text": seg[1]})
            elif seg[0] == "image":
                parts.append(self.image_to_content(seg[1]))

        messages = [{"role": "user", "content": parts}]
        resp = self.chat_with_retry(messages, n, max_retries, **kwargs)
        return [resp.choices[i].message.content for i in range(n)]
