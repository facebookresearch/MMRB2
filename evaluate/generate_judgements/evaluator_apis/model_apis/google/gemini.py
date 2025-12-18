"""Google Gemini API for text and multimodal generation."""

import datetime
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image
from google import genai
from google.genai import types

from ...types import Prompt, extract_text_from_prompt
from ..base import APIResponse, BaseAPI
from .utils import generate_content_with_retry


class Gemini(BaseAPI):
    """Google Gemini API for text and multimodal generation."""

    AVAILABLE_MODELS = {
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        temperature: float | None = None,
    ):
        """Initialize Gemini with specified model.

        Args:
            model_name: Gemini model to use.
            temperature: Temperature for generation (default: None, uses API default).
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unsupported Gemini model: {model_name}. "
                f"Available models: {list(self.AVAILABLE_MODELS.keys())}"
            )
        self.gemini_model_name = model_name
        self.model_id = self.AVAILABLE_MODELS[model_name]
        self.temperature = temperature

    @property
    def model_name(self) -> str:
        return f"Gemini_{self.gemini_model_name.replace('-', '_')}"

    def generate_text(
        self,
        prompt: Prompt,
        output_path: Path,
        n: int = 1,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> List[APIResponse]:
        """Generate text from the model for the given prompt.

        Args:
            prompt: Prompt object containing mixed content.
            output_path: Path (unused for text generation).
            n: Number of responses (limited to 1 for Gemini).
            system_prompt: Optional system prompt.
            temperature: Temperature for generation.

        Returns:
            List of APIResponse containing generated text.
        """
        assert n == 1, "Gemini only supports single response generation"
        try:
            api_key = self.get_api_key()["api_key"]
            client = genai.Client(api_key=api_key)

            # Prepare content for Gemini
            contents = self._prepare_contents(prompt)

            # Build config with conditional temperature
            config_params = {
                "response_modalities": ["TEXT"],
                "system_instruction": system_prompt if system_prompt else None,
            }
            if temperature is not None:
                config_params["temperature"] = temperature
            elif self.temperature is not None:
                config_params["temperature"] = self.temperature

            response = generate_content_with_retry(
                client,
                max_retries=8,
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(**config_params),
            )

            # Process the response to extract interleaved content
            interleaved_content = self._process_response_interleaved(response)

            processed_content = []
            n_images = 0
            for content in interleaved_content:
                if content[0] == "text":
                    processed_content.append(["text", content[1]])

            current_time = datetime.datetime.now().isoformat()

            return [
                APIResponse(
                    response=processed_content,
                    model_name=self.model_name,
                    date=current_time,
                    metadata={
                        "input_content": prompt.prompt,
                        "input_text": extract_text_from_prompt(prompt),
                        "input_images_count": len(
                            [c for c in prompt.prompt if c[0] == "image"]
                        ),
                        "num_images_generated": n_images,
                        "prompt_source": prompt.source,
                        "prompt_metadata": prompt.metadata,
                        "model_name": self.model_id,
                    },
                )
            ]

        except Exception as e:
            current_time = datetime.datetime.now().isoformat()
            return [
                APIResponse(
                    response=[["error", f"Error generating content: {str(e)}"]],
                    model_name=self.model_name,
                    date=current_time,
                    metadata={
                        "error": str(e),
                        "prompt_source": prompt.source,
                        "prompt_metadata": prompt.metadata,
                        "model_name": self.model_id,
                    },
                )
            ]

    def _prepare_contents(self, prompt: Prompt) -> List:
        """Prepare content for Gemini model input.

        Args:
            prompt: Prompt object containing mixed content.

        Returns:
            Content list for Gemini API.
        """
        contents = []

        for data_type, content in prompt.prompt:
            if data_type == "text":
                contents.append(str(content))
            elif data_type == "image":
                if isinstance(content, str):
                    contents.append(Image.open(content))
                elif isinstance(content, bytes):
                    contents.append(Image.open(BytesIO(content)))
                else:
                    contents.append(content)

        return contents

    def _process_response_interleaved(self, response) -> List:
        """Process Gemini response to extract interleaved text and images."""
        interleaved_content = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text is not None:
                        interleaved_content.append(["text", part.text])
                    elif hasattr(part, "inline_data") and part.inline_data is not None:
                        image_bytes = part.inline_data.data
                        image = Image.open(BytesIO(image_bytes))
                        interleaved_content.append(["image", image])

        return interleaved_content
