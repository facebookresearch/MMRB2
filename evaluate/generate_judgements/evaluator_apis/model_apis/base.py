"""Base class for model APIs."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from ..types import Prompt


@dataclass
class APIResponse:
    """Response from a model API call.

    Attributes:
        response: List of response content items.
        model_name: Name of the model that generated the response.
        date: Timestamp in ISO format.
        metadata: Additional response metadata.
    """

    response: List[Any]
    model_name: str
    date: str
    metadata: dict = field(default_factory=dict)


class BaseAPI(ABC):
    """Abstract base class for model APIs."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass

    def get_api_key(self) -> dict:
        """Return the API key for the model from environment variables.

        For OpenAI models: reads from OPENAI_API_KEY
        For Google models: reads from GOOGLE_API_KEY
        """
        if self.model_name.startswith("gpt-"):
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY environment variable:\n"
                    "  export OPENAI_API_KEY='your-api-key'"
                )
            return {"api_key": key}
        elif self.model_name.startswith("gemini-"):
            key = os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Google API key not found. Please set GOOGLE_API_KEY environment variable:\n"
                    "  export GOOGLE_API_KEY='your-api-key'"
                )
            return {"api_key": key}
        else:
            raise ValueError(f"Unknown model type: {self.model_name}")

    def generate_text(
        self,
        prompt: Prompt,
        n: int = 1,
    ) -> List[APIResponse]:
        """Return generated text from the model for the given prompt.

        Args:
            prompt: The prompt to generate the text from.
            n: Number of text responses to generate (default: 1).

        Returns:
            List of API responses containing the generated text.
        """
        raise NotImplementedError(
            f"generate_text not implemented for {self.model_name}"
        )

    def generate_interleaved(
        self,
        prompt: Prompt,
        output_path: Path,
        n: int = 1,
    ) -> List[APIResponse]:
        """Return interleaved text and image from the model for the given prompt.

        Args:
            prompt: The prompt to generate interleaved content from.
            output_path: The path where the generated content should be saved.
            n: Number of interleaved responses to generate (default: 1).

        Returns:
            List of API responses containing the interleaved content.
        """
        raise NotImplementedError(
            f"generate_interleaved not implemented for {self.model_name}"
        )
