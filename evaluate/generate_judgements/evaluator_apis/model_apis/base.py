"""Base class for model APIs."""

import getpass
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from ..types import Prompt

# Default API key directory - users can override by setting environment variables
API_KEY_DIR = os.environ.get("MMRB2_API_KEY_DIR", f"/home/{getpass.getuser()}/.api_keys/")


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

    def _get_api_input_from_user(self) -> dict:
        """Get API credentials from user input."""
        # Override for APIs which require multiple keys
        if self.model_name == "gpt-4o":
            azure_endpoint = input(
                f"Please enter the Azure endpoint for {self.model_name}: "
            )
            api_key = input(f"Please enter the API key for {self.model_name}: ")
            return {"azure_endpoint": azure_endpoint, "api_key": api_key}
        else:
            key = input(f"Please enter the API key for {self.model_name}: ")
            return {"api_key": key}

    def get_api_key(self) -> dict:
        """Return the API key for the model.
        
        Looks for credentials in the following order:
        1. Environment variables (OPENAI_API_KEY, GOOGLE_API_KEY, etc.)
        2. JSON file in API_KEY_DIR
        3. Interactive user input (saved for future use)
        """
        # Check environment variables first
        env_key = self._check_env_vars()
        if env_key:
            return env_key
            
        if not os.path.exists(API_KEY_DIR):
            os.makedirs(API_KEY_DIR, exist_ok=True)

        api_key_path = f"{API_KEY_DIR}/{self.model_name}.json"
        if not os.path.exists(api_key_path):
            # Get the API key by prompting the user
            api_key = self._get_api_input_from_user()
            # Save the API key to a file
            with open(api_key_path, "w") as f:
                json.dump(api_key, f)
        else:
            # Load the API key from the file
            with open(api_key_path) as f:
                api_key = json.load(f)

        return api_key
    
    def _check_env_vars(self) -> dict | None:
        """Check for API keys in environment variables."""
        if self.model_name.startswith("gpt-"):
            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if endpoint and key:
                return {"azure_endpoint": endpoint, "api_key": key}
        elif self.model_name.startswith("gemini-"):
            key = os.environ.get("GOOGLE_API_KEY")
            if key:
                return {"api_key": key}
        return None

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

