"""Type definitions for the evaluator APIs."""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Prompt:
    """Represents a multimodal prompt with text and images.
    
    Attributes:
        prompt: List of content segments, each as [type, content] where type is "text" or "image".
        source: Source identifier for the prompt.
        metadata: Additional metadata associated with the prompt.
    """
    prompt: List[List[Any]]
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for datatype, _ in self.prompt:
            assert datatype in [
                "text",
                "image",
            ], f"Invalid datatype: {datatype}. Must be 'text' or 'image'."

        if "task" not in self.metadata:
            self.metadata["task"] = None

    def to_json(self) -> dict:
        """Convert the Prompt object to a JSON-serializable dictionary."""
        return {
            "prompt": self.prompt,
            "source": self.source,
            "metadata": self.metadata,
        }


def create_prompt(
    prompt: List[List[Any]], source: str, metadata: dict
) -> Prompt:
    """Create a Prompt object with the given parameters.

    Args:
        prompt: The prompt content as list of [type, content] pairs.
        source: The source of the prompt.
        metadata: Additional metadata associated with the prompt.

    Returns:
        A Prompt instance.
    """
    return Prompt(prompt=prompt, source=source, metadata=metadata)


def extract_text_from_prompt(prompt: Prompt) -> str:
    """Extract text content from the prompt object.

    Args:
        prompt: Prompt object containing mixed content.

    Returns:
        Concatenated text content.
    """
    text_parts = []

    for data_type, content in prompt.prompt:
        if data_type == "text":
            text_parts.append(str(content))

    return " ".join(text_parts).strip()

