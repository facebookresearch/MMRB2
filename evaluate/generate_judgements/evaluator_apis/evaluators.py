"""Registry of available evaluators."""

from .base import BasePairwiseEvaluator
from .llm_judges import (
    Gemini25FlashPairwiseEvaluator,
    GPT4oPairwiseEvaluator,
    Qwen3VL8BPairwiseEvaluator,
)


class EvaluatorTypes:
    """Types of evaluators."""

    PAIRWISE = "pairwise"


class EvaluatorCapabilities:
    """Task types that evaluators can handle."""

    IMAGE = "image"
    TEXT = "text"
    EDIT = "edit"
    INTERLEAVED = "interleaved"
    REASONING = "reasoning"


# Registry of available evaluators
EVALUATORS = {
    "gpt4o-pairwise": {
        "class": GPT4oPairwiseEvaluator,
        "type": EvaluatorTypes.PAIRWISE,
        "is_api_based": True,
        "capabilities": [
            EvaluatorCapabilities.IMAGE,
            EvaluatorCapabilities.EDIT,
            EvaluatorCapabilities.INTERLEAVED,
            EvaluatorCapabilities.TEXT,
            EvaluatorCapabilities.REASONING,
        ],
    },
    "gemini25flash-pairwise": {
        "class": Gemini25FlashPairwiseEvaluator,
        "type": EvaluatorTypes.PAIRWISE,
        "is_api_based": True,
        "capabilities": [
            EvaluatorCapabilities.IMAGE,
            EvaluatorCapabilities.EDIT,
            EvaluatorCapabilities.INTERLEAVED,
            EvaluatorCapabilities.TEXT,
            EvaluatorCapabilities.REASONING,
        ],
    },
    "qwen3vl8b-pairwise": {
        "class": Qwen3VL8BPairwiseEvaluator,
        "type": EvaluatorTypes.PAIRWISE,
        "is_api_based": False,
        "capabilities": [
            EvaluatorCapabilities.IMAGE,
            EvaluatorCapabilities.EDIT,
            EvaluatorCapabilities.INTERLEAVED,
            EvaluatorCapabilities.TEXT,
            EvaluatorCapabilities.REASONING,
        ],
    },
}


def get_evaluator_by_name(
    name: str, task_type: str = None, device_id: int = None
) -> BasePairwiseEvaluator:
    """Get an evaluator instance by name.

    Args:
        name: Name of the evaluator (e.g., "gpt41-pairwise").
        task_type: Task type to validate against evaluator capabilities.
        device_id: GPU device ID for local models.

    Returns:
        An instance of the requested evaluator.

    Raises:
        ValueError: If evaluator not found or task type not supported.
    """
    if name not in EVALUATORS:
        raise ValueError(
            f"Evaluator {name} not found. Available: {list(EVALUATORS.keys())}"
        )

    evaluator_capabilities = EVALUATORS[name]["capabilities"]
    if len(evaluator_capabilities) != 1 and task_type is None:
        raise ValueError(
            f"Evaluator {name} has multiple capabilities: {evaluator_capabilities}. "
            "Please specify a task type."
        )
    if task_type is not None and task_type not in evaluator_capabilities:
        raise ValueError(
            f"Evaluator {name} does not support task type: {task_type}. "
            f"Available task types: {evaluator_capabilities}"
        )

    is_api_based = EVALUATORS[name]["is_api_based"]

    if is_api_based:
        print(f"Using API-based evaluator: {name}. No need to specify device_id.")
        device_id = None

    evaluator = EVALUATORS[name]["class"](device_id)
    if isinstance(evaluator, BasePairwiseEvaluator):
        return evaluator
