# Copyright (c) Meta Platforms, Inc. and affiliates.
"""LLM-based pairwise judges."""

from .api_pairwise_evaluator import (
    Gemini25FlashPairwiseEvaluator,
    GPT4oPairwiseEvaluator,
)
from .local_pairwise_evaluator import (
    Qwen3VL8BPairwiseEvaluator,
)

__all__ = [
    "GPT4oPairwiseEvaluator",
    "Gemini25FlashPairwiseEvaluator",
    "Qwen3VL8BPairwiseEvaluator",
]
