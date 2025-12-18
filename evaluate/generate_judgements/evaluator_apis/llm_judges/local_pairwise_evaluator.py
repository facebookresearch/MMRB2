# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Local model pairwise evaluators."""

import json
import re
from typing import List

import json_repair

from ..base import BasePairwiseEvaluator, EvaluatorResult
from .evaluation_prompts import (
    get_image_edit_prompt,
    get_image_gen_prompt,
    get_interleaved_prompt,
    get_reasoning_prompt,
)
from .local_models import LocalModelManager


class LocalPairwiseEvaluator(BasePairwiseEvaluator):
    """Pairwise evaluator using local VLM models."""

    def __init__(self, model_name: str, device_id: int = None):
        self.model_name = model_name
        self.device_id = device_id
        self.model_manager = LocalModelManager(model_name, device_id)

    @property
    def evaluator_name(self):
        return f"{self.model_name}_pairwise_evaluator"

    def parse_llm_json(self, text):
        """Parse JSON from LLM output that may be wrapped in markdown code blocks.

        Args:
            text: The raw text output from the LLM.

        Returns:
            Parsed JSON as a dictionary.
        """
        # Remove markdown code block formatting
        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?```\s*$", "", text.strip())

        try:
            return json_repair.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON: {e}\nText was:\n\n{text}\n\n...ends here...\n\n"
            )

    def pairwise_evaluate(
        self,
        prompt_content: List[List[str]],
        response_a: List[List[str]],
        response_b: List[List[str]],
        task_type,
        n: int = 1,
        verbose: bool = False,
    ) -> List[EvaluatorResult]:
        """Evaluate two responses and return judgements."""
        # Select evaluation prompt based on task type
        if task_type == "image":
            evaluation_prompt = get_image_gen_prompt()
        elif task_type == "edit":
            evaluation_prompt = get_image_edit_prompt()
        elif task_type == "interleaved":
            evaluation_prompt = get_interleaved_prompt()
        elif task_type in ("text", "reasoning"):
            evaluation_prompt = get_reasoning_prompt()
        else:
            raise ValueError(
                f"Invalid task type: {task_type}. Must be image, edit, interleaved, or reasoning."
            )

        prompt_text = evaluation_prompt + "\n\n"
        prompt_text += "CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON."

        # Build the content list
        content_list = []
        content_list.append(["text", evaluation_prompt])
        content_list.append(["text", "[ORIGINAL PROMPT TO MODEL:]"])
        content_list.extend(prompt_content)
        content_list.append(["text", "[RESPONSE A:]"])
        content_list.extend(response_a)
        content_list.append(["text", "[RESPONSE B:]"])
        content_list.extend(response_b)

        outputs = []
        for _ in range(n):
            response = self.model_manager.generate_response(content_list)
            try:
                parsed_response = self.parse_llm_json(response)
            except ValueError as e:
                raise ValueError(f"Failed to parse JSON: {e}")

            final_judgement = parsed_response["better_response"]

            outputs.append(
                EvaluatorResult(
                    judgement=final_judgement,
                    metadata=parsed_response,
                )
            )

        return outputs


class Qwen3VL8BPairwiseEvaluator(LocalPairwiseEvaluator):
    """Qwen3-VL-8B based pairwise evaluator."""

    def __init__(self, device_id: int = None):
        super().__init__(model_name="qwen3-vl-8b", device_id=device_id)
