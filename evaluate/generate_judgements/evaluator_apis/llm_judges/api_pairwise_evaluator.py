"""API-based pairwise evaluators using OpenAI and Google APIs."""

import json
import re
from pathlib import Path
from typing import List

import json_repair

from ..base import BasePairwiseEvaluator, EvaluatorResult
from ..model_apis.google.gemini import Gemini
from ..model_apis.openai import OpenAIGPT
from ..types import create_prompt
from .evaluation_prompts import (
    get_image_edit_prompt,
    get_image_gen_prompt,
    get_interleaved_prompt,
    get_reasoning_prompt,
)


class APIPairwiseEvaluator(BasePairwiseEvaluator):
    """Pairwise evaluator using API-based models (OpenAI, Google)."""
    
    AVAILABLE_MODELS = {
        "gpt-4o": "gpt-4o",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    def __init__(self, model_name: str = "gpt-4.1", device_id: int = None):
        assert (
            model_name in self.AVAILABLE_MODELS
        ), f"Model {model_name} not supported. Available models: {list(self.AVAILABLE_MODELS.keys())}"

        self.model_name = model_name
        self.device_id = device_id
        self.is_gemini = model_name.startswith("gemini-")

        if self.is_gemini:
            self.client = Gemini(model_name)
        else:
            self.client = OpenAIGPT(model_name)

    @property
    def evaluator_name(self):
        return f"{self.model_name}_pairwise_evaluator"

    def generate_response(self, prompt: List[List[str]]):
        """Generate a response from the model."""
        prompt = create_prompt(
            prompt=prompt,
            source="",
            metadata={},
        )
        response = self.client.generate_text(prompt, output_path=Path("."))
        return response[0].response[0][1]

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
            raise ValueError(f"Failed to parse JSON: {e}\nText was:\n{text}")

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
            response = self.generate_response(content_list)
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


class GPT4oPairwiseEvaluator(APIPairwiseEvaluator):
    """GPT-4o based pairwise evaluator."""
    def __init__(self, device_id: int = None):
        super().__init__(model_name="gpt-4o", device_id=device_id)


class Gemini25FlashPairwiseEvaluator(APIPairwiseEvaluator):
    """Gemini 2.5 Flash based pairwise evaluator."""
    def __init__(self, device_id: int = None):
        super().__init__(model_name="gemini-2.5-flash", device_id=device_id)

