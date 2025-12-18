"""Base classes for pairwise evaluators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvaluatorResult:
    """Result from a pairwise evaluation.
    
    Attributes:
        judgement: The judgement result - "A", "B", or "tie" for pairwise judges.
        metadata: Additional information such as rationales from the LLM.
    """
    judgement: str
    metadata: dict = field(default_factory=dict)


class BasePairwiseEvaluator(ABC):
    """Abstract base class for pairwise evaluators."""
    
    @property
    @abstractmethod
    def evaluator_name(self) -> str:
        """Return the name of the evaluator."""
        pass

    @abstractmethod
    def pairwise_evaluate(
        self,
        prompt_content: List[List[str]],
        response_a: List[List[str]],
        response_b: List[List[str]],
        task_type: str,
        n: int = 1,
        verbose: bool = False,
    ) -> List[EvaluatorResult]:
        """Compare two responses to a prompt and return n judgements.

        Args:
            prompt_content: The prompt content. Format: [["text", "..."], ["image", "path/to/image"]]
            response_a: The first response in the same format.
            response_b: The second response in the same format.
            task_type: The type of task (image, edit, interleaved, reasoning).
            n: The number of judgements to generate. Defaults to 1.
            verbose: Whether to print verbose output.
            
        Returns:
            List of EvaluatorResult objects containing judgements.
        """
        pass

