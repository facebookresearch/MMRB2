#!/usr/bin/env python3
"""
MMRB2 Benchmark - Generate Judgements

This script generates pairwise judgements for the MMRB2 benchmark using
LLM-based evaluators (API or local models).

Usage:
    python evaluate.py --evaluator_name gpt41-pairwise --task_type image \
                       --pairs_path /path/to/pairs.json --output_path ./outputs/results.json
"""

import argparse
import json
import os
from typing import List

from tqdm import tqdm

from evaluator_apis.base import BasePairwiseEvaluator


def fix_relative_path(
    content: List[List[str]], base_dir: str
) -> List[List[str]]:
    """Fix relative paths in content by prepending base_dir.
    
    Args:
        content: List of [type, content] pairs.
        base_dir: Base directory to prepend to relative paths.
        
    Returns:
        Content with fixed paths.
    """
    new_content = []
    for item in content:
        if item[0] == "image":
            if not os.path.isabs(item[1]):
                item[1] = os.path.join(base_dir, item[1])
            new_content.append(item)
        else:
            new_content.append(item)
    return new_content


def evaluate_pairs(
    prompt_response_tuples: List,
    evaluator: BasePairwiseEvaluator,
    task_type: str,
):
    """Evaluate multiple pairs with an evaluator."""
    results = []
    for prompt_response_tuple in prompt_response_tuples:
        prompt_content, response_a, response_b = prompt_response_tuple
        result = evaluator.pairwise_evaluate(
            prompt_content, response_a, response_b, task_type
        )
        results.append(result)
    return results


def process_pairs_with_evaluator(
    evaluator_name: str,
    task_type: str,
    pairs_path: str,
    output_path: str,
    n: int = 1,
    max_samples: int = None,
    device_id: int = None,
    base_dir: str = None,
):
    """Process pairs with an evaluator and save results.

    Args:
        evaluator_name: Name of the evaluator to use.
        task_type: Task type to use. Must be one of the evaluator capabilities.
        pairs_path: Path to the pairs JSON file.
        output_path: Path to save the output results.
        n: Number of judgements to generate per pair.
        max_samples: Maximum number of samples to process.
        device_id: Device ID to use for local models.
        base_dir: Base directory for relative paths in the pairs file.
    """
    # Load the evaluator
    from evaluator_apis.evaluators import get_evaluator_by_name

    evaluator = get_evaluator_by_name(evaluator_name, task_type, device_id)

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Verify it's a pairwise evaluator
    if not isinstance(evaluator, BasePairwiseEvaluator):
        raise ValueError(
            f"Evaluator {evaluator_name} is not a pairwise evaluator"
        )

    # Load the pairs
    print(f"Loading pairs from {pairs_path}...")
    with open(pairs_path, "r") as f:
        data = json.load(f)

    pairs = data.get("pairs", [])
    print(f"Loaded {len(pairs)} pairs")

    # Apply max_samples limit if specified
    if max_samples is not None:
        pairs = pairs[:max_samples]
        print(f"Processing only first {max_samples} pairs")

    # Process each pair
    results = {}
    print(f"Evaluating pairs with {evaluator_name}...")

    if base_dir is None:
        base_dir = os.path.dirname(pairs_path)

    for pair in tqdm(pairs, desc="Evaluating pairs"):
        pair_id = pair["id"]
        prompt_content = pair["prompt_content"]
        response_a = pair["response_a"]["response_content"]
        response_b = pair["response_b"]["response_content"]

        # Fix relative paths
        prompt_content = fix_relative_path(prompt_content, base_dir)
        response_a = fix_relative_path(response_a, base_dir)
        response_b = fix_relative_path(response_b, base_dir)

        # Get evaluation results (n judgements) - forward direction
        judge_results = []
        n_failure = 0

        while n_failure < n and len(judge_results) < n:
            try:
                judge_result = evaluator.pairwise_evaluate(
                    prompt_content=prompt_content,
                    response_a=response_a,
                    response_b=response_b,
                    task_type=task_type,
                    n=1,
                    verbose=False,
                )
                judge_results.extend(judge_result)
            except Exception as e:
                n_failure += 1
                print(f"Error evaluating pair {pair_id}: {e}")
                continue

        # Get evaluation results - reverse direction (swap A and B)
        judge_results_reverse = []
        n_failure = 0
        while n_failure < n and len(judge_results_reverse) < n:
            try:
                judge_result = evaluator.pairwise_evaluate(
                    prompt_content=prompt_content,
                    response_a=response_b,
                    response_b=response_a,
                    task_type=task_type,
                    n=1,
                    verbose=False,
                )
                judge_results_reverse.extend(judge_result)
            except Exception as e:
                n_failure += 1
                print(f"Error evaluating pair {pair_id}: {e}")
                continue

        # Convert EvaluatorResult objects to dictionaries for JSON serialization
        results[pair_id] = {}

        results[pair_id]["forward"] = [
            {
                "judgement": result.judgement,
                "evaluator": evaluator_name,
                "metadata": result.metadata,
            }
            for result in judge_results
        ]

        results[pair_id]["reverse"] = [
            {
                "judgement": result.judgement,
                "evaluator": evaluator_name,
                "metadata": result.metadata,
            }
            for result in judge_results_reverse
        ]

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done! Processed {len(results)} pairs")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise judgements for MMRB2 benchmark."
    )
    parser.add_argument(
        "--evaluator_name",
        type=str,
        required=True,
        help="Name of the evaluator to use (e.g., gpt41-pairwise, qwen3vl8b-pairwise)",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        help="Task type (image, edit, interleaved, reasoning)",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        required=True,
        help="Path to the pairs JSON file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output results",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of judgements to generate per pair (default: 1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="Device ID to use for local models (default: None)",
    )

    args = parser.parse_args()

    process_pairs_with_evaluator(
        evaluator_name=args.evaluator_name,
        task_type=args.task_type,
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        n=args.n,
        max_samples=args.max_samples,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()

