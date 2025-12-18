# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3
"""
MMRB2 Benchmark - Compute Accuracy Scores

This script computes accuracy scores from judgement files by comparing
predictions against ground truth labels.

Usage:
    # Evaluate a single task
    python compute_accuracy.py --task image --predictions /path/to/predictions.json
    
    # Evaluate all tasks (provide 4 prediction files)
    python compute_accuracy.py --task all \
        --predictions task1.json task2.json task3.json task4.json
"""

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TaskResult:
    """Results for a single task evaluation."""

    task_name: str
    total_pairs: int
    evaluated_pairs: int
    missing_pairs: int
    total_correct: int
    total_evaluated: int
    accuracy: float


def flip_judgement(judgement: str) -> str:
    """Flip A to B and B to A."""
    if judgement == "A":
        return "B"
    elif judgement == "B":
        return "A"
    else:
        return judgement


def load_json(path: str) -> dict:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def compute_task_accuracy(
    predictions: Dict, benchmark_data: Dict, task_name: str = "unknown"
) -> TaskResult:
    """Compute accuracy for a single task.

    Args:
        predictions: Dict mapping pair_id to prediction results.
        benchmark_data: Benchmark data containing pairs with ground truth.
        task_name: Name of the task for reporting.

    Returns:
        TaskResult with accuracy metrics.
    """
    pairs = benchmark_data.get("pairs", [])

    total_pairs = len(pairs)
    evaluated_pairs = 0
    missing_pairs = 0

    total_correct = 0
    total_evaluated = 0

    for pair in pairs:
        pair_id = pair["id"]
        chosen = pair["chosen"]  # Ground truth: "A" or "B"

        if pair_id not in predictions:
            missing_pairs += 1
            continue

        evaluated_pairs += 1
        pred = predictions[pair_id]

        # Forward evaluation
        if "forward" in pred and len(pred["forward"]) > 0:
            forward_judgement = pred["forward"][0].get("judgement")
            if forward_judgement:
                total_evaluated += 1
                if forward_judgement == chosen:
                    total_correct += 1

        # Reverse evaluation
        if "reverse" in pred and len(pred["reverse"]) > 0:
            reverse_pred = pred["reverse"][0].get("judgement")
            if reverse_pred:
                total_evaluated += 1
                flipped_judgement = flip_judgement(reverse_pred)
                if flipped_judgement == chosen:
                    total_correct += 1

    accuracy = total_correct / total_evaluated if total_evaluated > 0 else 0.0

    return TaskResult(
        task_name=task_name,
        total_pairs=total_pairs,
        evaluated_pairs=evaluated_pairs,
        missing_pairs=missing_pairs,
        total_correct=total_correct,
        total_evaluated=total_evaluated,
        accuracy=accuracy,
    )


def print_task_result(result: TaskResult):
    """Print formatted results for a task."""
    print(f"\n{'='*50}")
    print(f"Task: {result.task_name}")
    print(f"{'='*50}")
    print(f"Total pairs: {result.total_pairs}")
    print(f"Evaluated: {result.evaluated_pairs}")
    print(f"Missing: {result.missing_pairs}")
    print(f"Accuracy: {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
    print(f"{'='*50}")


def evaluate_single_task(
    predictions_path: str, benchmark_path: str, task_name: str
) -> TaskResult:
    """Evaluate a single task."""
    print(f"Loading predictions from {predictions_path}...")
    predictions = load_json(predictions_path)

    print(f"Loading benchmark from {benchmark_path}...")
    benchmark_data = load_json(benchmark_path)

    result = compute_task_accuracy(predictions, benchmark_data, task_name)
    print_task_result(result)

    return result


def evaluate_all_tasks(
    prediction_files: List[str],
    benchmark_dir: str,
) -> List[TaskResult]:
    """Evaluate all tasks.

    Args:
        prediction_files: List of prediction file paths (task1, task2, task3, task4).
        benchmark_dir: Directory containing benchmark files.

    Returns:
        List of TaskResult for each task.
    """
    # Task configurations
    tasks = [
        {"name": "task1_image", "benchmark": "t2i.json"},
        {"name": "task2_edit", "benchmark": "edit.json"},
        {"name": "task3_interleaved", "benchmark": "interleaved.json"},
        {"name": "task4_reasoning", "benchmark": "reasoning.json"},
    ]

    results = []

    for i, config in enumerate(tasks):
        if i >= len(prediction_files):
            print(f"Warning: No prediction file provided for {config['name']}")
            continue

        benchmark_path = os.path.join(benchmark_dir, config["benchmark"])
        predictions_path = prediction_files[i]

        if not os.path.exists(benchmark_path):
            print(f"Warning: Benchmark file not found: {benchmark_path}")
            continue

        if not os.path.exists(predictions_path):
            print(f"Warning: Prediction file not found: {predictions_path}")
            continue

        print(f"\nEvaluating {config['name']}...")
        predictions = load_json(predictions_path)
        benchmark_data = load_json(benchmark_path)

        result = compute_task_accuracy(predictions, benchmark_data, config["name"])
        print_task_result(result)
        results.append(result)

    # Print summary
    if results:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        print(f"{'Task':<25} {'Accuracy':<12} {'Missing'}")
        print(f"{'-'*50}")

        total_correct = 0
        total_evaluated = 0

        for r in results:
            print(f"{r.task_name:<25} {r.accuracy*100:>6.2f}%      {r.missing_pairs}")
            total_correct += r.total_correct
            total_evaluated += r.total_evaluated

        overall_accuracy = (
            total_correct / total_evaluated if total_evaluated > 0 else 0.0
        )
        print(f"{'-'*50}")
        print(f"{'Overall':<25} {overall_accuracy*100:>6.2f}%")
        print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy scores for MMRB2 benchmark predictions."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["image", "edit", "interleaved", "reasoning", "all"],
        help="Task to evaluate (image, edit, interleaved, reasoning, or all)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to prediction JSON file(s). For 'all', provide 4 files in order: task1, task2, task3, task4",
    )
    # Default to benchmark directory relative to this script
    default_benchmark_dir = str(
        Path(__file__).resolve().parent.parent.parent / "benchmark"
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=default_benchmark_dir,
        help="Directory containing benchmark files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)",
    )

    args = parser.parse_args()

    # Task to benchmark file mapping
    task_to_benchmark = {
        "image": "t2i.json",
        "edit": "edit.json",
        "interleaved": "interleaved.json",
        "reasoning": "reasoning.json",
    }

    if args.task == "all":
        if len(args.predictions) < 4:
            print(
                f"Warning: Expected 4 prediction files for 'all' tasks, got {len(args.predictions)}"
            )

        results = evaluate_all_tasks(args.predictions, args.benchmark_dir)

        if args.output:
            output_data = {
                "tasks": [
                    {
                        "task_name": r.task_name,
                        "accuracy": r.accuracy,
                        "total_pairs": r.total_pairs,
                        "evaluated_pairs": r.evaluated_pairs,
                        "missing_pairs": r.missing_pairs,
                    }
                    for r in results
                ]
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
    else:
        if len(args.predictions) != 1:
            parser.error(
                "Provide exactly one prediction file for single task evaluation"
            )

        benchmark_path = os.path.join(args.benchmark_dir, task_to_benchmark[args.task])
        result = evaluate_single_task(args.predictions[0], benchmark_path, args.task)

        if args.output:
            output_data = {
                "task_name": result.task_name,
                "accuracy": result.accuracy,
                "total_pairs": result.total_pairs,
                "evaluated_pairs": result.evaluated_pairs,
                "missing_pairs": result.missing_pairs,
            }
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
