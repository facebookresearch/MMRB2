#!/usr/bin/env python3
"""
MMRB2 Benchmark - Multi-GPU/Process Evaluation

This script enables parallel processing of pairwise evaluations using
multiple GPUs (for local models) or multiple processes (for API models).

Usage:
    python multi_gpu_evaluate.py --evaluator_name gpt41-pairwise --task_type image \
                                  --pairs_path /path/to/pairs.json --output_path ./outputs/results.json \
                                  --n_gpu 8
"""

import argparse
import json
import multiprocessing
import os
from multiprocessing import Process
from typing import List

from evaluate import process_pairs_with_evaluator


def process_pairs_worker_wrapper(
    evaluator_name: str,
    task_type: str,
    pairs_path: str,
    output_path: str,
    n: int,
    device_id: int,
    worker_id: int,
    base_dir: str = None,
):
    """Worker wrapper function that calls the evaluate.py logic.

    Args:
        evaluator_name: Name of the evaluator to use.
        task_type: Task type to use.
        pairs_path: Path to the pairs file for this worker.
        output_path: Path to save results.
        n: Number of judgements to generate.
        device_id: Device ID to use (None for API-based evaluators).
        worker_id: ID of this worker process.
        base_dir: Base directory of the pairs file for relative paths.
    """
    print(f"[Worker {worker_id}] Starting on device {device_id}")

    process_pairs_with_evaluator(
        evaluator_name=evaluator_name,
        task_type=task_type,
        pairs_path=pairs_path,
        output_path=output_path,
        n=n,
        max_samples=None,
        device_id=device_id,
        base_dir=base_dir,
    )

    print(f"[Worker {worker_id}] Done!")


def merge_results(partial_result_paths: List[str], final_output_path: str):
    """Merge results from multiple workers into a single file.

    Args:
        partial_result_paths: List of paths to partial result files.
        final_output_path: Path to save the merged results.
    """
    print(f"\nMerging results from {len(partial_result_paths)} workers...")

    merged_results = {}
    for path in partial_result_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                partial_results = json.load(f)
                merged_results.update(partial_results)
            print(f"  Loaded {len(partial_results)} results from {path}")
        else:
            print(f"  Warning: {path} not found, skipping...")

    # Save merged results
    output_dir = os.path.dirname(final_output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged results ({len(merged_results)} total) to {final_output_path}...")
    with open(final_output_path, "w") as f:
        json.dump(merged_results, f, indent=2)

    print(f"Done! Merged {len(merged_results)} pairs from {len(partial_result_paths)} workers")
    return merged_results


def multi_gpu_evaluate(
    evaluator_name: str,
    task_type: str,
    pairs_path: str,
    output_path: str,
    n_gpu: int = 8,
    n: int = 1,
    max_samples: int = None,
    keep_partial_results: bool = False,
):
    """Process pairs with an evaluator using multiple GPUs/processes in parallel.

    Args:
        evaluator_name: Name of the evaluator to use.
        task_type: Task type to use.
        pairs_path: Path to the pairs file.
        output_path: Path to the output file.
        n_gpu: Number of GPUs/processes to use.
        n: Number of judgements to generate per pair.
        max_samples: Maximum number of samples to process.
        keep_partial_results: Whether to keep partial result files after merging.
    """
    from evaluator_apis.evaluators import EVALUATORS

    if evaluator_name not in EVALUATORS:
        raise ValueError(f"Evaluator {evaluator_name} not found")

    is_api_based = EVALUATORS[evaluator_name]["is_api_based"]

    real_base_dir = os.path.dirname(pairs_path)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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

    # Divide pairs into chunks for each GPU/process
    chunk_size = len(pairs) // n_gpu
    remainder = len(pairs) % n_gpu

    pair_chunks = []
    start_idx = 0
    for i in range(n_gpu):
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        pair_chunks.append(pairs[start_idx:end_idx])
        start_idx = end_idx

    print(f"\nDividing {len(pairs)} pairs into {n_gpu} chunks:")
    for i, chunk in enumerate(pair_chunks):
        print(f"  Worker {i}: {len(chunk)} pairs")

    # Create temporary pair files and output paths for each worker
    base_name = os.path.splitext(output_path)[0]
    extension = os.path.splitext(output_path)[1]

    temp_pairs_dir = os.path.dirname(output_path)
    if not temp_pairs_dir:
        temp_pairs_dir = "."

    output_filename_base = os.path.basename(base_name)

    temp_pair_paths = []
    partial_output_paths = []

    for i in range(n_gpu):
        temp_pair_path = os.path.join(
            temp_pairs_dir,
            f".temp_pairs_{output_filename_base}_worker_{i}.json",
        )
        temp_pair_paths.append(temp_pair_path)
        partial_output_paths.append(f"{base_name}_worker_{i}{extension}")

        # Write temporary pair file for this worker
        temp_data = {"pairs": pair_chunks[i]}
        with open(temp_pair_path, "w") as f:
            json.dump(temp_data, f, indent=2)
        print(f"Created temporary pairs file: {temp_pair_path}")

    # Start worker processes
    processes = []
    for i in range(n_gpu):
        if is_api_based:
            device_id = None
            print(f"\nStarting worker {i} for API-based evaluator (no device_id)")
        else:
            device_id = i % 8  # Cycle through GPUs 0-7
            print(f"\nStarting worker {i} on GPU {device_id}")

        p = Process(
            target=process_pairs_worker_wrapper,
            args=(
                evaluator_name,
                task_type,
                temp_pair_paths[i],
                partial_output_paths[i],
                n,
                device_id,
                i,
                real_base_dir,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    print("\nWaiting for all workers to complete...")
    for i, p in enumerate(processes):
        p.join()
        print(f"Worker {i} finished")

    # Merge results
    merged_results = merge_results(partial_output_paths, output_path)

    # Clean up temporary pair files
    print("\nCleaning up temporary pair files...")
    for path in temp_pair_paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed {path}")

    # Optionally clean up partial results
    if not keep_partial_results:
        print("\nCleaning up partial result files...")
        for path in partial_output_paths:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed {path}")

    print(f"\nAll done! Final results saved to {output_path}")
    return merged_results


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Multi-GPU/process evaluation for MMRB2 benchmark."
    )
    parser.add_argument(
        "--evaluator_name",
        type=str,
        required=True,
        help="Name of the evaluator to use",
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
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs/processes to use (default: 1)",
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
        "--keep_partial_results",
        action="store_true",
        help="Keep partial result files after merging (default: False)",
    )

    args = parser.parse_args()

    multi_gpu_evaluate(
        evaluator_name=args.evaluator_name,
        task_type=args.task_type,
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        n_gpu=args.n_gpu,
        n=args.n,
        max_samples=args.max_samples,
        keep_partial_results=args.keep_partial_results,
    )


if __name__ == "__main__":
    main()

