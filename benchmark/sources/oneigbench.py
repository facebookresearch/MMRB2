# Copyright: Meta Platforms, Inc. and affiliates
"""
OneIG-Bench source module.

Downloads from HuggingFace: OneIG-Bench/OneIG-Bench

NOTE: IDs in this dataset are NOT unique - they repeat across categories.
We use a compound key of (id, category) for lookups.
"""

import json
import os
from typing import Any, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

SOURCE_NAME = "oneigbench"
HF_DATASET = "OneIG-Bench/OneIG-Bench"


def make_lookup_key(prompt_id: str, category: str) -> str:
    """Create a compound lookup key from id and category."""
    return f"{prompt_id}_{category}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download OneIG-Bench dataset from HuggingFace.

    Args:
        data_dir: Directory to save processed data
        force: If True, re-download even if data exists

    Returns:
        Path to the processed JSONL file
    """
    output_dir = os.path.join(data_dir, SOURCE_NAME)
    output_file = os.path.join(output_dir, "prompts.jsonl")

    # Check if already downloaded
    if os.path.exists(output_file) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {output_file}")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_DATASET}")

    # Load dataset
    dataset = load_dataset(
        HF_DATASET,
        "OneIG-Bench",
        split="train",
        cache_dir=os.path.join(output_dir, "cache"),
    )

    print(f"[{SOURCE_NAME}] Processing {len(dataset)} samples...")

    # Process and save
    with open(output_file, "w") as f:
        for row in tqdm(dataset, desc=f"Processing {SOURCE_NAME}"):
            prompt_data = {
                "id": row["id"],
                "category": row["category"],
                "lookup_key": make_lookup_key(row["id"], row["category"]),
                "prompt_content": [["text", row["prompt_en"]]],
                "metadata": {
                    "id": row["id"],
                    "category": row["category"],
                    "class": row["class"],
                    "prompt_length": row["prompt_length"],
                    "type": row["type"],
                },
            }
            f.write(json.dumps(prompt_data) + "\n")

    print(f"[{SOURCE_NAME}] Saved {len(dataset)} prompts to: {output_file}")
    return output_file


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all prompts into a lookup dictionary.

    Uses compound key (id_category) since IDs are not unique across categories.

    Args:
        data_dir: Directory containing downloaded data

    Returns:
        Dictionary mapping "id_category" -> {prompt_content, metadata}
    """
    output_file = os.path.join(data_dir, SOURCE_NAME, "prompts.jsonl")

    if not os.path.exists(output_file):
        raise FileNotFoundError(
            f"Prompts file not found: {output_file}. Run download() first."
        )

    prompts = {}
    with open(output_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # Use compound key
            key = make_lookup_key(data["id"], data["category"])
            prompts[key] = {
                "prompt_content": data["prompt_content"],
                "metadata": data["metadata"],
            }

    return prompts


def get_prompt_by_id(
    data_dir: str, prompt_id: str, category: str = None
) -> Optional[Dict[str, Any]]:
    """
    Get prompt data for a specific ID and category.

    Args:
        data_dir: Directory containing downloaded data
        prompt_id: The prompt ID to look up
        category: The category (required for unique lookup)

    Returns:
        Dictionary with prompt_content and metadata, or None if not found
    """
    prompts = load_prompts(data_dir)
    if category:
        key = make_lookup_key(prompt_id, category)
        return prompts.get(key)
    return None
