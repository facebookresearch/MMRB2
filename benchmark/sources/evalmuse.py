# Copyright: Meta Platforms, Inc. and affiliates
"""
EvalMuse source module.

Downloads from HuggingFace: https://huggingface.co/datasets/DY-Evalab/EvalMuse
File: test.json

ID format: string (prompt_id field, e.g., "03251")
"""

import json
import os
import subprocess
from typing import Any, Dict, Optional

from tqdm import tqdm

SOURCE_NAME = "evalmuse"
HF_URL = (
    "https://huggingface.co/datasets/DY-Evalab/EvalMuse/resolve/main/test.json"
)


def download(data_dir: str, force: bool = False) -> str:
    """
    Download EvalMuse dataset from HuggingFace.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the processed JSONL file
    """
    output_dir = os.path.join(data_dir, SOURCE_NAME)
    raw_file = os.path.join(output_dir, "test.json")
    output_file = os.path.join(output_dir, "prompts.jsonl")

    # Check if already processed
    if os.path.exists(output_file) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {output_file}")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    # Download test.json if not exists
    if not os.path.exists(raw_file) or force:
        print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_URL}")
        subprocess.run(
            ["curl", "-sL", HF_URL, "-o", raw_file],
            check=True,
        )

    # Load and process
    print(f"[{SOURCE_NAME}] Processing test.json...")
    with open(raw_file, "r") as f:
        data = json.load(f)

    # Extract unique prompts (same prompt_id may repeat for different images)
    seen_ids = set()
    prompts = []

    for ex in tqdm(data, desc=f"Processing {SOURCE_NAME}"):
        prompt_id = ex["prompt_id"]
        if prompt_id in seen_ids:
            continue
        seen_ids.add(prompt_id)

        prompt_data = {
            "id": prompt_id,
            "prompt_content": [["text", ex["prompt"]]],
            "metadata": {
                "id": prompt_id,
                "type": ex["type"],
                "elements": list(ex["element_score"].keys()),
            },
        }
        prompts.append(prompt_data)

    # Save to JSONL
    with open(output_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"[{SOURCE_NAME}] Saved {len(prompts)} prompts to: {output_file}")
    return output_file


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all prompts into a lookup dictionary.

    Args:
        data_dir: Directory containing downloaded data

    Returns:
        Dictionary mapping id -> {prompt_content, metadata}
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
            # Use string ID for lookup
            prompts[str(data["id"])] = {
                "prompt_content": data["prompt_content"],
                "metadata": data["metadata"],
            }

    return prompts


def get_prompt_by_id(
    data_dir: str, prompt_id: str, **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Get prompt data for a specific ID.

    Args:
        data_dir: Directory containing downloaded data
        prompt_id: The prompt ID (string like "03251")

    Returns:
        Dictionary with prompt_content and metadata, or None if not found
    """
    prompts = load_prompts(data_dir)
    return prompts.get(str(prompt_id))
