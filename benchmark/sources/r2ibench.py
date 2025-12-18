# Copyright: Meta Platforms, Inc. and affiliates
"""
R2I-Bench source module.

Downloads from GitHub: https://github.com/PLUM-Lab/R2I-Bench.git
Commit: 0cf0197d797c117ea4f9fc3b2a87a72bf7910d51

ID format in balanced.json: {category}_{Subcategory}_{id}
Example: logical_Disjunctive_73
"""

import csv
import json
import os
import subprocess
from typing import Any, Dict, Optional

from tqdm import tqdm

SOURCE_NAME = "r2ibench"
GITHUB_URL = "https://github.com/PLUM-Lab/R2I-Bench.git"
COMMIT_HASH = "0cf0197d797c117ea4f9fc3b2a87a72bf7910d51"


def make_lookup_key(category: str, subcategory: str, row_id: str) -> str:
    """Create lookup key matching the balanced.json format."""
    return f"{category}_{subcategory}_{row_id}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download R2I-Bench dataset from GitHub.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the processed JSONL file
    """
    output_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(output_dir, "R2I-Bench")
    output_file = os.path.join(output_dir, "prompts.jsonl")

    # Check if already processed
    if os.path.exists(output_file) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {output_file}")
        return output_file

    os.makedirs(output_dir, exist_ok=True)

    # Clone repo if not exists
    if not os.path.exists(repo_dir):
        print(f"[{SOURCE_NAME}] Cloning from GitHub: {GITHUB_URL}")
        subprocess.run(
            ["git", "clone", GITHUB_URL, repo_dir],
            check=True,
        )

    # Checkout specific commit
    print(f"[{SOURCE_NAME}] Checking out commit: {COMMIT_HASH}")
    subprocess.run(
        ["git", "-C", repo_dir, "checkout", COMMIT_HASH],
        check=True,
        capture_output=True,
    )

    # Find all CSV files
    prompts_dir = os.path.join(repo_dir, "data", "prompts")
    csv_files = []
    for root, dirs, files in os.walk(prompts_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    print(f"[{SOURCE_NAME}] Found {len(csv_files)} CSV files")

    # Process all CSV files
    prompts = []
    for csv_path in tqdm(csv_files, desc=f"Processing {SOURCE_NAME}"):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build lookup key
                category = row["Category"].lower()
                subcategory = row["Subcategory"]
                row_id = row["id"]
                lookup_key = make_lookup_key(category, subcategory, row_id)

                prompt_data = {
                    "lookup_key": lookup_key,
                    "prompt_content": [["text", row["Prompt"]]],
                    "metadata": {
                        "id": lookup_key,
                        "category": category,
                        "subcategory": subcategory,
                        "reference_caption": row.get("Reference Caption", ""),
                        "assessment_point": row.get("Assessment Point", ""),
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
        Dictionary mapping lookup_key -> {prompt_content, metadata}
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
            prompts[data["lookup_key"]] = {
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
        prompt_id: The prompt ID (format: {category}_{subcategory}_{id})

    Returns:
        Dictionary with prompt_content and metadata, or None if not found
    """
    prompts = load_prompts(data_dir)
    return prompts.get(prompt_id)
