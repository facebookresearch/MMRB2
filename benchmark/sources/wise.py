# Copyright: Meta Platforms, Inc. and affiliates
"""
WISE source module.

Downloads from GitHub: https://github.com/PKU-YuanGroup/WISE.git
Commit: aeef292aaf1715bcd9cc5dc28810197ad8881570

NOTE: WISE has both original and rewrite versions with the SAME prompt_id.
The balanced dataset uses a mix of both. We use id_version as lookup key.
"""

import json
import os
import subprocess
from typing import Any, Dict, Optional

from tqdm import tqdm

SOURCE_NAME = "wise"
GITHUB_URL = "https://github.com/PKU-YuanGroup/WISE.git"
COMMIT_HASH = "aeef292aaf1715bcd9cc5dc28810197ad8881570"


def make_lookup_key(prompt_id: int, version: str) -> str:
    """Create lookup key from id and version."""
    return f"{prompt_id}_{version}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download WISE dataset from GitHub.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the processed JSONL file
    """
    output_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(output_dir, "WISE")
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

    # Find all JSON files in data directory (include both original and rewrite)
    # The balanced dataset may use either version, so we need both
    data_dir_path = os.path.join(repo_dir, "data")
    json_files = [f for f in os.listdir(data_dir_path) if f.endswith(".json")]

    print(f"[{SOURCE_NAME}] Found {len(json_files)} JSON files")

    # Process all JSON files
    prompts = []
    for json_file in tqdm(json_files, desc=f"Processing {SOURCE_NAME}"):
        file_path = os.path.join(data_dir_path, json_file)
        with open(file_path, "r") as f:
            data = json.load(f)

        # Determine version from filename
        version = "rewrite" if "_rewrite" in json_file else "original"

        for ex in data:
            prompt_id = ex["prompt_id"]
            prompt_text = ex["Prompt"]
            lookup_key = make_lookup_key(prompt_id, version)
            prompt_data = {
                "id": prompt_id,
                "version": version,
                "lookup_key": lookup_key,
                "prompt_content": [["text", prompt_text]],
                "metadata": {
                    "id": prompt_id,
                    "category": ex["Category"],
                    "subcategory": ex["Subcategory"],
                    "explanation": ex["Explanation"],
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

    Uses id_version as key since IDs are not unique (original + rewrite).

    Args:
        data_dir: Directory containing downloaded data

    Returns:
        Dictionary mapping lookup_key (id_version) -> {prompt_content, metadata}
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
            # Use id_version as key
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
        prompt_id: The prompt ID (integer as string)

    Returns:
        Dictionary with prompt_content and metadata, or None if not found
    """
    prompts = load_prompts(data_dir)
    return prompts.get(str(prompt_id))
