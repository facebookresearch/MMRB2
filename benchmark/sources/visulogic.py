# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
VisuLogic source module.

HuggingFace: VisuLogic/VisuLogic
Paper: Visual Reasoning Benchmark for Multi-modal LLMs

Data structure:
- data.jsonl - questions with image paths
- images.zip - all images
"""

import json
import os
import shutil
import zipfile
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from tqdm import tqdm

SOURCE_NAME = "visulogic"
HF_REPO = "VisuLogic/VisuLogic"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download VisuLogic data from HuggingFace.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the source directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_file = os.path.join(source_dir, "prompts.jsonl")
    images_dir = os.path.join(source_dir, "images")

    if os.path.exists(prompts_file) and os.path.exists(images_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download data.jsonl
    data_path = hf_hub_download(HF_REPO, "data.jsonl", repo_type="dataset")
    shutil.copy(data_path, os.path.join(source_dir, "data.jsonl"))
    print(f"[{SOURCE_NAME}] Downloaded data.jsonl")

    # Download images.zip
    zip_path = hf_hub_download(HF_REPO, "images.zip", repo_type="dataset")
    print(f"[{SOURCE_NAME}] Downloaded images.zip")

    # Extract images
    os.makedirs(images_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in tqdm(z.namelist(), desc=f"Extracting {SOURCE_NAME}"):
            if member.endswith("/"):
                continue
            # Extract to images directory, flattening the path
            filename = os.path.basename(member)
            if filename:
                with z.open(member) as src:
                    dst_path = os.path.join(images_dir, filename)
                    with open(dst_path, "wb") as dst:
                        dst.write(src.read())

    # Process prompts
    with open(os.path.join(source_dir, "data.jsonl"), "r") as f:
        data = [json.loads(line) for line in f]

    prompts = []
    for row in data:
        prompt_id = row["id"]
        img_filename = os.path.basename(row["image_path"])
        rel_img_path = os.path.join(SOURCE_NAME, "images", img_filename)

        prompts.append(
            {
                "id": prompt_id,
                "prompt_content": [
                    ["image", rel_img_path],
                    ["text", row["question"]],
                ],
                "metadata": {
                    "id": prompt_id,
                    "answer": row["label"],
                    "tag": row["tag"],
                },
            }
        )

    # Save prompts
    with open(prompts_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"[{SOURCE_NAME}] Saved {len(prompts)} prompts to {prompts_file}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all prompts into a lookup dictionary.

    Args:
        data_dir: Directory containing downloaded data

    Returns:
        Dictionary mapping id -> prompt data
    """
    prompts_file = os.path.join(data_dir, SOURCE_NAME, "prompts.jsonl")

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_file}. Run download() first."
        )

    prompts = {}
    with open(prompts_file, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts[data["id"]] = {
                "id": data["id"],
                "lookup_key": data["id"],
                "prompt_content": data["prompt_content"],
                "metadata": data["metadata"],
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
