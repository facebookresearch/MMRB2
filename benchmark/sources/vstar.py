# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
V-Star source module.

HuggingFace: craigwu/vstar_bench
Paper: V-Star Benchmark for Visual Reasoning

Data structure:
- test_questions.jsonl - questions with image paths
- {category}/*.jpg - images organized by category
"""

import json
import os
import shutil
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from tqdm import tqdm

SOURCE_NAME = "vstar"
HF_REPO = "craigwu/vstar_bench"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download V-Star data from HuggingFace.

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
    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download test_questions.jsonl
    data_path = hf_hub_download(HF_REPO, "test_questions.jsonl", repo_type="dataset")
    shutil.copy(data_path, os.path.join(source_dir, "test_questions.jsonl"))
    print(f"[{SOURCE_NAME}] Downloaded test_questions.jsonl")

    # Load questions to find which images we need
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Create mapping from image path to HF path
    needed_images = set(row["image"] for row in data)

    print(f"[{SOURCE_NAME}] Downloading {len(needed_images)} images...")
    for img_path in tqdm(needed_images, desc=f"Downloading {SOURCE_NAME}"):
        # img_path like: direct_attributes/sa_58805.jpg
        try:
            img_hf = hf_hub_download(HF_REPO, img_path, repo_type="dataset")
            # Create category subdirectory
            category = img_path.split("/")[0]
            category_dir = os.path.join(images_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            dst_path = os.path.join(images_dir, img_path)
            if not os.path.exists(dst_path):
                shutil.copy(img_hf, dst_path)
        except Exception as e:
            print(f"[{SOURCE_NAME}] Warning: Failed to download {img_path}: {e}")

    # Process prompts
    prompts = []
    for row in data:
        prompt_id = str(row["question_id"])
        img_path = row["image"]
        rel_img_path = os.path.join(SOURCE_NAME, "images", img_path)

        # Clean up question text
        question = row["text"].replace(
            "Answer with the option's letter from the given choices directly.",
            "",
        )

        prompts.append(
            {
                "id": prompt_id,
                "prompt_content": [
                    ["image", rel_img_path],
                    ["text", question],
                ],
                "metadata": {
                    "id": prompt_id,
                    "category": row["category"],
                    "answer": row["label"],
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
