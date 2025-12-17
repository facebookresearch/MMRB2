# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
RealUnify GEU source module (for reasoning task).

HuggingFace: DogNeverSleep/RealUnify
File: GEU_direct.json (image + multiple choice questions)

ID format: GEU_{index}_{task_type}
"""

import json
import os
import subprocess
from typing import Any, Dict

from tqdm import tqdm

SOURCE_NAME = "realunify"
HF_BASE = "https://huggingface.co/datasets/DogNeverSleep/RealUnify/resolve/main"
GEU_URL = f"{HF_BASE}/GEU_direct.json"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download RealUnify GEU dataset from HuggingFace.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the source directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    raw_file = os.path.join(source_dir, "GEU_direct.json")
    prompts_file = os.path.join(source_dir, "prompts.jsonl")
    images_dir = os.path.join(source_dir, "images")

    if (
        os.path.exists(prompts_file)
        and os.path.exists(images_dir)
        and not force
    ):
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Download GEU_direct.json
    if not os.path.exists(raw_file) or force:
        print(f"[{SOURCE_NAME}] Downloading GEU_direct.json...")
        subprocess.run(
            ["curl", "-sL", GEU_URL, "-o", raw_file],
            check=True,
        )

    # Load data
    with open(raw_file, "r") as f:
        data = json.load(f)

    print(f"[{SOURCE_NAME}] Processing {len(data)} samples...")

    # Process and download images
    prompts = []
    for ex in tqdm(data, desc=f"Processing {SOURCE_NAME}"):
        # ID format: GEU_{index}_{task_type}
        prompt_id = f"GEU_{ex['index']}_{ex['task_type']}"

        # Build question with options
        option_letters = ["A", "B", "C", "D", "E"]
        text_query = (
            ex["question"]
            + "\nOption:\n"
            + "\n".join(
                f"{option_letters[i]}. {ex['option'][i]}"
                for i in range(len(ex["option"]))
            )
        )

        # Image path from original: ./image/task_type/subtask/images/img_xxx.jpg
        # We need to download and store locally
        img_rel_path = ex["input_image"].lstrip("./")  # Remove leading ./
        img_url = f"{HF_BASE}/{img_rel_path}"

        # Create local image path
        img_filename = img_rel_path.replace("/", "_")
        local_img_path = os.path.join(images_dir, img_filename)
        rel_img_path = os.path.join(SOURCE_NAME, "images", img_filename)

        # Download image if not exists
        if not os.path.exists(local_img_path):
            try:
                subprocess.run(
                    ["curl", "-sL", img_url, "-o", local_img_path],
                    check=True,
                    timeout=30,
                )
            except Exception as e:
                print(
                    f"[{SOURCE_NAME}] Warning: Failed to download {img_url}: {e}"
                )
                continue

        prompts.append(
            {
                "id": prompt_id,
                "prompt_content": [
                    ["image", rel_img_path],
                    ["text", text_query],
                ],
                "metadata": {
                    "id": prompt_id,
                    "task_type": ex["task_type"],
                    "answer": ex["answer"],
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
