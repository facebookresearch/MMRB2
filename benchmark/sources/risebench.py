# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
RISEBench source module.

RISEBench is a benchmark for reasoning-enhanced image editing.
HuggingFace: PhoenixZ/RISEBench

Data structure:
- JSON: datav2_total_w_subtask.json
- Images: data.zip containing category_images/*.png
"""

import json
import os
import shutil
import zipfile
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from tqdm import tqdm

SOURCE_NAME = "risebench"
HF_REPO = "PhoenixZ/RISEBench"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download RISEBench data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    json_path = os.path.join(source_dir, "datav2_total_w_subtask.json")
    images_dir = os.path.join(source_dir, "data")

    if os.path.exists(json_path) and os.path.exists(images_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download JSON file
    json_hf = hf_hub_download(
        HF_REPO, "datav2_total_w_subtask.json", repo_type="dataset"
    )
    shutil.copy(json_hf, json_path)
    print(f"[{SOURCE_NAME}] Downloaded datav2_total_w_subtask.json")

    # Download and extract data.zip
    zip_hf = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset")
    print(f"[{SOURCE_NAME}] Extracting data.zip...")

    with zipfile.ZipFile(zip_hf, "r") as z:
        z.extractall(source_dir)

    print(f"[{SOURCE_NAME}] Extracted images to {images_dir}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from RISEBench.

    The ID (index field) is unique in the source data.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    json_path = os.path.join(source_dir, "datav2_total_w_subtask.json")
    data_dir_path = os.path.join(source_dir, "data")
    output_images_dir = os.path.join(source_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load the JSON data
    with open(json_path, "r") as f:
        dataset = json.load(f)

    prompts = {}

    for row in tqdm(dataset, desc=f"Loading {SOURCE_NAME}"):
        prompt_id = row["index"]  # e.g., 'temporal_reasoning_1'
        category = row["category"]
        instruction = row["instruction"]
        image_rel_path = row["image"]  # e.g., 'temporal_reasoning_images/1.png'

        # Source image path
        src_image_path = os.path.join(data_dir_path, image_rel_path)
        if not os.path.exists(src_image_path):
            continue

        # Create output image filename (flatten the path)
        img_filename = image_rel_path.replace("/", "_")
        dst_image_path = os.path.join(output_images_dir, img_filename)
        rel_img_path = os.path.join(SOURCE_NAME, "images", img_filename)

        # Copy image if not already present
        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)

        # Use ID as lookup key (IDs are unique)
        lookup_key = prompt_id

        prompts[lookup_key] = {
            "id": prompt_id,
            "lookup_key": lookup_key,
            "prompt_content": [
                ["image", rel_img_path],
                ["text", instruction],
            ],
            "metadata": {
                "id": prompt_id,
                "category": category,
                "reference": row.get("reference", ""),
                "reference_image": None,  # Not storing reference image path
            },
        }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
