# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
BLINK source module.

BLINK is a benchmark for multimodal reasoning across various visual tasks.
HuggingFace: BLINK-Benchmark/BLINK

Data structure:
- Downloads images directly from the HuggingFace dataset
- Processes into prompts with image paths
"""

import json
import os
import shutil
from typing import Any, Dict, Optional, Set

from datasets import Image, load_dataset
from tqdm import tqdm

SOURCE_NAME = "blink"
HF_DATASET = "BLINK-Benchmark/BLINK"

# BLINK has multiple configs that need to be loaded separately
BLINK_CONFIGS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


def download(data_dir: str, force: bool = False) -> str:
    """
    Download BLINK data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_file = os.path.join(source_dir, "prompts.jsonl")
    images_dir = os.path.join(source_dir, "images")

    if os.path.exists(prompts_file) and os.path.exists(images_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_DATASET}")

    # Load all configs and combine them
    all_rows = []
    for config in BLINK_CONFIGS:
        print(f"[{SOURCE_NAME}] Loading config: {config}")
        try:
            dataset = load_dataset(HF_DATASET, config, split="val")
            # Cast images to not decode for raw access
            dataset = dataset.cast_column("image_1", Image(decode=False))
            dataset = dataset.cast_column("image_2", Image(decode=False))
            for row in dataset:
                # Add sub_task from config name if not present
                if "sub_task" not in row or not row["sub_task"]:
                    row["sub_task"] = config
                all_rows.append(row)
            print(f"[{SOURCE_NAME}]   Loaded {len(dataset)} samples from {config}")
        except Exception as e:
            print(f"[{SOURCE_NAME}]   Warning: Failed to load config {config}: {e}")

    print(f"[{SOURCE_NAME}] Processing {len(all_rows)} total samples...")

    prompts = []
    for idx, row in enumerate(tqdm(all_rows, desc=f"Processing {SOURCE_NAME}")):
        # Build ID: val_{Category}_{idx}
        category = row["sub_task"]
        prompt_id = f"val_{category}_{idx}"

        # Process images and question
        prompt_content = []
        image_count = 0

        # Image 1
        if row["image_1"] is not None and row["image_1"].get("bytes"):
            img_filename = f"{prompt_id}_1.jpg"
            img_path = os.path.join(images_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(row["image_1"]["bytes"])
            rel_path = os.path.join(SOURCE_NAME, "images", img_filename)
            prompt_content.append(["image", rel_path])
            image_count += 1

        # Image 2 (if exists)
        if row["image_2"] is not None and row["image_2"].get("bytes"):
            img_filename = f"{prompt_id}_2.jpg"
            img_path = os.path.join(images_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(row["image_2"]["bytes"])
            rel_path = os.path.join(SOURCE_NAME, "images", img_filename)
            prompt_content.append(["image", rel_path])
            image_count += 1

        # Add question text
        question = row["question"]
        choices = row.get("choices", [])
        if choices:
            question += "\nChoices:\n" + "\n".join(
                f"({chr(65+i)}) {c}" for i, c in enumerate(choices)
            )
        prompt_content.append(["text", question])

        prompts.append(
            {
                "id": prompt_id,
                "lookup_key": prompt_id,
                "prompt_content": prompt_content,
                "metadata": {
                    "id": prompt_id,
                    "sub_task": category,
                    "question": row["question"],
                    "choices": choices,
                    "answer": row.get("answer", ""),
                },
            }
        )

    # Save prompts
    with open(prompts_file, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"[{SOURCE_NAME}] Saved {len(prompts)} prompts to {prompts_file}")
    return source_dir


def load_prompts(
    data_dir: str, required_keys: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from BLINK.

    Args:
        data_dir: Base directory containing downloaded data
        required_keys: Optional set of lookup keys to filter

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_file = os.path.join(source_dir, "prompts.jsonl")

    if not os.path.exists(prompts_file):
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_file}. Run download() first."
        )

    prompts = {}
    with open(prompts_file, "r") as f:
        for line in f:
            data = json.loads(line)
            lookup_key = data["lookup_key"]

            # Filter if required_keys provided
            if required_keys is not None and lookup_key not in required_keys:
                continue

            prompts[lookup_key] = {
                "id": data["id"],
                "lookup_key": lookup_key,
                "prompt_content": data["prompt_content"],
                "metadata": data["metadata"],
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
