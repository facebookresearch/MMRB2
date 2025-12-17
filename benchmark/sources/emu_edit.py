# Copyright: Meta Platforms, Inc. and affiliates
"""
Emu-Edit source module.

Downloads from HuggingFace: facebook/emu_edit_test_set

ID format: integer (idx field)
Additional key fields: task, split

Downloads both text prompts AND input images.
"""

import json
import os
from typing import Any, Dict, Optional

from datasets import Image, load_dataset
from tqdm import tqdm

SOURCE_NAME = "emu-edit"
HF_DATASET = "facebook/emu_edit_test_set"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download Emu-Edit dataset from HuggingFace.

    Downloads both text prompts and input images.

    Args:
        data_dir: Directory to save data
        force: If True, re-download even if data exists

    Returns:
        Path to the processed JSONL file
    """
    output_dir = os.path.join(data_dir, SOURCE_NAME)
    output_file = os.path.join(output_dir, "prompts.jsonl")
    image_dir = os.path.join(output_dir, "images")

    # Check if already processed
    if os.path.exists(output_file) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {output_file}")
        return output_file

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_DATASET}")

    # Load dataset with images (decode=False to get raw bytes, avoids Pillow dependency)
    dataset = load_dataset(HF_DATASET)

    prompts = []
    for split in ["test", "validation"]:
        if split not in dataset:
            continue

        # Cast images to not decode (returns dict with 'bytes' and 'path')
        split_data = dataset[split].cast_column("image", Image(decode=False))

        print(f"[{SOURCE_NAME}] Processing {split} split (with images)...")
        for ex in tqdm(split_data, desc=f"Processing {split}"):
            # Create unique ID: combines idx, task, split
            prompt_id = ex["idx"]
            lookup_key = f"{split}_{ex['task']}_{prompt_id}"

            # Save input image from raw bytes
            image_filename = f"{split}_{ex['task']}_{prompt_id}.jpg"
            image_path = os.path.join(image_dir, image_filename)
            # Store path relative to data_dir for portability
            relative_image_path = os.path.join(SOURCE_NAME, "images", image_filename)
            if ex["image"] is not None and ex["image"].get("bytes"):
                with open(image_path, "wb") as f:
                    f.write(ex["image"]["bytes"])

            prompt_data = {
                "id": prompt_id,
                "lookup_key": lookup_key,
                "prompt_content": [
                    ["image", relative_image_path],
                    ["text", ex["instruction"]],
                ],
                "metadata": {
                    "id": prompt_id,
                    "task": ex["task"],
                    "split": split,
                    "input_caption": ex["input_caption"],
                    "output_caption": ex["output_caption"],
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
            # Use lookup_key for matching
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
        prompt_id: The prompt ID

    Returns:
        Dictionary with prompt_content and metadata, or None if not found
    """
    prompts = load_prompts(data_dir)
    return prompts.get(prompt_id)
