# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
MMMG source module.

Data source: https://huggingface.co/datasets/UW-FMRL2/MMMG
"""

import os
import re
import shutil
from typing import Any, Dict

from datasets import load_dataset
from tqdm import tqdm

SOURCE_NAME = "mmmg"
HF_DATASET = "UW-FMRL2/MMMG"


def make_lookup_key(prompt_id: int, category: str) -> str:
    """Create lookup key from id and category."""
    return f"{prompt_id}_{category}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download MMMG data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    cache_dir = os.path.join(source_dir, "cache")

    # Check if already downloaded
    if os.path.exists(cache_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_DATASET}")

    # Download dataset (this caches it)
    load_dataset(HF_DATASET, cache_dir=cache_dir, split="test")

    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from MMMG.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    cache_dir = os.path.join(source_dir, "cache")
    images_dir = os.path.join(source_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Fallback images directory (relative to data_dir parent, i.e., release/images)
    fallback_images_dir = os.path.join(os.path.dirname(data_dir), "images")
    use_fallback = os.path.exists(fallback_images_dir)
    if use_fallback:
        print(f"[{SOURCE_NAME}] Fallback images available at {fallback_images_dir}")

    print(f"[{SOURCE_NAME}] Loading dataset from HuggingFace...")

    # Load dataset
    if os.path.exists(cache_dir):
        dataset = load_dataset(HF_DATASET, cache_dir=cache_dir, split="test")
    else:
        dataset = load_dataset(HF_DATASET, split="test")

    prompts = {}
    pattern = re.compile(r"<(image_\d+)>")

    for idx, row in enumerate(tqdm(dataset, desc=f"Loading {SOURCE_NAME}")):
        category = row["category"]

        # Skip if not image-related (same filter as original processing)
        if category[0] != "i":
            continue

        lookup_key = make_lookup_key(idx, category)
        text = row["instruction"]

        # Parse text and images
        prompt_content = []
        last_idx = 0

        for match in pattern.finditer(text):
            # Text before the <image_X>
            if match.start() > last_idx:
                part = text[last_idx : match.start()].strip()
                if part:
                    prompt_content.append(["text", part])

            # The <image_X> part
            img_key = match.group(1)  # e.g. "image_0"
            if img_key in row and row[img_key] is not None:
                image_filename = f"{idx}_{img_key}.jpg"
                abs_path = os.path.join(images_dir, image_filename)
                rel_path = os.path.join(SOURCE_NAME, "images", image_filename)

                # Get image from fallback or save from dataset
                if not os.path.exists(abs_path):
                    fallback_path = os.path.join(fallback_images_dir, image_filename)
                    if use_fallback and os.path.exists(fallback_path):
                        shutil.copy(fallback_path, abs_path)
                    else:
                        # Save from dataset
                        row[img_key].convert("RGB").save(abs_path)

                prompt_content.append(["image", rel_path])

            last_idx = match.end()

        # Remaining text after the last <image_X>
        if last_idx < len(text):
            part = text[last_idx:].strip()
            if part:
                # Clean up "ONLY" text (same as original processing)
                part = (
                    part.replace(" ONLY ", " ")
                    .replace(" only ", " ")
                    .replace(" Only ", " ")
                )
                prompt_content.append(["text", part])

        if prompt_content:
            prompts[lookup_key] = {
                "id": idx,
                "lookup_key": lookup_key,
                "prompt_content": prompt_content,
                "metadata": {
                    "id": idx,
                    "category": category,
                },
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
