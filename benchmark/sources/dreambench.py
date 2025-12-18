# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
DreamBench source module.

HuggingFace: yuangpeng/dreambench_plus
Paper: DreamBench++ for subject-driven image editing

Data structure in zip:
- data/images/{category}/{subcategory}/{id}.jpg  (150 images)
- data/captions/{category}/{subcategory}/{id}.txt  (first line=subject, rest=9 prompts)

Categories: live_subject/animal, live_subject/human, object, style
"""

import json
import os
import shutil
import zipfile
from typing import Any, Dict, Optional, Set

from huggingface_hub import hf_hub_download
from tqdm import tqdm

SOURCE_NAME = "dreambench"
HF_REPO = "yuangpeng/dreambench_plus"


def make_lookup_key(prompt_id: int, category: str) -> str:
    """Create lookup key from id and category."""
    return f"{prompt_id}_{category}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download DreamBench data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    images_dir = os.path.join(source_dir, "images")
    captions_dir = os.path.join(source_dir, "captions")

    if os.path.exists(images_dir) and os.path.exists(captions_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download data.zip
    zip_path = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset")
    print(f"[{SOURCE_NAME}] Downloaded data.zip")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract only data/images and data/captions
        for member in tqdm(z.namelist(), desc=f"Extracting {SOURCE_NAME}"):
            if member.startswith("data/images/") or member.startswith("data/captions/"):
                # Remove 'data/' prefix when extracting
                target = os.path.join(source_dir, member[5:])  # Skip 'data/'
                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with z.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

    print(f"[{SOURCE_NAME}] Extracted to {source_dir}")
    return source_dir


def load_prompts(
    data_dir: str, required_keys: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from DreamBench.

    This generates all possible prompts from the dataset.
    Use required_keys to filter to only needed prompts.

    Args:
        data_dir: Base directory containing downloaded data
        required_keys: Optional set of lookup keys to load (for filtering)

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    images_dir = os.path.join(source_dir, "images")
    captions_dir = os.path.join(source_dir, "captions")
    output_images_dir = os.path.join(source_dir, "output_images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(images_dir) or not os.path.exists(captions_dir):
        raise FileNotFoundError(
            f"DreamBench data not found at {source_dir}. Run download first."
        )

    # Build ID mapping from required_keys if provided
    # Keys are in format: {prompt_id}_{category}
    # We need to map this to HF structure
    id_mapping = {}
    if required_keys:
        for key in required_keys:
            # Key format: {id}_{category}
            parts = key.split("_", 1)
            if len(parts) == 2:
                prompt_id, category = parts
                id_mapping[key] = {"prompt_id": int(prompt_id), "category": category}

    prompts = {}
    prompt_counter = 0

    # Walk through all categories
    for category_path in [
        "live_subject/animal",
        "live_subject/human",
        "object",
        "style",
    ]:
        category_img_dir = os.path.join(images_dir, category_path)
        category_cap_dir = os.path.join(captions_dir, category_path)

        if not os.path.exists(category_img_dir):
            continue

        for img_file in os.listdir(category_img_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_id = os.path.splitext(img_file)[0]  # e.g., "00"
            img_path = os.path.join(category_img_dir, img_file)
            caption_path = os.path.join(category_cap_dir, f"{img_id}.txt")

            if not os.path.exists(caption_path):
                continue

            # Read captions (first line is subject, then prompts)
            with open(caption_path, "r") as f:
                lines = f.read().strip().split("\n")

            if len(lines) < 2:
                continue

            subject = lines[0]
            prompt_texts = lines[1:]

            # Copy image to output directory with unique name
            output_img_name = f"{category_path.replace('/', '_')}_{img_file}"
            output_img_path = os.path.join(output_images_dir, output_img_name)
            rel_img_path = os.path.join(SOURCE_NAME, "output_images", output_img_name)

            if not os.path.exists(output_img_path):
                shutil.copy(img_path, output_img_path)

            # Generate prompts for each prompt line
            for prompt_idx, prompt_text in enumerate(prompt_texts):
                # Generate a unique prompt ID based on category and indices
                category_flat = category_path.replace("/", "_")
                category_key = f"{category_flat}_{img_id}"

                # Try to find matching required key
                prompt_id = prompt_counter
                lookup_key = make_lookup_key(prompt_id, category_key)

                # If filtering by required_keys, check if this matches
                if required_keys is not None:
                    # Check if any required key matches this image
                    found = False
                    for rkey in required_keys:
                        # Match by category pattern
                        if category_key in rkey or rkey.startswith(f"{prompt_id}_"):
                            found = True
                            lookup_key = rkey
                            break
                    if not found:
                        prompt_counter += 1
                        continue

                prompts[lookup_key] = {
                    "id": prompt_id,
                    "lookup_key": lookup_key,
                    "prompt_content": [
                        ["image", rel_img_path],
                        ["text", prompt_text],
                    ],
                    "metadata": {
                        "id": prompt_id,
                        "category": category_key,
                        "category_type": category_path.split("/")[0],
                        "subject": subject,
                        "hf_category": category_path,
                        "hf_id": img_id,
                        "prompt_idx": prompt_idx,
                    },
                }
                prompt_counter += 1

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
