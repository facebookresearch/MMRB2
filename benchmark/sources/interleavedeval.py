# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
InterleavedEval source module.

HuggingFace: mqliu/InterleavedBench
Paper: https://arxiv.org/abs/2406.14643

Data structure:
- JSON: interleaved_bench.json
- Images: zipped_images/ directory
"""

import json
import os
import shutil
import zipfile
from typing import Any, Dict

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

SOURCE_NAME = "interleavedeval"
HF_REPO = "mqliu/InterleavedBench"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download InterleavedEval data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    json_path = os.path.join(source_dir, "interleaved_bench.json")
    images_dir = os.path.join(source_dir, "zipped_images")

    if os.path.exists(json_path) and os.path.exists(images_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download interleaved_bench.json
    # Note: This is a model repo, not a dataset repo
    json_hf = hf_hub_download(HF_REPO, "interleaved_bench.json", repo_type="model")
    shutil.copy(json_hf, json_path)
    print(f"[{SOURCE_NAME}] Downloaded interleaved_bench.json")

    # Download zipped_images directory
    # List all files in zipped_images
    files = list_repo_files(HF_REPO, repo_type="model")
    image_files = [f for f in files if f.startswith("zipped_images/")]

    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading {len(image_files)} image files...")
    for img_file in tqdm(image_files, desc=f"Downloading {SOURCE_NAME} images"):
        # Download each file
        img_hf = hf_hub_download(HF_REPO, img_file, repo_type="model")

        # Create subdirectory structure and copy
        rel_path = img_file[len("zipped_images/") :]  # Remove prefix
        dst_path = os.path.join(images_dir, rel_path)
        dst_dir = os.path.dirname(dst_path)
        if dst_dir:
            os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(dst_path):
            shutil.copy(img_hf, dst_path)

    # Extract any zip files
    print(f"[{SOURCE_NAME}] Extracting zip files...")
    for filename in os.listdir(images_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(images_dir, filename)
            # Extract to images_dir (creates subdirectory with zip name minus .zip)
            extract_dir = os.path.join(images_dir, filename[:-4])
            if not os.path.exists(extract_dir):
                print(f"[{SOURCE_NAME}] Extracting {filename}...")
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(images_dir)

    print(f"[{SOURCE_NAME}] Downloaded images to {images_dir}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from InterleavedEval.

    The lookup key is just the index (0, 1, 2, ...) as that's what's stored
    in the balanced dataset metadata.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    json_path = os.path.join(source_dir, "interleaved_bench.json")
    images_base = os.path.join(source_dir, "zipped_images")
    output_images_dir = os.path.join(source_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"interleaved_bench.json not found at {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = {}
    IMG_TOKEN = "<image>"

    for idx, sample in enumerate(tqdm(data, desc=f"Loading {SOURCE_NAME}")):
        # The lookup key is the index (as stored in balanced dataset)
        lookup_key = str(idx)

        conversation = sample["conversations"]
        img_list = sample["image"]
        prompt = conversation[0]
        prompt_value = prompt["value"]

        # Process prompt content
        prompt_content = []
        parts = prompt_value.split(IMG_TOKEN)

        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            if part.startswith("."):
                part = part[1:]
            part = part.strip()
            if part:
                prompt_content.append(["text", part])
            if not is_last and i < len(img_list):
                img = img_list[i]
                src_path = os.path.join(images_base, img)
                if os.path.exists(src_path):
                    # Create subdirectory structure
                    img_subdir = os.path.dirname(img)
                    if img_subdir:
                        os.makedirs(
                            os.path.join(output_images_dir, img_subdir),
                            exist_ok=True,
                        )
                    dst_path = os.path.join(output_images_dir, img)
                    if not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)
                    rel_path = os.path.join(SOURCE_NAME, "images", img)
                    prompt_content.append(["image", rel_path])
                else:
                    # Image not found, skip this prompt
                    break

        if prompt_content:
            prompts[lookup_key] = {
                "id": idx,
                "lookup_key": lookup_key,
                "prompt_content": prompt_content,
                "metadata": {
                    "id": idx,
                    "original_id": sample["id"],
                    "task_name": sample["task_name"],
                    "goal": sample.get("goal", ""),
                    "category": sample.get("category", []),
                    "dataset_id": sample.get("dataset_id", ""),
                },
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
