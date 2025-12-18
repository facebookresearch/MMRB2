# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
ISG-Bench source module.

Data source: https://github.com/Dongping-Chen/ISG
HuggingFace: https://huggingface.co/datasets/shuaishuaicdp/ISG-Bench
"""

import json
import os
import shutil
import subprocess
from typing import Any, Dict

from tqdm import tqdm

from . import get_source_config

SOURCE_NAME = "isgbench"


def make_lookup_key(prompt_id: str, category: str) -> str:
    """Create lookup key from id and category."""
    return f"{prompt_id}_{category}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download ISG-Bench data from GitHub and HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    config = get_source_config(SOURCE_NAME)
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(source_dir, "ISG")
    images_dir = os.path.join(repo_dir, "ISG_eval", "images")

    # Check if already downloaded
    if os.path.exists(images_dir) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    # Clone the repository
    if not os.path.exists(repo_dir) or force:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        print(f"[{SOURCE_NAME}] Cloning repository...")
        subprocess.run(
            ["git", "clone", config["url"], repo_dir],
            check=True,
            capture_output=True,
        )

        # Checkout specific commit for reproducibility
        if "commit" in config:
            print(f"[{SOURCE_NAME}] Checking out commit {config['commit'][:8]}...")
            subprocess.run(
                ["git", "checkout", config["commit"]],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )

    # Download images.zip from HuggingFace and extract
    if not os.path.exists(images_dir) or force:
        print(f"[{SOURCE_NAME}] Downloading images.zip from HuggingFace...")
        try:
            import zipfile

            from huggingface_hub import hf_hub_download

            zip_path = hf_hub_download(
                repo_id="shuaishuaicdp/ISG-Bench",
                filename="images.zip",
                repo_type="dataset",
                local_dir=os.path.join(repo_dir, "ISG_eval"),
            )

            # Extract the zip file
            print(f"[{SOURCE_NAME}] Extracting images...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(repo_dir, "ISG_eval"))

            print(f"[{SOURCE_NAME}] Images extracted to: {images_dir}")

        except ImportError:
            print(
                "  Warning: huggingface_hub not installed. Please run: pip install huggingface_hub"
            )
            raise

    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from ISG-Bench.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(source_dir, "ISG")
    jsonl_path = os.path.join(repo_dir, "ISG_eval", "ISG-Bench.jsonl")
    images_base = os.path.join(repo_dir, "ISG_eval")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"ISG-Bench.jsonl not found at {jsonl_path}")

    # Create output images directory for portable paths
    output_images_dir = os.path.join(source_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    prompts = {}

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=f"Loading {SOURCE_NAME}"):
            ex = json.loads(line)
            prompt_id = ex["id"]
            category = ex["Category"]
            lookup_key = make_lookup_key(prompt_id, category)

            # Process Query into prompt_content
            prompt_content = []
            for item in ex["Query"]:
                if item["type"] == "text":
                    prompt_content.append(["text", item["content"]])
                elif item["type"] == "image":
                    # Copy image to output directory and use relative path
                    src_path = os.path.join(images_base, item["content"])
                    image_filename = os.path.basename(item["content"])

                    # Handle extension mismatch (case-insensitive matching)
                    # JSONL may have .JPG/.PNG but files are .jpg/.png
                    if not os.path.exists(src_path):
                        base_name = os.path.splitext(src_path)[0]
                        base_filename = os.path.splitext(image_filename)[0]
                        # Try common image extensions in lowercase
                        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                            alt_path = base_name + ext
                            if os.path.exists(alt_path):
                                src_path = alt_path
                                image_filename = base_filename + ext
                                break

                    dst_path = os.path.join(output_images_dir, image_filename)

                    if os.path.exists(src_path) and not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)

                    # Store relative path for portability
                    rel_path = os.path.join(SOURCE_NAME, "images", image_filename)
                    prompt_content.append(["image", rel_path])

            # Process Golden answer for metadata
            gold_answer = []
            for item in ex["Golden"]:
                if item["type"] == "text":
                    gold_answer.append(["text", item["content"]])
                elif item["type"] == "image":
                    # Use relative path for gold images too
                    image_filename = os.path.basename(item["content"])
                    src_path = os.path.join(images_base, item["content"])

                    # Handle extension mismatch (case-insensitive matching)
                    # JSONL may have .JPG/.PNG but files are .jpg/.png
                    if not os.path.exists(src_path):
                        base_name = os.path.splitext(src_path)[0]
                        base_filename = os.path.splitext(image_filename)[0]
                        # Try common image extensions in lowercase
                        for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                            alt_path = base_name + ext
                            if os.path.exists(alt_path):
                                src_path = alt_path
                                image_filename = base_filename + ext
                                break

                    dst_path = os.path.join(output_images_dir, image_filename)

                    if os.path.exists(src_path) and not os.path.exists(dst_path):
                        shutil.copy(src_path, dst_path)

                    rel_path = os.path.join(SOURCE_NAME, "images", image_filename)
                    gold_answer.append(["image", rel_path])

            prompts[lookup_key] = {
                "id": prompt_id,
                "lookup_key": lookup_key,
                "prompt_content": prompt_content,
                "metadata": {
                    "id": prompt_id,
                    "category": category,
                    "gold_answer": gold_answer,
                },
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
