# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
Chameleon source module.

Data source: https://github.com/facebookresearch/chameleon
File: data/prompts_for_human_evaluations.jsonl
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from . import get_source_config

SOURCE_NAME = "chameleon"


def make_lookup_key(prompt_id: str, category: str) -> str:
    """Create lookup key from id and category."""
    return f"{prompt_id}_{category}"


def download_image(url: str, save_path: str, max_retries: int = 3) -> bool:
    """Download an image from URL."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False
    return False


def get_image_filename(url: str, category: str, sample_id: str) -> str:
    """Generate a unique filename for an image."""
    parsed = urlparse(url)
    path = parsed.path
    ext = os.path.splitext(path)[1]
    if not ext or ext.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        ext = ".jpg"

    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # Sanitize category for filename
    safe_category = category.replace(" ", "_").replace("/", "_")
    return f"{safe_category}_{sample_id}_{url_hash}{ext}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download Chameleon data from GitHub.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    config = get_source_config(SOURCE_NAME)
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(source_dir, "chameleon")
    jsonl_path = os.path.join(repo_dir, "data", "prompts_for_human_evaluations.jsonl")

    # Check if already downloaded
    if os.path.exists(jsonl_path) and not force:
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    # Clone the repository
    if not os.path.exists(repo_dir) or force:
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)
        print(f"[{SOURCE_NAME}] Cloning repository...")

        # If a specific commit is needed, we need to do a full clone
        if "commit" in config:
            subprocess.run(
                ["git", "clone", config["url"], repo_dir],
                check=True,
                capture_output=True,
            )
            print(f"[{SOURCE_NAME}] Checking out commit {config['commit'][:7]}...")
            subprocess.run(
                ["git", "checkout", config["commit"]],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )
        else:
            # No specific commit needed, shallow clone is fine
            subprocess.run(
                ["git", "clone", "--depth", "1", config["url"], repo_dir],
                check=True,
                capture_output=True,
            )

    return source_dir


def load_prompts(
    data_dir: str, required_keys: Optional[Set[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from Chameleon.

    Args:
        data_dir: Base directory containing downloaded data
        required_keys: Optional set of lookup keys to load. If None, load all.

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    repo_dir = os.path.join(source_dir, "chameleon")
    jsonl_path = os.path.join(repo_dir, "data", "prompts_for_human_evaluations.jsonl")

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"prompts_for_human_evaluations.jsonl not found at {jsonl_path}"
        )

    # Create images directory for downloaded images
    images_dir = os.path.join(source_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Fallback images directory (relative to data_dir parent, i.e., release_clean/images)
    # data_dir is typically ./sources, so fallback is ../images
    fallback_images_dir = os.path.join(os.path.dirname(data_dir), "images")

    if required_keys:
        print(f"[{SOURCE_NAME}] Filtering to {len(required_keys)} required prompts")

    prompts = {}
    failed_downloads = 0
    copied_from_fallback = 0
    skipped = 0

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc=f"Loading {SOURCE_NAME}"):
            sample = json.loads(line.strip())
            sample_id = sample["id"]
            prompt_text = sample["prompt"]
            task_type = sample["task_type"]
            image_urls = sample["image_urls"]

            lookup_key = make_lookup_key(sample_id, task_type)

            # Skip if not in required keys (when filtering is enabled)
            if required_keys is not None and lookup_key not in required_keys:
                skipped += 1
                continue

            # Process prompt content
            prompt_content = []
            local_image_paths = []
            all_images_ok = True

            if "<img>" in prompt_text and image_urls:
                parts = prompt_text.split("<img>")

                for i, part in enumerate(parts):
                    if part.strip():
                        prompt_content.append(["text", part.strip()])

                    if i < len(parts) - 1 and i < len(image_urls):
                        url = image_urls[i]
                        if url.strip():
                            filename = get_image_filename(url, task_type, sample_id)
                            abs_path = os.path.join(images_dir, filename)
                            rel_path = os.path.join(SOURCE_NAME, "images", filename)

                            # Try: 1) already exists, 2) download from URL, 3) fallback folder
                            if not os.path.exists(abs_path):
                                if not download_image(url, abs_path):
                                    # URL failed, try fallback folder
                                    fallback_path = os.path.join(
                                        fallback_images_dir, filename
                                    )
                                    if os.path.exists(fallback_path):
                                        shutil.copy(fallback_path, abs_path)
                                        copied_from_fallback += 1
                                    else:
                                        failed_downloads += 1
                                        all_images_ok = False
                                        break

                            local_image_paths.append(rel_path)
                            prompt_content.append(["image", rel_path])
            else:
                # Text-only prompt
                prompt_content.append(["text", prompt_text])

            if all_images_ok:
                prompts[lookup_key] = {
                    "id": sample_id,
                    "lookup_key": lookup_key,
                    "prompt_content": prompt_content,
                    "metadata": {
                        "id": sample_id,
                        "category": task_type,
                        "has_images": len(local_image_paths) > 0,
                        "image_count": len(local_image_paths),
                    },
                }

    if copied_from_fallback > 0:
        print(
            f"[{SOURCE_NAME}] Copied {copied_from_fallback} images from fallback folder"
        )
    if failed_downloads > 0:
        print(f"[{SOURCE_NAME}] Warning: {failed_downloads} image downloads failed")
    if skipped > 0:
        print(f"[{SOURCE_NAME}] Skipped {skipped} prompts (not in required set)")

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
