"""
Text Heavy Edit source module.

This is a synthetic dataset for text-heavy image editing tasks.
The prompts and images are from local processed data.

Data structure:
- JSONL: .../test-heavy-edit/processed_text_heavy_edit_200.jsonl
- Images: .../multi-image/images/ and .../multi-image/new_images/
"""

import json
import os
import shutil
from typing import Any, Dict

from tqdm import tqdm

SOURCE_NAME = "text_heavy_edit"

# Note: This source requires local processed data that is not publicly available.
# The JSONL path should be set via environment variable or data will be skipped.
LOCAL_JSONL = os.environ.get(
    "TEXT_HEAVY_EDIT_JSONL",
    ""  # Empty default - will skip if not set
)


def download(data_dir: str, force: bool = False) -> str:
    """
    Download text_heavy_edit data.

    Uses local processed data.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)

    if os.path.exists(source_dir) and not force:
        print(f"[{SOURCE_NAME}] Already set up: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    if LOCAL_JSONL and os.path.exists(LOCAL_JSONL):
        print(f"[{SOURCE_NAME}] Using local processed data")
    else:
        print(f"[{SOURCE_NAME}] Note: Local data not configured (set TEXT_HEAVY_EDIT_JSONL env var)")

    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from text_heavy_edit.

    The ID is unique per prompt.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    images_dir = os.path.join(source_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    if not LOCAL_JSONL or not os.path.exists(LOCAL_JSONL):
        print(f"[{SOURCE_NAME}] Skipping: JSONL not configured (set TEXT_HEAVY_EDIT_JSONL env var)")
        return {}

    prompts = {}

    with open(LOCAL_JSONL, "r") as f:
        for line in tqdm(f, desc=f"Loading {SOURCE_NAME}"):
            sample = json.loads(line)
            metadata = sample["metadata"]
            prompt_id = metadata["id"]
            category = metadata.get("category", "")
            prompt_data = sample["prompt"]

            # Extract image path and text from prompt
            image_path = None
            text = None
            for item in prompt_data:
                if item[0] == "image":
                    image_path = item[1]
                elif item[0] == "text":
                    text = item[1]

            if not image_path or not text:
                continue

            if not os.path.exists(image_path):
                continue

            # Copy image to output directory
            img_filename = f"{prompt_id}_{os.path.basename(image_path)}"
            dst_image_path = os.path.join(images_dir, img_filename)
            rel_img_path = os.path.join(SOURCE_NAME, "images", img_filename)

            if not os.path.exists(dst_image_path):
                shutil.copy(image_path, dst_image_path)

            # Use ID as lookup key
            lookup_key = str(prompt_id)

            prompts[lookup_key] = {
                "id": prompt_id,
                "lookup_key": lookup_key,
                "prompt_content": [
                    ["image", rel_img_path],
                    ["text", text],
                ],
                "metadata": {
                    "id": prompt_id,
                    "category": category,
                },
            }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
