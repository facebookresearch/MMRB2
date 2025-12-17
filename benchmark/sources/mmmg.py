# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""
MMMG source module.

Data source: https://huggingface.co/datasets/UW-FMRL2/MMMG
"""

import gc
import glob
import os
import re
import shutil
from io import BytesIO
from typing import Any, Dict

import pyarrow as pa
from datasets import load_dataset
from PIL import Image

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
    Load prompts from MMMG using PyArrow with memory mapping for efficiency.

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

    # Find the arrow file in cache
    arrow_pattern = os.path.join(cache_dir, "**", "*.arrow")
    arrow_files = glob.glob(arrow_pattern, recursive=True)

    if not arrow_files:
        raise FileNotFoundError(
            f"No arrow file found in {cache_dir}. Run download() first."
        )

    arrow_file = arrow_files[0]
    print(f"[{SOURCE_NAME}] Loading from arrow file: {arrow_file}")

    prompts = {}
    pattern = re.compile(r"<(image_\d+)>")

    # Open as stream (HuggingFace uses Arrow IPC stream format)
    with pa.memory_map(arrow_file, "r") as mmap:
        reader = pa.ipc.open_stream(mmap)
        row_idx = 0
        batch_idx = 0

        # Read batches one at a time from the stream
        for batch in reader:
            batch_len = batch.num_rows

            # Convert batch to pandas for easier access (one batch at a time)
            df = batch.to_pandas()

            for local_idx in range(batch_len):
                idx = row_idx + local_idx
                category = df.iloc[local_idx]["category"]

                # Skip if not image-related
                if category[0] != "i":
                    continue

                lookup_key = make_lookup_key(idx, category)
                text = df.iloc[local_idx]["instruction"]

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
                    image_filename = f"{idx}_{img_key}.jpg"
                    abs_path = os.path.join(images_dir, image_filename)
                    rel_path = os.path.join(SOURCE_NAME, "images", image_filename)
                    fallback_path = os.path.join(fallback_images_dir, image_filename)

                    # Check if image already exists on disk
                    if os.path.exists(abs_path):
                        prompt_content.append(["image", rel_path])
                    elif use_fallback and os.path.exists(fallback_path):
                        shutil.copy(fallback_path, abs_path)
                        prompt_content.append(["image", rel_path])
                    else:
                        # Get image bytes from dataframe
                        try:
                            img_data = df.iloc[local_idx].get(img_key)
                            if img_data is not None:
                                # Handle different formats
                                if (
                                    isinstance(img_data, dict)
                                    and "bytes" in img_data
                                    and img_data["bytes"]
                                ):
                                    img = Image.open(BytesIO(img_data["bytes"]))
                                elif isinstance(img_data, bytes):
                                    img = Image.open(BytesIO(img_data))
                                elif hasattr(img_data, "convert"):
                                    img = img_data
                                else:
                                    img = None

                                if img is not None:
                                    img_rgb = img.convert("RGB")
                                    img_rgb.save(abs_path)
                                    img_rgb.close()
                                    if hasattr(img, "close"):
                                        img.close()
                                    del img_rgb, img
                                    prompt_content.append(["image", rel_path])
                        except Exception as e:
                            print(
                                f"[{SOURCE_NAME}] Warning: Error processing {img_key} at idx {idx}: {e}"
                            )

                    last_idx = match.end()

                # Remaining text after the last <image_X>
                if last_idx < len(text):
                    part = text[last_idx:].strip()
                    if part:
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

            # Update row index and cleanup after each batch
            row_idx += batch_len
            del df, batch
            gc.collect()
            batch_idx += 1
            print(f"[{SOURCE_NAME}] Processed batch {batch_idx}")

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
