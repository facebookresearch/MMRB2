"""
HQ-Edit source module.

HQ-Edit is a high-quality image editing dataset.
HuggingFace: UCSC-VLAA/HQEdit-test

Data structure:
- Images: HuggingFace dataset images/*.png
- Prompts: HuggingFace dataset prompts.jsonl
"""

import json
import os
import shutil
from typing import Any, Dict

from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

SOURCE_NAME = "hq-edit"
HF_REPO = "UCSC-VLAA/HQEdit-test"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download HQ-Edit data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "prompts.jsonl")
    images_dir = os.path.join(source_dir, "hf_images")

    if (
        os.path.exists(prompts_path)
        and os.path.exists(images_dir)
        and not force
    ):
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download prompts.jsonl
    prompts_hf = hf_hub_download(HF_REPO, "prompts.jsonl", repo_type="dataset")
    shutil.copy(prompts_hf, prompts_path)
    print(f"[{SOURCE_NAME}] Downloaded prompts.jsonl")

    # Download all images
    files = list_repo_files(HF_REPO, repo_type="dataset")
    image_files = [f for f in files if f.startswith("images/")]

    for img_file in tqdm(image_files, desc=f"Downloading {SOURCE_NAME} images"):
        img_path = hf_hub_download(HF_REPO, img_file, repo_type="dataset")
        # Copy to our directory with just the filename
        img_name = os.path.basename(img_file)
        dst_path = os.path.join(images_dir, img_name)
        if not os.path.exists(dst_path):
            shutil.copy(img_path, dst_path)

    print(f"[{SOURCE_NAME}] Downloaded {len(image_files)} images")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from HQ-Edit.

    The ID is the index in the dataset (0, 1, 2, ...).

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "prompts.jsonl")
    hf_images_dir = os.path.join(source_dir, "hf_images")
    output_images_dir = os.path.join(source_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    # Load prompts
    prompt_list = []
    with open(prompts_path, "r") as f:
        for line in f:
            prompt_list.append(json.loads(line))

    prompts = {}

    for i, prompt_data in tqdm(
        enumerate(prompt_list), desc=f"Loading {SOURCE_NAME}"
    ):
        prompt_id = i
        edit_text = prompt_data["edit"]
        input_caption = prompt_data["input"]
        output_caption = prompt_data["output"]

        # Source image path (format: 000.png, 001.png, etc.)
        img_filename = f"{i:03d}.png"
        src_image_path = os.path.join(hf_images_dir, img_filename)

        if not os.path.exists(src_image_path):
            continue

        # Copy to output directory with simpler name
        out_filename = f"{i}.png"
        dst_image_path = os.path.join(output_images_dir, out_filename)
        rel_img_path = os.path.join(SOURCE_NAME, "images", out_filename)

        if not os.path.exists(dst_image_path):
            shutil.copy(src_image_path, dst_image_path)

        # Use ID as lookup key
        lookup_key = str(prompt_id)

        prompts[lookup_key] = {
            "id": prompt_id,
            "lookup_key": lookup_key,
            "prompt_content": [
                ["image", rel_img_path],
                ["text", edit_text],
            ],
            "metadata": {
                "id": prompt_id,
                "input_caption": input_caption,
                "output_caption": output_caption,
            },
        }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
