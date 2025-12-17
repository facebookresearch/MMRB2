"""
MindCube source module.

HuggingFace: MLL-Lab/MindCube
Paper: Spatial Mental Modeling from Limited Views
GitHub: https://github.com/mll-lab-nu/MindCube

Data structure in zip:
- data/raw/MindCube_tinybench.jsonl - questions and answers
- data/other_all_image/{among,around,rotation,translation}/ - images
"""

import json
import os
import shutil
import zipfile
from typing import Any, Dict

from huggingface_hub import hf_hub_download
from tqdm import tqdm

SOURCE_NAME = "mindcube"
HF_REPO = "MLL-Lab/MindCube"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download MindCube data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "MindCube_tinybench.jsonl")
    images_dir = os.path.join(source_dir, "other_all_image")

    if (
        os.path.exists(prompts_path)
        and os.path.exists(images_dir)
        and not force
    ):
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_REPO}")

    # Download data.zip
    zip_path = hf_hub_download(HF_REPO, "data.zip", repo_type="dataset")
    print(f"[{SOURCE_NAME}] Downloaded data.zip")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        # Extract MindCube_tinybench.jsonl
        z.extract("data/raw/MindCube_tinybench.jsonl", source_dir)
        shutil.move(
            os.path.join(source_dir, "data/raw/MindCube_tinybench.jsonl"),
            prompts_path,
        )

        # Extract images
        image_members = [
            m
            for m in z.namelist()
            if m.startswith("data/other_all_image/") and not m.endswith("/")
        ]
        print(f"[{SOURCE_NAME}] Extracting {len(image_members)} image files...")
        for member in tqdm(image_members, desc=f"Extracting {SOURCE_NAME}"):
            # Extract to source_dir and adjust path
            z.extract(member, source_dir)

        # Move from data/other_all_image to other_all_image
        src_img_dir = os.path.join(source_dir, "data/other_all_image")
        if os.path.exists(src_img_dir):
            if os.path.exists(images_dir):
                shutil.rmtree(images_dir)
            shutil.move(src_img_dir, images_dir)

        # Clean up empty data directory
        data_dir_path = os.path.join(source_dir, "data")
        if os.path.exists(data_dir_path):
            shutil.rmtree(data_dir_path)

    print(f"[{SOURCE_NAME}] Extracted to {source_dir}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from MindCube.

    The lookup key is the ID from the JSONL file.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "MindCube_tinybench.jsonl")
    images_base = os.path.join(source_dir, "other_all_image")
    output_images_dir = os.path.join(source_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(
            f"MindCube_tinybench.jsonl not found at {prompts_path}"
        )

    # Load all samples
    samples = []
    with open(prompts_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    prompts = {}

    for sample in tqdm(samples, desc=f"Loading {SOURCE_NAME}"):
        sample_id = sample["id"]
        question = sample["question"]
        image_paths = sample["images"]
        gt_answer = sample.get("gt_answer", "")

        # Build prompt content with images and question
        prompt_content = []
        valid = True

        for img_rel_path in image_paths:
            # img_rel_path like: other_all_image/among/xxx/front_126.png
            # We need to extract just the part after other_all_image/
            if img_rel_path.startswith("other_all_image/"):
                img_subpath = img_rel_path[len("other_all_image/") :]
            else:
                img_subpath = img_rel_path

            src_path = os.path.join(images_base, img_subpath)

            if not os.path.exists(src_path):
                valid = False
                break

            # Copy image to output directory with flattened name
            img_filename = img_subpath.replace("/", "_")
            dst_path = os.path.join(output_images_dir, img_filename)
            rel_path = os.path.join(SOURCE_NAME, "images", img_filename)

            if not os.path.exists(dst_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

            prompt_content.append(["image", rel_path])

        if not valid:
            continue

        # Add the question text
        prompt_content.append(["text", question])

        prompts[sample_id] = {
            "id": sample_id,
            "lookup_key": sample_id,
            "prompt_content": prompt_content,
            "metadata": {
                "id": sample_id,
                "category": sample.get("category", []),
                "type": sample.get("type", ""),
                "gt_answer": gt_answer,
            },
        }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
