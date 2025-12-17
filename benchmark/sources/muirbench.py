"""
MUIRBench source module.

HuggingFace: MUIRBENCH/MUIRBENCH
Paper: Multi-Image Understanding and Reasoning Benchmark

Data structure:
- Images in dataset with image_list field
- Questions with multiple choice options
"""

import json
import os
from typing import Any, Dict

from datasets import load_dataset
from tqdm import tqdm

SOURCE_NAME = "muirbench"
HF_DATASET = "MUIRBENCH/MUIRBENCH"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download MUIRBench data from HuggingFace.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "prompts.jsonl")
    images_dir = os.path.join(source_dir, "images")

    if (
        os.path.exists(prompts_path)
        and os.path.exists(images_dir)
        and not force
    ):
        print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
        return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from HuggingFace: {HF_DATASET}")

    # Load dataset
    dataset = load_dataset(HF_DATASET, split="test")
    print(f"[{SOURCE_NAME}] Loaded {len(dataset)} samples")

    # Process and save
    prompts = []
    for row in tqdm(dataset, desc=f"Processing {SOURCE_NAME}"):
        idx = row["idx"]
        question = row["question"]
        options = row["options"]
        answer = row["answer"]
        task = row["task"]
        image_relation = row["image_relation"]
        image_type = row["image_type"]

        # Build prompt with choices
        prompt_text = "Question: " + question + "\nChoices:"
        choice_idxs = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        for choice_idx, option in zip(choice_idxs, options):
            prompt_text += f"\n{choice_idx} {option}"

        # Save images and build image paths
        image_paths = []
        for i, img in enumerate(row["image_list"]):
            if img is not None:
                img_filename = f"{idx}_{i}.jpg"
                img_path = os.path.join(images_dir, img_filename)
                if not os.path.exists(img_path):
                    img.convert("RGB").save(img_path)
                image_paths.append(img_filename)

        prompts.append(
            {
                "id": idx,
                "task": task,
                "image_relation": image_relation,
                "image_type": image_type,
                "question": question,
                "options": options,
                "answer": answer,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
            }
        )

    # Save prompts
    with open(prompts_path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    print(f"[{SOURCE_NAME}] Saved {len(prompts)} prompts to {prompts_path}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from MUIRBench.

    The lookup key is the idx (id) field.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    prompts_path = os.path.join(source_dir, "prompts.jsonl")

    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"prompts.jsonl not found at {prompts_path}")

    # Load prompts
    samples = []
    with open(prompts_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    prompts = {}

    for sample in tqdm(samples, desc=f"Loading {SOURCE_NAME}"):
        sample_id = str(sample["id"])
        prompt_text = sample["prompt_text"]
        image_paths = sample["image_paths"]

        # Build prompt content by parsing the prompt text and inserting images
        # The original format has <image> placeholders
        # We need to reconstruct the interleaved format
        prompt_content = []

        # Split prompt text at image boundaries
        # Images appear at <image> positions in the original question
        segments = prompt_text.split("<image>")

        # If no <image> tokens, images come before text
        if len(segments) == 1:
            # Add images first
            for img_filename in image_paths:
                rel_path = os.path.join(SOURCE_NAME, "images", img_filename)
                prompt_content.append(["image", rel_path])
            # Then add text
            prompt_content.append(["text", prompt_text])
        else:
            # Interleave text and images
            img_idx = 0
            for i, segment in enumerate(segments):
                if segment.strip():
                    prompt_content.append(["text", segment])
                if i < len(segments) - 1 and img_idx < len(image_paths):
                    rel_path = os.path.join(
                        SOURCE_NAME, "images", image_paths[img_idx]
                    )
                    prompt_content.append(["image", rel_path])
                    img_idx += 1

        prompts[sample_id] = {
            "id": sample["id"],
            "lookup_key": sample_id,
            "prompt_content": prompt_content,
            "metadata": {
                "id": sample["id"],
                "task": sample["task"],
                "image_relation": sample["image_relation"],
                "image_type": sample["image_type"],
                "answer": sample["answer"],
            },
        }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts
