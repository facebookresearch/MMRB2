#!/usr/bin/env python3
"""
Build MMRB2 benchmark locally from HuggingFace datasets.

This script downloads the benchmark data from HuggingFace and converts it
back to the local format (JSON files + images folders).

Source repo: https://huggingface.co/datasets/rl-research/multimodal-rewardbench-2
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from datasets import load_dataset
import re
from tqdm import tqdm


# Subset configurations
SUBSETS = ["t2i", "edit", "interleaved", "reasoning"]


def extract_images_and_text(
    text: str,
    images: List[Image.Image],
    image_prefix: str,
    output_dir: Path,
    pair_id: str,
    suffix: str,
) -> List[List[str]]:
    """
    Convert HuggingFace format (text with placeholders + images) back to
    the benchmark content format (list of ["type", "content"] pairs).

    Args:
        text: Text with <image_N> placeholders
        images: List of PIL Images
        image_prefix: Prefix for image paths ("images" or "input_images")
        output_dir: Output directory
        pair_id: Pair ID for naming images
        suffix: Suffix for image filenames (e.g., "prompt", "a", "b")

    Returns:
        List of ["type", "content"] pairs
    """
    content = []

    # Handle empty text case
    if not text and not images:
        return content

    if not text:
        # Only images, no text
        for i, img in enumerate(images):
            img_filename = f"{pair_id}_{suffix}_{i}.jpg"
            img_path = output_dir / image_prefix / img_filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path, "JPEG", quality=95)
            content.append(["image", f"{image_prefix}/{img_filename}"])
        return content

    # Find all image placeholders
    pattern = r"<image_(\d+)>"

    # Split text by image placeholders
    parts = re.split(pattern, text)

    image_idx = 0
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text part
            if part.strip():
                content.append(["text", part.strip()])
        else:
            # Image index
            if image_idx < len(images):
                img = images[image_idx]
                img_filename = f"{pair_id}_{suffix}_{image_idx}.jpg"
                img_path = output_dir / image_prefix / img_filename
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path, "JPEG", quality=95)
                content.append(["image", f"{image_prefix}/{img_filename}"])
                image_idx += 1

    # If no placeholders found but we have images, append them at the end
    if not re.search(pattern, text) and images:
        for i, img in enumerate(images):
            img_filename = f"{pair_id}_{suffix}_{i}.jpg"
            img_path = output_dir / image_prefix / img_filename
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(img_path, "JPEG", quality=95)
            content.append(["image", f"{image_prefix}/{img_filename}"])

    return content


def convert_hf_to_benchmark_format(
    row: Dict,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Convert a HuggingFace dataset row back to benchmark format.

    Args:
        row: A row from the HuggingFace dataset
        output_dir: Output directory for saving images

    Returns:
        Dictionary in benchmark format
    """
    pair_id = row["pair_id"]

    # Convert prompt
    prompt_content = extract_images_and_text(
        row["prompt_text"],
        row["prompt_images"] or [],
        "input_images",
        output_dir,
        pair_id,
        "prompt",
    )

    # Convert response_a
    response_a_content = extract_images_and_text(
        row["response_a_text"],
        row["response_a_images"] or [],
        "images",
        output_dir,
        pair_id,
        "a",
    )

    # Convert response_b
    response_b_content = extract_images_and_text(
        row["response_b_text"],
        row["response_b_images"] or [],
        "images",
        output_dir,
        pair_id,
        "b",
    )

    # Normalize chosen
    chosen = row["chosen"].upper() if row["chosen"] else ""

    # Parse human_annotations from JSON string
    human_annotations_str = row.get("human_annotations", "")
    if human_annotations_str:
        try:
            human_annotations = json.loads(human_annotations_str)
        except json.JSONDecodeError:
            human_annotations = None
    else:
        human_annotations = None

    # Parse prompt_metadata from JSON string
    prompt_metadata_str = row.get("prompt_metadata", "")
    if prompt_metadata_str:
        try:
            prompt_metadata = json.loads(prompt_metadata_str)
        except json.JSONDecodeError:
            prompt_metadata = {}
    else:
        prompt_metadata = {}

    result = {
        "id": pair_id,
        "prompt_content": prompt_content,
        "prompt_source": row.get("prompt_source", ""),
        "prompt_metadata": prompt_metadata,
        "response_a": {
            "model_name": row.get("response_a_model", ""),
            "response_content": response_a_content,
        },
        "response_b": {
            "model_name": row.get("response_b_model", ""),
            "response_content": response_b_content,
        },
        "chosen": chosen,
    }

    # Add human_annotations if present
    if human_annotations is not None:
        result["human_annotations"] = human_annotations

    return result


def download_and_build_subset(
    subset: str,
    repo_id: str,
    output_dir: Path,
) -> None:
    """
    Download a subset from HuggingFace and convert to benchmark format.

    Args:
        subset: Subset name (t2i, edit, interleaved, reasoning)
        repo_id: HuggingFace repository ID
        output_dir: Output directory
    """
    print(f"\n{'='*60}")
    print(f"Downloading subset: {subset}")
    print(f"{'='*60}")

    # Load dataset from HuggingFace
    print(f"Loading {subset} from {repo_id}...")
    dataset = load_dataset(repo_id, subset, split="test")

    print(f"Converting {len(dataset)} pairs to benchmark format...")

    pairs = []
    for row in tqdm(dataset, desc=f"Processing {subset}"):
        try:
            pair = convert_hf_to_benchmark_format(row, output_dir)
            pairs.append(pair)
        except Exception as e:
            print(f"Error processing {row.get('pair_id', 'unknown')}: {e}")
            continue

    # Save JSON file
    json_path = output_dir / f"{subset}.json"
    print(f"Saving {json_path}...")
    with open(json_path, "w") as f:
        json.dump({"pairs": pairs}, f, indent=2)

    print(f"Successfully built {subset} with {len(pairs)} pairs!")


def build_from_huggingface(
    output_dir: str = ".",
    repo_id: str = "rl-research/multimodal-rewardbench-2",
    subsets: Optional[List[str]] = None,
):
    """
    Download and build all benchmark subsets from HuggingFace.

    Args:
        output_dir: Output directory for benchmark files
        repo_id: HuggingFace repository ID
        subsets: List of subsets to download (default: all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create image directories
    (output_path / "images").mkdir(exist_ok=True)
    (output_path / "input_images").mkdir(exist_ok=True)

    subsets_to_download = subsets or SUBSETS

    for subset in subsets_to_download:
        if subset not in SUBSETS:
            print(f"Warning: Unknown subset {subset}, skipping")
            continue

        try:
            download_and_build_subset(subset, repo_id, output_path)
        except Exception as e:
            print(f"Error downloading {subset}: {e}")
            continue

    print(f"\n{'='*60}")
    print("Build complete!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - {output_path}/t2i.json")
    print(f"  - {output_path}/edit.json")
    print(f"  - {output_path}/interleaved.json")
    print(f"  - {output_path}/reasoning.json")
    print(f"  - {output_path}/images/")
    print(f"  - {output_path}/input_images/")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build MMRB2 benchmark from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for benchmark files (default: current directory)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="rl-research/multimodal-rewardbench-2",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        choices=SUBSETS,
        help="Download only specific subsets",
    )

    args = parser.parse_args()

    build_from_huggingface(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        subsets=args.subset,
    )


if __name__ == "__main__":
    main()
