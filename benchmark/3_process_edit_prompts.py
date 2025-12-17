# Copyright: Meta Platforms, Inc. and affiliates
"""
Process edit prompts from new_task_release.json and merge into edit_full.json.

This script:
1. Reads edit_full.json (existing processed pairs)
2. Reads new_task_release.json with prompts (dreambench, new_synthetic_edit, text_heavy_edit)
3. For dreambench: downloads images via sources/dreambench.py, references from sources folder
4. For new_synthetic_edit/text_heavy_edit:
   - dreambench_source images -> from dreambench sources
   - dreambench/dreambooth images -> from respective sources
   - new_images/* -> from images/ folder (already present)
   - synthetic_images/* -> from images/ folder (already present)
5. Adds prompt_content to pairs missing it and outputs updated edit_full.json

Usage:
    python 3_process_edit_prompts.py \
        --input ./edit_full.json \
        --prompts ./new_task_release.json \
        --output ./edit_full.json \
        --data_dir ./sources \
        --image_folder ./images \
        --download
"""

import argparse
import importlib
import json
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# DreamBooth subjects (from dreambooth.py)
DREAMBOOTH_SUBJECTS = {
    # Live subjects
    "cat", "cat2", "dog", "dog2", "dog3", "dog5", "dog6", "dog7", "dog8",
    # Objects
    "backpack", "backpack_dog", "bear_plushie", "berry_bowl", "can", "candle",
    "clock", "colorful_sneaker", "duck_toy", "fancy_boot", "grey_sloth_plushie",
    "monster_toy", "pink_sunglasses", "poop_emoji", "rc_car", "red_cartoon",
    "robot_toy", "shiny_sneaker", "teapot", "vase", "wolf_plushie",
}


def is_dreambench_image(image_name: str) -> bool:
    """Check if an image name is from DreamBench based on naming pattern."""
    return (
        image_name.startswith("live_subject_animal_")
        or image_name.startswith("live_subject_human_")
        or image_name.startswith("object_")
        or image_name.startswith("style_")
    )


def is_dreambooth_image(image_name: str) -> bool:
    """Check if an image name is from DreamBooth based on subject list."""
    return image_name in DREAMBOOTH_SUBJECTS


def get_image_source(image_path: str) -> Tuple[str, str, str]:
    """
    Determine the source of an image and return (source_type, image_name, extra_info).
    
    Handles relative identifiers:
    - dreambench_source:object_39_guitar/1_0.jpg
    - multi-image/images/live_subject_animal_23_lion.jpg
    - new_images/Futuristic_hoverboard.jpg
    - synthetic_images/GPTGeminiAgent_...jpg
    
    Returns:
        Tuple of (source_type, image_name, extra_info) where source_type is one of:
        - "dreambench_source": from dreambench source (extra_info = category/file)
        - "dreambench": from multi-image/images, originally dreambench
        - "dreambooth": from multi-image/images, originally dreambooth
        - "new_images": from new_images (already in images/)
        - "synthetic_images": from synthetic_images (already in images/)
        - "unknown": cannot determine
    """
    # New format: dreambench_source:category/file
    if image_path.startswith("dreambench_source:"):
        category_file = image_path.split(":", 1)[1]
        return "dreambench_source", category_file, category_file
    
    # New format: multi-image/images/filename
    if image_path.startswith("multi-image/images/"):
        filename = image_path.split("/")[-1]
        image_name = os.path.splitext(filename)[0]
        if is_dreambench_image(image_name):
            return "dreambench", image_name, filename
        elif is_dreambooth_image(image_name):
            return "dreambooth", image_name, filename
        else:
            return "unknown", image_name, filename
    
    # New format: new_images/filename
    if image_path.startswith("new_images/"):
        filename = image_path.split("/")[-1]
        return "new_images", filename, filename
    
    # New format: synthetic_images/filename
    if image_path.startswith("synthetic_images/"):
        filename = image_path.split("/")[-1]
        return "synthetic_images", filename, filename
    
    return "unknown", os.path.basename(image_path), image_path


def download_sources(data_dir: str, sources: List[str], force: bool = False) -> None:
    """Download required source datasets."""
    for source in sources:
        try:
            module = importlib.import_module(f"sources.{source}")
            print(f"Downloading: {source}")
            module.download(data_dir, force=force)
        except Exception as e:
            print(f"Error downloading {source}: {e}")


def build_dreambench_image_mapping(data_dir: str) -> Dict[str, str]:
    """
    Build mapping from multi-image naming to dreambench source paths.
    
    Returns:
        Dict mapping image_name prefix (e.g., "live_subject_animal_00") to 
        relative path in sources folder (e.g., "sources/dreambench/output_images/live_subject_animal_00.jpg")
    """
    source_dir = os.path.join(data_dir, "dreambench")
    images_dir = os.path.join(source_dir, "images")
    output_images_dir = os.path.join(source_dir, "output_images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Warning: DreamBench images not found at {images_dir}")
        return {}
    
    image_mapping = {}
    
    # Walk through all categories
    for category in ["live_subject/animal", "live_subject/human", "object", "style"]:
        category_dir = os.path.join(images_dir, category)
        if not os.path.exists(category_dir):
            continue
            
        for img_file in os.listdir(category_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
                
            img_id = os.path.splitext(img_file)[0]  # e.g., "00"
            
            # Build the image name used in multi-image/images
            # e.g., "live_subject/animal" + "00" -> "live_subject_animal_00"
            image_name_prefix = category.replace("/", "_") + "_" + img_id
            
            src_path = os.path.join(category_dir, img_file)
            output_name = f"{category.replace('/', '_')}_{img_file}"
            output_path = os.path.join(output_images_dir, output_name)
            
            # Copy to output_images if not exists (for consistent referencing)
            if not os.path.exists(output_path):
                shutil.copy(src_path, output_path)
            
            # Map to relative path in sources folder
            image_mapping[image_name_prefix] = os.path.join(
                "sources", "dreambench", "output_images", output_name
            )
    
    return image_mapping


def build_dreambooth_image_mapping(data_dir: str) -> Dict[str, str]:
    """
    Build mapping from subject name to dreambooth source paths.
    
    Returns:
        Dict mapping subject name to relative path in sources folder
    """
    source_dir = os.path.join(data_dir, "dreambooth")
    images_dir = os.path.join(source_dir, "images")
    output_images_dir = os.path.join(source_dir, "output_images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Warning: DreamBooth images not found at {images_dir}")
        return {}
    
    image_mapping = {}
    
    for subject in os.listdir(images_dir):
        subject_dir = os.path.join(images_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
            
        # Use 00.jpg as the canonical image (matching the notebook)
        canonical_img = os.path.join(subject_dir, "00.jpg")
        if not os.path.exists(canonical_img):
            # Try first available image
            images = [f for f in os.listdir(subject_dir) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                canonical_img = os.path.join(subject_dir, images[0])
            else:
                continue
        
        output_name = f"{subject}.jpg"
        output_path = os.path.join(output_images_dir, output_name)
        
        # Copy to output_images if not exists
        if not os.path.exists(output_path):
            shutil.copy(canonical_img, output_path)
        
        # Map to relative path in sources folder
        image_mapping[subject] = os.path.join(
            "sources", "dreambooth", "output_images", output_name
        )
    
    return image_mapping


def process_image_path(
    image_path: str,
    dreambench_images: Dict[str, str],
    dreambooth_images: Dict[str, str],
    image_folder: str = None,
) -> Optional[str]:
    """
    Process an image path and return the new relative path.
    
    Args:
        image_path: Original image path (can be relative identifier)
        dreambench_images: Mapping of dreambench image names to relative paths
        dreambooth_images: Mapping of dreambooth subject names to relative paths
        image_folder: Path to the images folder (for new_images/synthetic_images)
    
    Returns:
        New image path (relative), or None if not found
    """
    source_type, image_name, extra_info = get_image_source(image_path)
    
    if source_type == "new_images":
        # Already in images/ folder, return just the filename
        return image_name
    
    if source_type == "synthetic_images":
        # Already copied to images/ folder, return just the filename
        return image_name
    
    if source_type == "dreambench_source":
        # Map from category/file (e.g., "object_39_guitar/1_0.jpg") to source path
        # Try to extract category prefix and find a match
        # image_name is like "object_39_guitar/1_0.jpg"
        parts = image_name.split("/")
        if len(parts) == 2:
            category = parts[0]  # e.g., "object_39_guitar"
            # Extract the base category (e.g., "object_39")
            category_parts = category.split("_")
            if category.startswith("live_subject_animal_"):
                base_category = "_".join(category_parts[:4])  # live_subject_animal_XX
            elif category.startswith("live_subject_human_"):
                base_category = "_".join(category_parts[:4])
            elif category.startswith("object_"):
                base_category = "_".join(category_parts[:2])  # object_XX
            elif category.startswith("style_"):
                base_category = "_".join(category_parts[:2])  # style_XX
            else:
                base_category = category
            
            # Find in dreambench_images mapping
            for prefix in dreambench_images:
                if prefix == base_category:
                    return dreambench_images[prefix]
        return None
    
    if source_type == "dreambench":
        # Find matching dreambench image
        # image_name is like "live_subject_animal_23_lion"
        # We need to match to "live_subject_animal_23"
        for prefix in dreambench_images:
            if image_name.startswith(prefix):
                return dreambench_images[prefix]
        return None
    
    if source_type == "dreambooth":
        # Find matching dreambooth image
        if image_name in dreambooth_images:
            return dreambooth_images[image_name]
        return None
    
    return None


def process_edit_prompts(
    input_path: str,
    prompts_path: str,
    output_path: str,
    data_dir: str,
    image_folder: str,
) -> Dict[str, Any]:
    """
    Process edit prompts and merge into the existing edit_full.json.
    
    Args:
        input_path: Path to edit_full.json (existing processed pairs)
        prompts_path: Path to new_task_release.json (prompts to add)
        output_path: Path to save updated edit_full.json
        data_dir: Directory containing downloaded source data
        image_folder: Target folder for release images (for reference)
    
    Returns:
        Statistics about the processing
    """
    # Load existing edit_full.json
    print(f"Loading existing JSON: {input_path}")
    with open(input_path, "r") as f:
        existing_data = json.load(f)
    
    existing_pairs = existing_data["pairs"]
    print(f"Found {len(existing_pairs)} existing pairs")
    
    # Build lookup by id
    pairs_by_id = {p["id"]: p for p in existing_pairs}
    
    # Load prompts to add
    print(f"Loading prompts JSON: {prompts_path}")
    with open(prompts_path, "r") as f:
        prompts_data = json.load(f)
    
    prompt_pairs = prompts_data["pairs"]
    print(f"Found {len(prompt_pairs)} prompt pairs to process")
    
    # Build image mappings
    print("Building DreamBench image mapping...")
    dreambench_images = build_dreambench_image_mapping(data_dir)
    print(f"  Found {len(dreambench_images)} DreamBench image mappings")
    
    print("Building DreamBooth image mapping...")
    dreambooth_images = build_dreambooth_image_mapping(data_dir)
    print(f"  Found {len(dreambooth_images)} DreamBooth image mappings")
    
    # Track statistics
    stats = {
        "total": len(prompt_pairs),
        "processed": 0,
        "failed": 0,
        "not_found": 0,
        "by_source": {},
    }
    
    for prompt_pair in tqdm(prompt_pairs, desc="Processing edit prompts"):
        source = prompt_pair["prompt_source"]
        pair_id = prompt_pair["pair_id"]
        
        # Initialize source stats
        if source not in stats["by_source"]:
            stats["by_source"][source] = {"processed": 0, "failed": 0, "not_found": 0}
        
        # Find matching pair in existing data
        if pair_id not in pairs_by_id:
            stats["not_found"] += 1
            stats["by_source"][source]["not_found"] += 1
            continue
        
        existing_pair = pairs_by_id[pair_id]
        
        if source in ["dreambench", "new_synthetic_edit", "text_heavy_edit"]:
            # Process each image in prompt_content
            new_content = []
            all_images_found = True
            
            for item in prompt_pair["prompt_content"]:
                if item[0] == "image":
                    new_path = process_image_path(
                        item[1],
                        dreambench_images,
                        dreambooth_images,
                        image_folder,
                    )
                    if new_path:
                        new_content.append(["image", new_path])
                    else:
                        all_images_found = False
                        print(f"Warning: Could not process image: {item[1]}")
                        break
                else:
                    new_content.append(item)
            
            if all_images_found:
                existing_pair["prompt_content"] = new_content
                # Keep existing prompt_metadata but could update if needed
                stats["processed"] += 1
                stats["by_source"][source]["processed"] += 1
            else:
                stats["failed"] += 1
                stats["by_source"][source]["failed"] += 1
        
        else:
            # Unknown source
            stats["failed"] += 1
    
    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output_data = {"pairs": existing_pairs}
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print("Statistics:")
    print(f"  Total prompts: {stats['total']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Not found in existing: {stats['not_found']}")
    print("\nBy source:")
    for source, source_stats in stats["by_source"].items():
        print(f"  {source}: processed={source_stats['processed']}, failed={source_stats['failed']}, not_found={source_stats['not_found']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process edit prompts and merge into edit_full.json"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to edit_full.json (existing processed pairs)",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="Path to new_task_release.json (prompts to add)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save updated edit_full.json",
    )
    parser.add_argument(
        "--data_dir",
        default="./sources",
        help="Directory to save/load downloaded source data",
    )
    parser.add_argument(
        "--image_folder",
        default="./images",
        help="Target folder for release images (for reference)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download source data before processing",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if data exists",
    )
    
    args = parser.parse_args()
    
    # Check inputs exist
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    if not os.path.exists(args.prompts):
        print(f"Error: Prompts file not found: {args.prompts}")
        return 1
    
    # Download sources if requested
    if args.download:
        print("Downloading source data...")
        download_sources(
            args.data_dir,
            ["dreambench", "dreambooth"],
            force=args.force_download,
        )
    
    # Process
    print("\nProcessing edit prompts...")
    process_edit_prompts(
        args.input,
        args.prompts,
        args.output,
        args.data_dir,
        args.image_folder,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

