# Copyright: Meta Platforms, Inc. and affiliates
"""
Finalize release JSON files by copying all prompt images to input_images folder
and updating all image paths to be relative.

This script:
1. Reads each *_full.json file
2. Copies all images in prompt_content to the input_images folder
3. Renames images to avoid conflicts (e.g., sources/dreambench/output_images/x.jpg -> dreambench_output_images_x.jpg)
4. Creates cleaned JSON files (t2i.json, edit.json, etc.) with relative paths
5. All response images remain in images/ folder with relative paths

Usage:
    python 4_finalize_release.py \
        --release_dir . \
        --input_images_dir ./input_images
"""

import argparse
import json
import os
import shutil
from typing import Dict, List, Tuple

from tqdm import tqdm


def normalize_image_name(image_path: str, actual_path: str = None) -> str:
    """
    Convert an image path to a flat filename that avoids conflicts.
    
    Uses the actual file's extension if provided (to handle case mismatches).
    
    Examples:
        sources/dreambench/output_images/object_39.jpg -> dreambench_output_images_object_39.jpg
        sources/dreambooth/output_images/vase.jpg -> dreambooth_output_images_vase.jpg
        sources/emu-edit/images/test_global_2121.jpg -> emu-edit_images_test_global_2121.jpg
        emu-edit/images/test_global_2121.jpg -> emu-edit_images_test_global_2121.jpg
        Futuristic_hoverboard.jpg -> Futuristic_hoverboard.jpg
        GPTGeminiAgent_xxx.jpg -> GPTGeminiAgent_xxx.jpg
    
    Args:
        image_path: The image path from JSON
        actual_path: The actual file path (may have different extension case)
    """
    # If it's already a simple filename (no path separators), keep as is
    if '/' not in image_path:
        # But use actual extension if available
        if actual_path and os.path.exists(actual_path):
            base = os.path.splitext(image_path)[0]
            actual_ext = os.path.splitext(actual_path)[1]
            return base + actual_ext
        return image_path
    
    # Remove 'sources/' prefix if present
    if image_path.startswith('sources/'):
        image_path = image_path[len('sources/'):]
    
    # Replace '/' with '_' to flatten the path
    flat_name = image_path.replace('/', '_')
    
    # Use actual extension if available (handles .JPG vs .jpg)
    if actual_path and os.path.exists(actual_path):
        base = os.path.splitext(flat_name)[0]
        actual_ext = os.path.splitext(actual_path)[1]
        flat_name = base + actual_ext
    
    return flat_name


def try_find_file(base_path: str) -> str:
    """
    Try to find a file, handling case sensitivity for extensions.
    
    Args:
        base_path: The expected file path
    
    Returns:
        The actual file path if found, otherwise the original path
    """
    if os.path.exists(base_path):
        return base_path
    
    # Try different extension cases (.JPG -> .jpg, .PNG -> .png, etc.)
    base, ext = os.path.splitext(base_path)
    if ext:
        # Try lowercase
        alt_path = base + ext.lower()
        if os.path.exists(alt_path):
            return alt_path
        # Try uppercase
        alt_path = base + ext.upper()
        if os.path.exists(alt_path):
            return alt_path
    
    return base_path


def get_full_image_path(image_path: str, release_dir: str, images_dir: str) -> str:
    """
    Get the full filesystem path for an image reference.
    
    Args:
        image_path: The image path from the JSON (could be relative or just filename)
        release_dir: The release directory
        images_dir: The images directory within release
    
    Returns:
        Full filesystem path to the image (with case-insensitive extension matching)
    """
    if image_path.startswith('sources/'):
        full_path = os.path.join(release_dir, image_path)
    elif '/' in image_path:
        # Path with subdirectory but not starting with 'sources/'
        # Could be like 'emu-edit/images/...' which should be in sources/
        full_path = os.path.join(release_dir, 'sources', image_path)
    else:
        # Simple filename, should be in images folder
        full_path = os.path.join(images_dir, image_path)
    
    # Try to find the file with case-insensitive extension matching
    return try_find_file(full_path)


def process_json_file(
    input_path: str,
    output_path: str,
    release_dir: str,
    images_dir: str,
    input_images_dir: str,
) -> Dict[str, int]:
    """
    Process a single JSON file, copying images and updating paths.
    
    Args:
        input_path: Path to input *_full.json
        output_path: Path to output cleaned JSON
        release_dir: The release directory
        images_dir: The images directory within release
        input_images_dir: Target directory for all input images
    
    Returns:
        Statistics dict
    """
    print(f"\nProcessing: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    stats = {
        'total_pairs': len(pairs),
        'pairs_with_prompt': 0,
        'images_copied': 0,
        'images_missing': 0,
        'images_skipped': 0,
    }
    
    # Track copied images to avoid duplicates
    copied_images: Dict[str, str] = {}  # original_path -> new_name
    
    for pair in tqdm(pairs, desc=f"Processing {os.path.basename(input_path)}"):
        # Update response image paths to be relative (images/filename)
        for response_key in ['response_a', 'response_b']:
            if response_key in pair:
                response = pair[response_key]
                if 'response_content' in response:
                    new_content = []
                    for item in response['response_content']:
                        if item[0] == 'image':
                            # Convert to images/filename format
                            filename = os.path.basename(item[1])
                            new_content.append(['image', f'images/{filename}'])
                        else:
                            new_content.append(item)
                    response['response_content'] = new_content
        
        # Process prompt_content
        if 'prompt_content' not in pair:
            continue
        
        stats['pairs_with_prompt'] += 1
        new_content = []
        
        for item in pair['prompt_content']:
            if item[0] != 'image':
                new_content.append(item)
                continue
            
            original_path = item[1]
            
            # Check if we've already processed this image
            if original_path in copied_images:
                new_content.append(['image', f'input_images/{copied_images[original_path]}'])
                stats['images_skipped'] += 1
                continue
            
            # Get full path (with case-insensitive extension matching)
            full_path = get_full_image_path(original_path, release_dir, images_dir)
            
            # Get new name using actual file's extension
            new_name = normalize_image_name(original_path, full_path)
            
            # Copy the image
            if os.path.exists(full_path):
                dest_path = os.path.join(input_images_dir, new_name)
                if not os.path.exists(dest_path):
                    shutil.copy(full_path, dest_path)
                copied_images[original_path] = new_name
                new_content.append(['image', f'input_images/{new_name}'])
                stats['images_copied'] += 1
            else:
                print(f"  Warning: Image not found: {full_path}")
                # Keep original path if not found
                new_content.append(item)
                stats['images_missing'] += 1
        
        pair['prompt_content'] = new_content
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  Saved: {output_path}")
    print(f"  Pairs with prompt: {stats['pairs_with_prompt']}")
    print(f"  Images copied: {stats['images_copied']}")
    print(f"  Images skipped (duplicates): {stats['images_skipped']}")
    print(f"  Images missing: {stats['images_missing']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Finalize release JSON files"
    )
    parser.add_argument(
        "--release_dir",
        default=".",
        help="Path to release directory containing *_full.json files",
    )
    parser.add_argument(
        "--input_images_dir",
        default="./input_images",
        help="Target directory for all input images",
    )
    
    args = parser.parse_args()
    
    release_dir = args.release_dir
    input_images_dir = args.input_images_dir
    images_dir = os.path.join(release_dir, 'images')
    
    # Create input_images directory
    os.makedirs(input_images_dir, exist_ok=True)
    
    # Define input/output file mappings
    file_mappings = [
        ('t2i_full.json', 't2i.json'),
        ('edit_full.json', 'edit.json'),
        ('interleaved_full.json', 'interleaved.json'),
        ('reasoning_full.json', 'reasoning.json'),
    ]
    
    total_stats = {
        'total_pairs': 0,
        'pairs_with_prompt': 0,
        'images_copied': 0,
        'images_missing': 0,
        'images_skipped': 0,
    }
    
    for input_name, output_name in file_mappings:
        input_path = os.path.join(release_dir, input_name)
        output_path = os.path.join(release_dir, output_name)
        
        if not os.path.exists(input_path):
            print(f"\nSkipping {input_name}: not found")
            continue
        
        stats = process_json_file(
            input_path,
            output_path,
            release_dir,
            images_dir,
            input_images_dir,
        )
        
        for key in total_stats:
            total_stats[key] += stats[key]
    
    print("\n" + "=" * 50)
    print("TOTAL STATISTICS")
    print("=" * 50)
    print(f"Total pairs: {total_stats['total_pairs']}")
    print(f"Pairs with prompt: {total_stats['pairs_with_prompt']}")
    print(f"Images copied: {total_stats['images_copied']}")
    print(f"Images skipped (duplicates): {total_stats['images_skipped']}")
    print(f"Images missing: {total_stats['images_missing']}")
    
    # Count unique images in input_images_dir
    if os.path.exists(input_images_dir):
        unique_images = len(os.listdir(input_images_dir))
        print(f"Unique images in {input_images_dir}: {unique_images}")
    
    return 0


if __name__ == "__main__":
    exit(main())

