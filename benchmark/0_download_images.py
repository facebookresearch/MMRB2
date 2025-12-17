# Copyright: Meta Platforms, Inc. and affiliates
"""
Download images from HuggingFace dataset repository.

This script downloads and extracts the image archives from HuggingFace.

Prerequisites:
    pip install huggingface_hub

Usage:
    python 0_download_images.py
    python 0_download_images.py --output_dir /path/to/release
"""

import argparse
import os
import zipfile

from huggingface_hub import hf_hub_download
from tqdm import tqdm

HF_REPO = "facebook/MMRB2_image"

# Files to download (only response images, input_images are generated during processing)
IMAGE_FILES = [
    ("images.zip", "images"),  # Response images
]


def download_and_extract(repo_id: str, filename: str, output_dir: str, extract_to: str):
    """
    Download a zip file from HuggingFace and extract it.

    Args:
        repo_id: HuggingFace dataset repo ID
        filename: Name of the zip file in the repo
        output_dir: Directory to save/extract to
        extract_to: Name of the folder to extract into
    """
    extract_path = os.path.join(output_dir, extract_to)

    # Check if already extracted
    if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
        print(f"[{filename}] Already extracted to {extract_path}")
        return

    print(f"[{filename}] Downloading from HuggingFace: {repo_id}")

    # Download the zip file
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=output_dir,
    )

    print(f"[{filename}] Extracting to {extract_path}")

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of files for progress bar
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc=f"Extracting {filename}"):
            zip_ref.extract(file, output_dir)

    # Remove zip file after extraction to save space
    os.remove(zip_path)
    print(f"[{filename}] Done! Extracted {len(file_list)} files")


def main():
    parser = argparse.ArgumentParser(
        description="Download MMRB2 images from HuggingFace"
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for extracted images",
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("MMRB2 Image Download")
    print("=" * 50)
    print(f"Repository: {args.repo}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("")

    for filename, extract_to in IMAGE_FILES:
        try:
            download_and_extract(args.repo, filename, args.output_dir, extract_to)
        except Exception as e:
            print(f"[{filename}] Error: {e}")
            print(f"[{filename}] You may need to run: huggingface-cli login")
            return 1

    print("")
    print("=" * 50)
    print("Download Complete!")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
