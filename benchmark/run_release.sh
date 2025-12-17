#!/bin/bash
# MMRB2 Release Pipeline
# 
# This script builds the full MMRB2 dataset by:
# 0. Downloading images from HuggingFace (optional, for fallback)
# 1. Downloading prompts from original sources
# 2. Merging prompts into the response-only JSON files
# 3. Finalizing the release with relative image paths
# 4. Cleaning up intermediate files
#
# Prerequisites:
#   - Python with datasets, huggingface_hub, tqdm, requests installed
#   - Git (for cloning source repositories)
#   - Sufficient disk space for downloaded data
#
# Usage:
#   cd release_clean
#   ./run_release.sh
#
# Or to run specific tasks:
#   ./run_release.sh download    # Download images from HuggingFace
#   ./run_release.sh t2i         # Only T2I task
#   ./run_release.sh edit        # Only Edit task
#   ./run_release.sh interleaved # Only Interleaved task
#   ./run_release.sh reasoning   # Only Reasoning task
#   ./run_release.sh finalize    # Only finalize (after all tasks)
#   ./run_release.sh clean       # Only cleanup intermediate files

set -e

# Change to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Directories
SOURCES_DIR="./sources"
IMAGES_DIR="./images"
INPUT_IMAGES_DIR="./input_images"

# Create directories
mkdir -p "$SOURCES_DIR"
mkdir -p "$INPUT_IMAGES_DIR"

# Source configurations
T2I_SOURCES="oneigbench r2ibench wise evalmuse realunify_ueg"
EDIT_SOURCES="emu-edit risebench hq-edit"
INTERLEAVED_SOURCES="isgbench chameleon interleavedeval mmmg"
REASONING_SOURCES="blink mindcube muirbench realunify visulogic vstar"

# HuggingFace image repository
HF_IMAGE_REPO="facebook/MMRB2_image"

# Function to download images from HuggingFace
download_images() {
    echo "========================================"
    echo "Downloading Images from HuggingFace"
    echo "========================================"
    
    python3 0_download_images.py \
        --output_dir "." \
        --repo "$HF_IMAGE_REPO"
    
    echo "Image download completed!"
}

# Function to run T2I task
run_t2i() {
    echo "========================================"
    echo "Processing T2I Task"
    echo "========================================"
    
    if [ ! -f "t2i_response_only.json" ]; then
        echo "Error: t2i_response_only.json not found"
        exit 1
    fi
    
    python3 2_download_and_merge.py \
        --input "t2i_response_only.json" \
        --output "t2i_full.json" \
        --data_dir "$SOURCES_DIR" \
        --sources $T2I_SOURCES \
        --download
    
    echo "T2I task completed!"
}

# Function to run Edit task
run_edit() {
    echo "========================================"
    echo "Processing Edit Task"
    echo "========================================"
    
    if [ ! -f "edit_response_only.json" ]; then
        echo "Error: edit_response_only.json not found"
        exit 1
    fi
    
    # Step 1: Download and merge standard edit sources
    python3 2_download_and_merge.py \
        --input "edit_response_only.json" \
        --output "edit_full.json" \
        --data_dir "$SOURCES_DIR" \
        --sources $EDIT_SOURCES \
        --download
    
    # Step 2: Process new edit prompts (dreambench, new_synthetic_edit, text_heavy_edit)
    if [ -f "new_task_release.json" ]; then
        echo ""
        echo "Processing new edit prompts..."
        python3 3_process_edit_prompts.py \
            --input "edit_full.json" \
            --prompts "new_task_release.json" \
            --output "edit_full.json" \
            --data_dir "$SOURCES_DIR" \
            --image_folder "$IMAGES_DIR" \
            --download
    fi
    
    echo "Edit task completed!"
}

# Function to run Interleaved task
run_interleaved() {
    echo "========================================"
    echo "Processing Interleaved Task"
    echo "========================================"
    
    if [ ! -f "interleaved_response_only.json" ]; then
        echo "Error: interleaved_response_only.json not found"
        exit 1
    fi
    
    python3 2_download_and_merge.py \
        --input "interleaved_response_only.json" \
        --output "interleaved_full.json" \
        --data_dir "$SOURCES_DIR" \
        --sources $INTERLEAVED_SOURCES \
        --download
    
    echo "Interleaved task completed!"
}

# Function to run Reasoning task
run_reasoning() {
    echo "========================================"
    echo "Processing Reasoning Task"
    echo "========================================"
    
    if [ ! -f "reasoning_response_only.json" ]; then
        echo "Error: reasoning_response_only.json not found"
        exit 1
    fi
    
    python3 2_download_and_merge.py \
        --input "reasoning_response_only.json" \
        --output "reasoning_full.json" \
        --data_dir "$SOURCES_DIR" \
        --sources $REASONING_SOURCES \
        --download
    
    echo "Reasoning task completed!"
}

# Function to finalize release
run_finalize() {
    echo "========================================"
    echo "Finalizing Release"
    echo "========================================"
    
    python3 4_finalize_release.py \
        --release_dir "." \
        --input_images_dir "$INPUT_IMAGES_DIR"
    
    echo ""
    echo "========================================"
    echo "Release Complete!"
    echo "========================================"
    echo ""
    echo "Output files:"
    echo "  - t2i.json"
    echo "  - edit.json"
    echo "  - interleaved.json"
    echo "  - reasoning.json"
    echo ""
    echo "Image folders:"
    echo "  - images/        : Response images"
    echo "  - input_images/  : Input prompt images"
    echo ""
}

# Function to clean up intermediate files
run_cleanup() {
    echo "========================================"
    echo "Cleaning Up Intermediate Files"
    echo "========================================"
    
    # Remove downloaded source data (subdirectories only, preserve .py files)
    if [ -d "$SOURCES_DIR" ]; then
        echo "Removing downloaded data from sources directory..."
        # Only remove subdirectories (downloaded data), keep .py files and __init__.py
        find "$SOURCES_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;
        # Also remove __pycache__ if it exists
        rm -rf "$SOURCES_DIR/__pycache__"
        echo "  Kept source code (.py files)"
    fi
    
    # Remove intermediate _full.json files
    for f in t2i_full.json edit_full.json interleaved_full.json reasoning_full.json; do
        if [ -f "$f" ]; then
            echo "Removing intermediate file: $f"
            rm -f "$f"
        fi
    done
    
    echo "Cleanup completed!"
}

# Main
case "${1:-all}" in
    download)
        download_images
        ;;
    t2i)
        run_t2i
        ;;
    edit)
        run_edit
        ;;
    interleaved)
        run_interleaved
        ;;
    reasoning)
        run_reasoning
        ;;
    finalize)
        run_finalize
        ;;
    clean)
        run_cleanup
        ;;
    all)
        echo "========================================"
        echo "MMRB2 Full Release Pipeline"
        echo "========================================"
        echo ""
        download_images
        echo ""
        run_t2i
        echo ""
        run_edit
        echo ""
        run_interleaved
        echo ""
        run_reasoning
        echo ""
        run_finalize
        echo ""
        run_cleanup
        ;;
    *)
        echo "Usage: $0 [download|t2i|edit|interleaved|reasoning|finalize|clean|all]"
        exit 1
        ;;
esac

