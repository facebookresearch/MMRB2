#!/bin/bash
# MMRB2 Benchmark - GPT-4o Evaluation Script
# Runs all tasks sequentially with multi-process parallelism

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Data paths
BASE_DATA_PATH="${MMRB2_DATA_PATH:-$SCRIPT_DIR/../../benchmark}"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Evaluator
EVALUATOR="gpt4o-pairwise"

# Number of parallel workers (API-based)
N_GPU=32

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "MMRB2 Benchmark - GPT-4o Pairwise Evaluation"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Data path: $BASE_DATA_PATH"
echo "Parallel workers: $N_GPU"
echo ""

# Task 1: Text-to-Image Generation
echo "[Task 1/4] Text-to-Image Generation (image)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type image \
    --pairs_path "$BASE_DATA_PATH/t2i.json" \
    --output_path "$OUTPUT_DIR/task1_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 1 complete!"
echo ""

# Task 2: Image Editing
echo "[Task 2/4] Image Editing (edit)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type edit \
    --pairs_path "$BASE_DATA_PATH/edit.json" \
    --output_path "$OUTPUT_DIR/task2_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 2 complete!"
echo ""

# Task 3: Interleaved Generation
echo "[Task 3/4] Interleaved Generation (interleaved)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type interleaved \
    --pairs_path "$BASE_DATA_PATH/interleaved.json" \
    --output_path "$OUTPUT_DIR/task3_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 3 complete!"
echo ""

# Task 4: Visual Reasoning
echo "[Task 4/4] Visual Reasoning (reasoning)"
python multi_gpu_evaluate.py \
    --evaluator_name "$EVALUATOR" \
    --task_type reasoning \
    --pairs_path "$BASE_DATA_PATH/reasoning.json" \
    --output_path "$OUTPUT_DIR/task4_${EVALUATOR}.json" \
    --n_gpu $N_GPU \
    --n 1
echo "Task 4 complete!"
echo ""

echo "=========================================="
echo "All tasks completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

