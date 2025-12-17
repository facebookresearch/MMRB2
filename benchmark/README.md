# MMRB2 Benchmark Build System

This directory contains the scripts and data needed to build the complete MMRB2 benchmark.

## Directory Structure

```
benchmark/
├── sources/                    # Source modules for downloading benchmark data
│   ├── __init__.py            # Source registry and configuration
│   ├── blink.py               # BLINK benchmark loader
│   ├── chameleon.py           # Chameleon benchmark loader
│   └── ...                    # Other source loaders
├── t2i_response_only.json     # T2I task responses (without prompts)
├── edit_response_only.json    # Edit task responses
├── interleaved_response_only.json
├── reasoning_response_only.json
├── new_task_release.json      # Additional edit prompts
├── 0_download_images.py       # Download images from HuggingFace
├── 2_download_and_merge.py    # Download and merge prompts
├── 3_process_edit_prompts.py  # Process edit task prompts
├── 4_finalize_release.py      # Finalize with relative paths
└── run_release.sh             # Main build script
```

## Building the Benchmark

### Prerequisites

1. Python 3.8+
2. Install dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```
3. Git (for cloning source repositories)

### Quick Build

Run the complete build pipeline:

```bash
./run_release.sh
```

This will:
1. **Download images** from HuggingFace (`facebook/MMRB2_image`)
2. **Build T2I task**: Download prompts from OneIG-Bench, R2I-Bench, WISE, EvalMuse, RealUnify-UEG
3. **Build Edit task**: Download prompts from Emu-Edit, RISEBench, HQ-Edit + process DreamBench data
4. **Build Interleaved task**: Download prompts from ISG-Bench, Chameleon, InterleavedEval, MMMG
5. **Build Reasoning task**: Download prompts from BLINK, MindCube, MuirBench, RealUnify, VisuLogic, V*
6. **Finalize**: Copy images to input_images/, update paths
7. **Cleanup**: Remove intermediate files

### Step-by-Step Build

You can also run individual steps:

```bash
# Step 0: Download images from HuggingFace
./run_release.sh download

# Step 1-4: Build individual tasks
./run_release.sh t2i
./run_release.sh edit
./run_release.sh interleaved
./run_release.sh reasoning

# Step 5: Finalize release
./run_release.sh finalize

# Step 6: Cleanup intermediate files
./run_release.sh clean
```

## Output Files

After building, you'll have:

| File | Description |
|------|-------------|
| `t2i.json` | Complete T2I benchmark |
| `edit.json` | Complete Edit benchmark |
| `interleaved.json` | Complete Interleaved benchmark |
| `reasoning.json` | Complete Reasoning benchmark |
| `images/` | Response images |
| `input_images/` | Input/prompt images |

## Data Format

### Response-Only Files (`*_response_only.json`)

These contain only the responses (no prompts). The build process merges prompts from original sources:

```json
{
  "pairs": [
    {
      "pair_id": "...",
      "prompt_source": "blink",
      "prompt_metadata": {"id": "...", ...},
      "response_a": {...},
      "response_b": {...},
      "human_preference": "A"
    }
  ]
}
```

### Complete Files (`*.json`)

After building, the complete files include `prompt_content`:

```json
{
  "pairs": [
    {
      "pair_id": "...",
      "prompt_source": "blink",
      "prompt_content": [
        ["image", "input_images/..."],
        ["text", "Question text..."]
      ],
      "prompt_metadata": {...},
      "response_a": {
        "model": "...",
        "response_content": [
          ["image", "images/..."],
          ["text", "..."]
        ]
      },
      "response_b": {...},
      "human_preference": "A"
    }
  ]
}
```

## Source Modules

Each source module in `sources/` implements:

- `download(data_dir, force=False)`: Download source data
- `load_prompts(data_dir, required_keys=None)`: Load and return prompts

### Adding New Sources

1. Create a new module in `sources/` (e.g., `sources/new_source.py`)
2. Implement `download()` and `load_prompts()` functions
3. Register in `sources/__init__.py`

## Troubleshooting

### HuggingFace Authentication

If you encounter authentication errors:
```bash
huggingface-cli login
```

### Missing Dependencies

```bash
pip install datasets huggingface_hub tqdm requests Pillow
```

### Disk Space

The complete benchmark requires approximately:
- ~2GB for images
- ~500MB for downloaded source data (cleaned up after build)

## License

See the main repository LICENSE file.

