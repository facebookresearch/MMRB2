# Multimodal RewardBench 2 (MMRB2)

**Multimodal RewardBench 2** is a comprehensive benchmark for evaluating reward models on multimodal tasks including text-to-image generation, image editing, interleaved image-text generation, and visual reasoning.

## ⚠️ Important Notice

**This data is released under CC BY-NC 4.0 and is intended for benchmarking purposes only. This dataset should not be used for training models.**

Third-party content (prompts, images from source benchmarks) pulled from other locations are subject to their own licenses and you may have other legal obligations or restrictions that govern your use of that content.

**AI-Generated Content Disclosure**: This dataset contains outputs generated using artificial intelligence technologies, including but not limited to FLUX.1, and other generative models. Users should be aware that:
- All model outputs in this dataset were generated using AI systems
- Some outputs may be subject to additional license terms from respective model providers
- Users must comply with applicable laws regarding AI-generated content disclosure

## 📋 Overview

MMRB2 provides:
- **4 Task Categories**: T2I, Edit, Interleaved, and Reasoning
- **~4,000 Evaluation Pairs**: Human-annotated preference data
- **Diverse Sources**: Aggregated from 20+ benchmark datasets
- **Human Annotations**: High-quality preference labels indicating which model output is better
- **Standardized Evaluation**: Consistent evaluation protocol

## 🏗️ Repository Structure

```
MMRB2/
├── benchmark/           # Benchmark data and build scripts
│   ├── sources/         # Source modules for downloading data
│   ├── *_response_only.json  # Response data (prompts downloaded at build time)
│   ├── run_release.sh   # Main build script
│   └── ...
├── evaluate/            # Evaluation scripts (coming soon)
├── requirements.txt     # Python dependencies
├── LICENSE              # CC BY-NC 4.0 License
├── CODE_OF_CONDUCT.md   # Community guidelines
├── CONTRIBUTING.md      # Contribution guidelines
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/facebookresearch/MMRB2.git
cd MMRB2

# Install dependencies
pip install -r requirements.txt
```

### Building the Benchmark

The benchmark data requires downloading prompts from original sources and images from HuggingFace:

```bash
cd benchmark
./run_release.sh
```

This will:
1. Download response images from HuggingFace (`facebook/MMRB2_image`)
2. Download and merge prompts from original benchmark sources
3. Finalize the release with proper image paths
4. Clean up intermediate files

After building, you'll have:
- `t2i.json`, `edit.json`, `interleaved.json`, `reasoning.json` - Complete benchmark files
- `images/` - Response images
- `input_images/` - Input/prompt images

### Running Specific Tasks

```bash
./run_release.sh download    # Only download images
./run_release.sh t2i         # Only build T2I task
./run_release.sh edit        # Only build Edit task
./run_release.sh interleaved # Only build Interleaved task
./run_release.sh reasoning   # Only build Reasoning task
./run_release.sh finalize    # Only finalize (after all tasks)
./run_release.sh clean       # Only cleanup intermediate files
```

## 📊 Data Format

Each task JSON file contains pairs with the following structure:

```json
{
  "pairs": [
    {
      "pair_id": "unique_pair_id",
      "prompt_source": "source_benchmark_name",
      "prompt_content": [
        ["text", "Describe this image..."],
        ["image", "input_images/image.jpg"]
      ],
      "prompt_metadata": { ... },
      "response_a": {
        "model": "model_a_name",
        "response_content": [
          ["image", "images/response_a.jpg"],
          ["text", "Response text..."]
        ]
      },
      "response_b": {
        "model": "model_b_name", 
        "response_content": [
          ["image", "images/response_b.jpg"],
          ["text", "Response text..."]
        ]
      },
      "human_preference": "A" | "B" | "tie"
    }
  ]
}
```

## 📈 Benchmark Statistics

| Task | # Pairs | # Sources |
|------|---------|-----------|
| T2I | ~500 | 5 |
| Edit | ~500 | 6 |
| Interleaved | ~1000 | 4 |
| Reasoning | ~1000 | 6 |
| **Total** | **~4000** | **21** |

## 🔗 Data Sources

MMRB2 aggregates prompts from the following benchmarks:

**T2I**: OneIG-Bench, R2I-Bench, WISE, EvalMuse, RealUnify-UEG

**Edit**: Emu-Edit, RISEBench, HQ-Edit, DreamBench, DreamBooth

**Interleaved**: ISG-Bench, Chameleon, InterleavedEval, MMMG

**Reasoning**: BLINK, MindCube, MuirBench, RealUnify, VisuLogic, V*

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**. See the [LICENSE](LICENSE) file for details.

### Key Terms:
- **NonCommercial Use Only**: This data may not be used for commercial purposes
- **Benchmarking Only**: This dataset is intended for benchmarking and evaluation purposes only, not for training models
- **Attribution Required**: You must give appropriate credit when using this dataset

### Third-Party Content

This dataset includes:
- **Prompts** from various third-party benchmarks (subject to their original licenses)
- **Model outputs** from various AI models (subject to respective model licenses)

Users are responsible for ensuring compliance with all applicable licenses when using this dataset.

### Model Output Licenses

Some model outputs in this dataset may be subject to specific license terms:

- **FLUX.1 [dev]** outputs require compliance with content filtering and AI disclosure requirements
- Other model outputs may have their own license restrictions

**Note**: If upstream model authors/owners identify that our release breaches their output restriction terms, affected portions of the dataset may be removed.

## 📖 Citation

If you use MMRB2 in your research, please cite:

```bibtex
@article{mmrb2,
  title={Multimodal RewardBench 2: A Comprehensive Benchmark for Multimodal Reward Models},
  author={...},
  journal={...},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating in our community.
