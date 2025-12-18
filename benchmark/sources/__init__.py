# Copyright: Meta Platforms, Inc. and affiliates
"""
Source modules for downloading and processing benchmark data.

Each source module implements:
- download(data_dir): Download the source dataset
- load_prompts(data_dir): Load prompts into a lookup dictionary
"""

# Source configurations with URLs and commit hashes for reproducibility
SOURCES = {
    # T2I sources
    "oneigbench": {
        "type": "huggingface",
        "dataset": "OneIG-Bench/OneIG-Bench",
        "module": "oneigbench",
    },
    "r2ibench": {
        "type": "github",
        "url": "https://github.com/PLUM-Lab/R2I-Bench.git",
        "commit": "0cf0197d797c117ea4f9fc3b2a87a72bf7910d51",
        "module": "r2ibench",
    },
    "wise": {
        "type": "github",
        "url": "https://github.com/PKU-YuanGroup/WISE.git",
        "commit": "aeef292aaf1715bcd9cc5dc28810197ad8881570",
        "module": "wise",
    },
    "evalmuse": {
        "type": "github",
        "url": "https://github.com/DYEvaLab/EvalMuse.git",
        "commit": "45c8bb4af5d8c0d1d9f3733156177cc67ad9b7e5",
        "module": "evalmuse",
    },
    "realunify_ueg": {
        "type": "huggingface",
        "dataset": "DogNeverSleep/RealUnify",
        "module": "realunify_ueg",
    },
    # Image Editing sources
    "emu-edit": {
        "type": "huggingface",
        "dataset": "facebook/emu_edit_test_set",
        "module": "emu_edit",
    },
    "dreambench": {
        "type": "huggingface",
        "dataset": "yuangpeng/dreambench_plus",
        "module": "dreambench",
    },
    "dreambooth": {
        "type": "github",
        "url": "https://github.com/google/dreambooth.git",
        "module": "dreambooth",
    },
    "risebench": {
        "type": "huggingface",
        "dataset": "PhoenixZ/RISEBench",
        "module": "risebench",
    },
    "hq-edit": {
        "type": "huggingface",
        "dataset": "UCSC-VLAA/HQEdit-test",
        "module": "hq_edit",
    },
    # Interleaved generation sources
    "isgbench": {
        "type": "github",
        "url": "https://github.com/Dongping-Chen/ISG.git",
        "commit": "de504cd",
        "module": "isgbench",
    },
    "chameleon": {
        "type": "github",
        "url": "https://github.com/facebookresearch/chameleon.git",
        "commit": "dbbe47f",
        "module": "chameleon",
    },
    "interleavedeval": {
        "type": "huggingface",
        "repo": "mqliu/InterleavedBench",
        "repo_type": "model",
        "module": "interleavedeval",
    },
    "mmmg": {
        "type": "huggingface",
        "dataset": "UW-FMRL2/MMMG",
        "module": "mmmg",
    },
    # Reasoning sources
    "blink": {
        "type": "huggingface",
        "dataset": "BLINK-Benchmark/BLINK",
        "module": "blink",
    },
    "mindcube": {
        "type": "huggingface",
        "dataset": "MLL-Lab/MindCube",
        "module": "mindcube",
    },
    "muirbench": {
        "type": "huggingface",
        "dataset": "MUIRBENCH/MUIRBENCH",
        "module": "muirbench",
    },
    "realunify": {
        "type": "huggingface",
        "dataset": "DogNeverSleep/RealUnify",
        "module": "realunify",
    },
    "visulogic": {
        "type": "huggingface",
        "dataset": "VisuLogic/VisuLogic",
        "module": "visulogic",
    },
    "vstar": {
        "type": "huggingface",
        "dataset": "craigwu/vstar_bench",
        "module": "vstar",
    },
}


def get_source_config(source_name: str) -> dict:
    """Get configuration for a source."""
    if source_name not in SOURCES:
        raise ValueError(
            f"Unknown source: {source_name}. Available: {list(SOURCES.keys())}"
        )
    return SOURCES[source_name]


def list_sources() -> list:
    """List all available sources."""
    return list(SOURCES.keys())
