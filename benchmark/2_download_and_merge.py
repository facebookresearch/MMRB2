# Copyright: Meta Platforms, Inc. and affiliates
"""
Download benchmark data and merge prompts into the release JSON.

This script:
1. Reads the release JSON (which has prompt_source and prompt_metadata.id)
2. Downloads the original benchmark data for each source
3. Merges the prompt_content and full prompt_metadata into the release JSON
4. Outputs the complete dataset

Usage:
    python 2_download_and_merge.py \
        --input ./t2i_processed.json \
        --output ./t2i_full.json \
        --data_dir ./sources
"""

import argparse
import gc
import importlib
import inspect
import json
import os
from typing import Any, Dict, List

from sources import SOURCES, get_source_config
from tqdm import tqdm


def download_source(source_name: str, data_dir: str, force: bool = False) -> None:
    """
    Download a single source dataset.

    Args:
        source_name: Name of the source (e.g., 'oneigbench')
        data_dir: Directory to save downloaded data
        force: If True, re-download even if data exists
    """
    config = get_source_config(source_name)
    module_name = config["module"]

    # Import the source module
    module = importlib.import_module(f"sources.{module_name}")

    # Download
    module.download(data_dir, force=force)


def load_source_prompts(
    source_name: str, data_dir: str, required_keys: set = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from a downloaded source.

    Args:
        source_name: Name of the source
        data_dir: Directory containing downloaded data
        required_keys: Optional set of lookup keys to load (for filtering)

    Returns:
        Dictionary mapping lookup_key -> {prompt_content, metadata}
        Note: lookup_key format varies by source (some use compound keys)
    """
    config = get_source_config(source_name)
    module_name = config["module"]

    # Import the source module
    module = importlib.import_module(f"sources.{module_name}")

    # Check if the module's load_prompts supports required_keys filtering
    sig = inspect.signature(module.load_prompts)
    if "required_keys" in sig.parameters:
        return module.load_prompts(data_dir, required_keys=required_keys)
    else:
        return module.load_prompts(data_dir)


def get_lookup_key(source_name: str, prompt_metadata: Dict[str, Any]) -> str:
    """
    Get the lookup key for a prompt based on its source.

    Different sources may require different key formats:
    - oneigbench: uses compound key (id_category) since IDs repeat across categories
    - r2ibench: ID is already in format {category}_{subcategory}_{id}
    - wise: uses id_version since IDs repeat for original/rewrite
    - emu-edit: uses {split}_{task}_{id} for unique identification
    - others: typically just use id

    Args:
        source_name: Name of the source
        prompt_metadata: The prompt_metadata dict from the pair

    Returns:
        The lookup key string
    """
    prompt_id = str(prompt_metadata.get("id", ""))

    if source_name == "oneigbench":
        # OneIG-Bench uses compound key (id_category)
        category = prompt_metadata.get("category", "")
        return f"{prompt_id}_{category}"

    if source_name == "isgbench":
        # ISG-Bench uses compound key (id_category)
        category = prompt_metadata.get("category", "")
        return f"{prompt_id}_{category}"

    if source_name == "chameleon":
        # Chameleon uses compound key (id_category)
        category = prompt_metadata.get("category", "")
        return f"{prompt_id}_{category}"

    if source_name == "interleavedeval":
        # InterleavedEval uses index as lookup key
        return str(prompt_id)

    if source_name == "mmmg":
        # MMMG uses compound key (id_category)
        category = prompt_metadata.get("category", "")
        return f"{prompt_id}_{category}"

    if source_name == "dreambench":
        # DreamBench uses compound key (id_category)
        category = prompt_metadata.get("category", "")
        return f"{prompt_id}_{category}"

    if source_name == "r2ibench":
        # R2I-Bench ID is already in format: {category}_{subcategory}_{id}
        # e.g., "logical_Disjunctive_73"
        return prompt_id

    if source_name == "wise":
        # WISE uses id_version since IDs repeat for original/rewrite versions
        version = prompt_metadata.get("version", "original")
        return f"{prompt_id}_{version}"

    if source_name == "emu-edit":
        # Emu-Edit uses {split}_{task}_{id} for unique identification
        split = prompt_metadata.get("split", "test")
        task = prompt_metadata.get("task", "")
        return f"{split}_{task}_{prompt_id}"

    # Default: just use id
    return prompt_id


def merge_prompts(
    release_path: str,
    output_path: str,
    data_dir: str,
    sources_to_process: List[str] = None,
) -> Dict[str, Any]:
    """
    Merge prompts from downloaded sources into the release JSON.

    Args:
        release_path: Path to release JSON
        output_path: Path to save merged JSON
        data_dir: Directory containing downloaded source data
        sources_to_process: List of sources to process (default: all in release)

    Returns:
        Statistics about the merge
    """
    # Load release data
    print(f"Loading release JSON: {release_path}")
    with open(release_path, "r") as f:
        release_data = json.load(f)

    pairs = release_data["pairs"]
    print(f"Found {len(pairs)} pairs")

    # Identify sources in the release
    sources_in_release = set(p["prompt_source"] for p in pairs)
    print(f"Sources in release: {sources_in_release}")

    if sources_to_process is None:
        sources_to_process = list(sources_in_release)

    # Collect required lookup keys per source for efficient filtering
    required_keys_by_source = {}
    for pair in pairs:
        source = pair["prompt_source"]
        if source not in required_keys_by_source:
            required_keys_by_source[source] = set()
        lookup_key = get_lookup_key(source, pair["prompt_metadata"])
        required_keys_by_source[source].add(lookup_key)

    # Initialize stats
    stats = {
        "total": len(pairs),
        "merged": 0,
        "missing": 0,
        "skipped_source": 0,
        "by_source": {},
    }

    # Group pairs by source for efficient processing
    pairs_by_source = {}
    for i, pair in enumerate(pairs):
        source = pair["prompt_source"]
        if source not in pairs_by_source:
            pairs_by_source[source] = []
        pairs_by_source[source].append((i, pair))

    # Initialize merged_pairs with copies of original pairs
    merged_pairs = [pair.copy() for pair in pairs]

    # Process one source at a time to minimize memory usage
    for source in sources_to_process:
        if source not in SOURCES:
            print(f"Warning: Source '{source}' not implemented yet, skipping")
            continue

        # Initialize source stats
        if source not in stats["by_source"]:
            stats["by_source"][source] = {
                "merged": 0,
                "missing": 0,
                "skipped": 0,
            }

        if source not in pairs_by_source:
            continue

        print(f"Loading prompts for: {source}")
        required_keys = required_keys_by_source.get(source)

        try:
            source_prompts = load_source_prompts(
                source, data_dir, required_keys=required_keys
            )
            print(f"  Loaded {len(source_prompts)} prompts")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            print(f"  Run download first for source: {source}")
            for idx, pair in pairs_by_source[source]:
                stats["skipped_source"] += 1
                stats["by_source"][source]["skipped"] += 1
            continue

        # Merge prompts for this source
        for idx, pair in tqdm(pairs_by_source[source], desc=f"Merging {source}"):
            prompt_metadata = pair["prompt_metadata"]
            lookup_key = get_lookup_key(source, prompt_metadata)
            prompt_data = source_prompts.get(lookup_key)

            if prompt_data is None:
                print(f"Warning: Prompt not found - source={source}, key={lookup_key}")
                stats["missing"] += 1
                stats["by_source"][source]["missing"] += 1
                continue

            # Merge prompt into pair
            merged_pairs[idx]["prompt_content"] = prompt_data["prompt_content"]
            merged_pairs[idx]["prompt_metadata"] = prompt_data["metadata"]
            stats["merged"] += 1
            stats["by_source"][source]["merged"] += 1

        # Release memory for this source before loading the next
        del source_prompts
        gc.collect()
        print(f"  Released memory for {source}")

    # Save merged data
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    merged_data = {"pairs": merged_pairs}
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(f"\nSaved merged JSON to: {output_path}")
    print("Statistics:")
    print(f"  Total pairs: {stats['total']}")
    print(f"  Merged: {stats['merged']}")
    print(f"  Missing: {stats['missing']}")
    print(f"  Skipped (source not processed): {stats['skipped_source']}")
    print("\nBy source:")
    for source, source_stats in stats["by_source"].items():
        print(
            f"  {source}: merged={source_stats['merged']}, missing={source_stats['missing']}, skipped={source_stats['skipped']}"
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark data and merge prompts into release JSON"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to release JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save merged JSON",
    )
    parser.add_argument(
        "--data_dir",
        default="./sources",
        help="Directory to save/load downloaded source data",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Sources to process (default: all in release)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download source data before merging",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if data exists",
    )

    args = parser.parse_args()

    # Check input exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Determine sources to process
    if args.sources:
        sources = args.sources
    else:
        # Read release to find sources
        with open(args.input, "r") as f:
            release_data = json.load(f)
        sources = list(set(p["prompt_source"] for p in release_data["pairs"]))

    # Download if requested
    if args.download:
        print("Downloading source data...")
        for source in sources:
            if source in SOURCES:
                print(f"\nDownloading: {source}")
                try:
                    download_source(source, args.data_dir, force=args.force_download)
                except Exception as e:
                    print(f"  Error downloading {source}: {e}")
            else:
                print(f"Skipping {source}: not implemented yet")

    # Merge
    print("\nMerging prompts...")
    merge_prompts(
        args.input,
        args.output,
        args.data_dir,
        sources_to_process=sources,
    )

    return 0


if __name__ == "__main__":
    exit(main())
