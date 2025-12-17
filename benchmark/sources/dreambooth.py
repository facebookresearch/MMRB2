"""
DreamBooth source module.

GitHub: https://github.com/google/dreambooth/tree/main/dataset
Paper: DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

Data structure:
- dataset/{subject}/*.jpg  (images for each subject)

Subjects include live subjects (cat, dog variants) and objects (backpack, teapot, etc.)
"""

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict

from tqdm import tqdm

SOURCE_NAME = "dreambooth"
GITHUB_REPO = "https://github.com/google/dreambooth.git"

# Live subjects and their class names
LIVESUBJECT2CLASS = {
    "cat": "cat",
    "cat2": "cat",
    "dog": "dog",
    "dog2": "dog",
    "dog3": "dog",
    "dog5": "dog",
    "dog6": "dog",
    "dog7": "dog",
    "dog8": "dog",
}

# Objects and their class names
OBJECT2CLASS = {
    "backpack": "backpack",
    "backpack_dog": "backpack",
    "bear_plushie": "stuffed animal",
    "berry_bowl": "bowl",
    "can": "can",
    "candle": "candle",
    "clock": "clock",
    "colorful_sneaker": "sneaker",
    "duck_toy": "toy",
    "fancy_boot": "boot",
    "grey_sloth_plushie": "stuffed animal",
    "monster_toy": "toy",
    "pink_sunglasses": "glasses",
    "poop_emoji": "toy",
    "rc_car": "toy",
    "red_cartoon": "cartoon",
    "robot_toy": "toy",
    "shiny_sneaker": "sneaker",
    "teapot": "teapot",
    "vase": "vase",
    "wolf_plushie": "stuffed animal",
}

# All subjects combined
ALL_SUBJECTS = {**LIVESUBJECT2CLASS, **OBJECT2CLASS}

# Prompts for objects
OBJECT_PROMPTS = [
    "a {} in the jungle",
    "a {} in the snow",
    "a {} on the beach",
    "a {} on a cobblestone street",
    "a {} on top of pink fabric",
    "a {} on top of a wooden floor",
    "a {} with a city in the background",
    "a {} with a mountain in the background",
    "a {} with a blue house in the background",
    "a {} on top of a purple rug in a forest",
    "a {} with a wheat field in the background",
    "a {} with a tree and autumn leaves in the background",
    "a {} with the Eiffel Tower in the background",
    "a {} floating on top of water",
    "a {} floating in an ocean of milk",
    "a {} on top of green grass with sunflowers around it",
    "a {} on top of a mirror",
    "a {} on top of the sidewalk in a crowded street",
    "a {} on top of a dirt road",
    "a {} on top of a white rug",
    "a red {}",
    "a purple {}",
    "a shiny {}",
    "a wet {}",
    "a cube shaped {}",
]

# Prompts for live subjects
LIVE_SUBJECT_PROMPTS = [
    "a {} in the jungle",
    "a {} in the snow",
    "a {} on the beach",
    "a {} on a cobblestone street",
    "a {} on top of pink fabric",
    "a {} on top of a wooden floor",
    "a {} with a city in the background",
    "a {} with a mountain in the background",
    "a {} with a blue house in the background",
    "a {} on top of a purple rug in a forest",
    "a {} wearing a red hat",
    "a {} wearing a santa hat",
    "a {} wearing a rainbow scarf",
    "a {} wearing a black top hat and a monocle",
    "a {} in a chef outfit",
    "a {} in a firefighter outfit",
    "a {} in a police outfit",
    "a {} wearing pink glasses",
    "a {} wearing a yellow shirt",
    "a {} in a purple wizard outfit",
    "a red {}",
    "a purple {}",
    "a shiny {}",
    "a wet {}",
    "a cube shaped {}",
]


def make_lookup_key(subject: str, image_name: str, prompt_idx: int) -> str:
    """Create lookup key from subject, image name, and prompt index."""
    return f"{subject}_{image_name}_{prompt_idx}"


def download(data_dir: str, force: bool = False) -> str:
    """
    Download DreamBooth data from GitHub.

    Args:
        data_dir: Base directory for downloaded data
        force: If True, re-download even if data exists

    Returns:
        Path to the source data directory
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    images_dir = os.path.join(source_dir, "images")

    if os.path.exists(images_dir) and not force:
        # Check if we have at least some subjects
        existing_subjects = os.listdir(images_dir) if os.path.isdir(images_dir) else []
        if len(existing_subjects) >= 10:
            print(f"[{SOURCE_NAME}] Already downloaded: {source_dir}")
            return source_dir

    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    print(f"[{SOURCE_NAME}] Downloading from GitHub: {GITHUB_REPO}")

    # Clone with sparse checkout to only get the dataset folder
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_dir = os.path.join(tmp_dir, "dreambooth")

        # Initialize sparse checkout
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", GITHUB_REPO, repo_dir],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "dataset"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        # Copy dataset to our images directory
        dataset_dir = os.path.join(repo_dir, "dataset")
        if os.path.exists(dataset_dir):
            for subject in tqdm(os.listdir(dataset_dir), desc=f"Copying {SOURCE_NAME}"):
                subject_src = os.path.join(dataset_dir, subject)
                subject_dst = os.path.join(images_dir, subject)
                if os.path.isdir(subject_src):
                    if os.path.exists(subject_dst):
                        shutil.rmtree(subject_dst)
                    shutil.copytree(subject_src, subject_dst)

    print(f"[{SOURCE_NAME}] Downloaded to {source_dir}")
    return source_dir


def load_prompts(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from DreamBooth.

    Generates prompts for each subject image using predefined prompt templates.

    Args:
        data_dir: Base directory containing downloaded data

    Returns:
        Dictionary mapping lookup keys to prompt data
    """
    source_dir = os.path.join(data_dir, SOURCE_NAME)
    images_dir = os.path.join(source_dir, "images")
    output_images_dir = os.path.join(source_dir, "output_images")
    os.makedirs(output_images_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(
            f"DreamBooth data not found at {images_dir}. Run download first."
        )

    prompts = {}

    for subject in tqdm(os.listdir(images_dir), desc=f"Loading {SOURCE_NAME}"):
        subject_dir = os.path.join(images_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        # Determine if this is a live subject or object
        if subject in LIVESUBJECT2CLASS:
            class_name = LIVESUBJECT2CLASS[subject]
            prompt_templates = LIVE_SUBJECT_PROMPTS
            subject_type = "live_subject"
        elif subject in OBJECT2CLASS:
            class_name = OBJECT2CLASS[subject]
            prompt_templates = OBJECT_PROMPTS
            subject_type = "object"
        else:
            # Unknown subject, skip
            continue

        # Process each image in the subject folder
        for img_file in os.listdir(subject_dir):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(subject_dir, img_file)
            img_name = os.path.splitext(img_file)[0]

            # Copy image to output directory with unique name
            output_img_name = f"{subject}_{img_file}"
            output_img_path = os.path.join(output_images_dir, output_img_name)
            rel_img_path = os.path.join(SOURCE_NAME, "output_images", output_img_name)

            if not os.path.exists(output_img_path):
                shutil.copy(img_path, output_img_path)

            # Generate prompts for this image
            for prompt_idx, prompt_template in enumerate(prompt_templates):
                prompt_text = prompt_template.format(class_name)

                lookup_key = make_lookup_key(subject, img_name, prompt_idx)

                prompts[lookup_key] = {
                    "id": hash(lookup_key) % (10**9),
                    "lookup_key": lookup_key,
                    "prompt_content": [
                        ["image", rel_img_path],
                        ["text", prompt_text],
                    ],
                    "metadata": {
                        "subject": subject,
                        "class_name": class_name,
                        "subject_type": subject_type,
                        "image_file": img_file,
                        "prompt_idx": prompt_idx,
                        "prompt_template": prompt_template,
                    },
                }

    print(f"[{SOURCE_NAME}] Loaded {len(prompts)} prompts")
    return prompts

