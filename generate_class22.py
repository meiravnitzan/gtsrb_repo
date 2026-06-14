from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt


PROMPTS = [
    "a centered German road work traffic sign, red triangle border, white background, road work icon, clean dataset style, front facing, high contrast",
    "a German warning road work sign, triangular red border, construction worker icon, isolated sign, front view, realistic but clean training image",
    "close-up of a German road work traffic sign, centered, no background clutter, traffic sign dataset style, sharp edges, front facing",
    "a single German road work warning sign on plain background, red triangular border, black construction symbol, centered composition",
]

NEGATIVE_PROMPT = (
    "blurry, cropped, tilted, text, watermark, extra objects, multiple signs, "
    "distorted shape, low contrast, occluded, background clutter"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate optional synthetic class 22 images for GTSRB augmentation "
            "using Stable Diffusion."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/generated/class22",
        help="Directory where generated images will be saved.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="Total number of synthetic images to generate.",
    )
    pa