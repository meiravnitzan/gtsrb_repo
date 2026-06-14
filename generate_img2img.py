

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

NEGATIVE_PROMPT = (
    "blurry, cropped, tilted, text, watermark, extra objects, multiple signs, "
    "distorted shape, low contrast, occluded, background clutter, bent sign, cut off sign"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GTSRB images from a JSON generation plan using img2img."
    )
    parser.add_argument("--plan_json", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--default_strength",
        type=float,
        default=0.45,
        help="Fallback img2img strength if not provided in the plan.",
    )
    parser.add_argument(
        "--black_threshold",
        type=float,
        default=3.0,
        help="Reject image if mean pixel value is below this threshold (0-255 scale).",
    )
    parser.add_argument(
        "--std_threshold",
        type=float,
        default=3.0,
        help="Reject image if pixel stddev is below this threshold.",
    )
    return parser.parse_args()


def build_pipeline(model_id: str, device: str) -> StableDiffusionImg2ImgPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()

    pipe.set_progress_bar_config(disable=False)
    return pipe


def load_plan(plan_json: str):
    with open(plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    if not isinstance(plan, list):
        raise ValueError("plan_json must contain a list of generation specs")

    required = {"class_id", "class_name", "num_images", "prompts", "source_dir"}
    for item in plan:
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Plan item missing required keys: {sorted(missing)}")
        if not isinstance(item["prompts"], list) or len(item["prompts"]) == 0:
            raise ValueError(f"class_id={item['class_id']} must have at least one prompt")

    return plan


def is_invalid_image(image, black_threshold: float, std_threshold: float) -> bool:
    arr = np.asarray(image).astype(np.float32)
    mean_val = float(arr.mean())
    std_val = float(arr.std())
    return mean_val < black_threshold or std_val < std_threshold


def list_source_images(source_dir: Path) -> List[Path]:
    exts = {".ppm", ".png", ".jpg", ".jpeg"}
    files = sorted([p for p in source_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise ValueError(f"No source images found in {source_dir}")
    return files


def prepare_init_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan_json)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_csv)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = build_pipeline(args.model_id, device)

    rows = []

    for item in plan:
        class_id = int(item["class_id"])
        class_name = str(item["class_name"])
        target_num_images = int(item["num_images"])
        prompts = list(item["prompts"])
        source_dir = Path(item["source_dir"])
        strength = float(item.get("strength", args.default_strength))

        if not source_dir.exists():
            raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

        source_images = list_source_images(source_dir)

        class_dir = output_root / f"{class_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        attempted = 0

        print(f"\nGenerating class {class_id} ({class_name}) into {class_dir}")
        print(f"Using {len(source_images)} source images from {source_dir}")
        print(f"Strength={strength}")

        while saved < target_num_images:
            current_batch = min(args.batch_size, target_num_images - saved)

            batch_prompts = [
                prompts[(attempted + i) % len(prompts)]
                for i in range(current_batch)
            ]

            batch_source_paths = [
                source_images[(attempted + i) % len(source_images)]
                for i in range(current_batch)
            ]

            batch_init_images = [
                prepare_init_image(src, args.width, args.height)
                for src in batch_source_paths
            ]

            generators = [
                torch.Generator(device=device).manual_seed(
                    args.seed + class_id * 10000 + attempted + i
                )
                for i in range(current_batch)
            ]

            result = pipe(
                prompt=batch_prompts,
                image=batch_init_images,
                negative_prompt=[NEGATIVE_PROMPT] * current_batch,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                strength=strength,
                generator=generators,
            )

            for i, image in enumerate(result.images):
                prompt_used = batch_prompts[i]
                source_used = batch_source_paths[i]
                seed_used = args.seed + class_id * 10000 + attempted + i

                if is_invalid_image(
                    image,
                    black_threshold=args.black_threshold,
                    std_threshold=args.std_threshold,
                ):
                    print(
                        f" skipped invalid image for class {class_id} "
                        f"(seed={seed_used}, source='{source_used.name}')"
                    )
                    continue

                out_name = f"{class_id:05d}_synth_{saved:03d}.png"
                out_path = class_dir / out_name
                image.save(out_path)

                rows.append(
                    {
                        "image_path": str(out_path),
                        "label": class_id,
                        "class_name": class_name,
                        "prompt": prompt_used,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "seed": seed_used,
                        "width": args.width,
                        "height": args.height,
                        "model_id": args.model_id,
                        "source_image": str(source_used),
                        "strength": strength,
                    }
                )

                saved += 1
                print(f" saved {saved}/{target_num_images}")

                if saved >= target_num_images:
                    break

            attempted += current_batch

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label",
                "class_name",
                "prompt",
                "negative_prompt",
                "seed",
                "width",
                "height",
                "model_id",
                "source_image",
                "strength",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
