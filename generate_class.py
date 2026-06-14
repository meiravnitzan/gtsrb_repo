
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


NEGATIVE_PROMPT = (
    "blurry, cropped, tilted, text, watermark, extra objects, multiple signs, "
    "distorted shape, low contrast, occluded, background clutter, bent sign, cut off sign"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic GTSRB images from a JSON generation plan."
    )
    parser.add_argument(
        "--plan_json",
        type=str,
        required=True,
        help="Path to a JSON file describing target classes, prompts, and counts.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory where class subfolders will be created.",
    )
    parser.add_argument(
        "--manifest_csv",
        type=str,
        required=True,
        help="Combined CSV manifest for all generated images.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def build_pipeline(model_id: str, device: str) -> StableDiffusionPipeline:
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
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

    required = {"class_id", "class_name", "num_images", "prompts"}
    for item in plan:
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Plan item missing required keys: {sorted(missing)}")
        if not isinstance(item["prompts"], list) or len(item["prompts"]) == 0:
            raise ValueError(f"class_id={item['class_id']} must have at least one prompt")

    return plan


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
        num_images = int(item["num_images"])
        prompts = list(item["prompts"])

        class_dir = output_root / f"{class_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        print(f"\nGenerating class {class_id} ({class_name}) into {class_dir}")

        while saved < num_images:
            current_batch = min(args.batch_size, num_images - saved)
            batch_prompts = [
                prompts[(saved + i) % len(prompts)]
                for i in range(current_batch)
            ]

            generators = [
                torch.Generator(device=device).manual_seed(
                    args.seed + class_id * 10000 + saved + i
                )
                for i in range(current_batch)
            ]

            result = pipe(
                batch_prompts,
                negative_prompt=[NEGATIVE_PROMPT] * current_batch,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generators,
            )

            for i, image in enumerate(result.images):
                idx = saved + i
                out_name = f"{class_id:05d}_synth_{idx:03d}.png"
                out_path = class_dir / out_name
                image.save(out_path)

                rows.append(
                    {
                        "image_path": str(out_path),
                        "label": class_id,
                        "class_name": class_name,
                        "prompt": batch_prompts[i],
                        "negative_prompt": NEGATIVE_PROMPT,
                        "seed": args.seed + class_id * 10000 + idx,
                        "width": args.width,
                        "height": args.height,
                        "model_id": args.model_id,
                    }
                )

            saved += current_batch
            print(f"  saved {saved}/{num_images}")

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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
