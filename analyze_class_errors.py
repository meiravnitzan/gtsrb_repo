import os
import argparse
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm")


def list_images(folder):
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(IMG_EXTS):
            files.append(path)
    return files


def load_image(path):
    return Image.open(path).convert("RGB")


def safe_title(path, max_len=28):
    base = os.path.basename(path)
    return base if len(base) <= max_len else base[: max_len - 3] + "..."


def resolve_class_dir(data_dir, pred_idx, pred_class=None):
    candidates = []

    if pred_class is not None:
        candidates.append(os.path.join(data_dir, str(pred_class)))

    candidates.append(os.path.join(data_dir, str(pred_idx)))
    candidates.append(os.path.join(data_dir, f"{int(pred_idx):05d}"))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return None


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    required_cols = {"image_path", "true_label", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    target_errors = df[
        (df["true_label"] == args.class_idx) & (df["pred_label"] != args.class_idx)
    ].copy()

    if target_errors.empty:
        print(f"No misclassified samples found for class {args.class_idx}.")
        return

    if "pred_class" not in target_errors.columns:
        target_errors["pred_class"] = target_errors["pred_label"].astype(str)

    grouped = target_errors.groupby(["pred_label", "pred_class"])

    for (pred_idx, pred_class), group in grouped:
        misclassified_paths = group["image_path"].tolist()

        pred_class_dir = resolve_class_dir(args.data_dir, pred_idx, pred_class)
        if pred_class_dir is None:
            print(
                f"Skipping predicted class '{pred_class}' (label {pred_idx}) "
                f"because no matching folder was found under: {args.data_dir}"
            )
            continue

        misclassified_abs = {os.path.abspath(x) for x in misclassified_paths}
        candidate_paths = [
            p for p in list_images(pred_class_dir)
            if os.path.abspath(p) not in misclassified_abs
        ]

        rng = random.Random(args.seed)
        example_paths = candidate_paths[:]
        rng.shuffle(example_paths)
        example_paths = example_paths[: args.examples_per_class]

        n_top = len(misclassified_paths)
        n_cols = max(n_top, args.examples_per_class)

        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
        if n_cols == 1:
            axes = [[axes[0]], [axes[1]]]

        fig.suptitle(
            f"True class {args.class_idx} misclassified as {pred_class} (label {pred_idx})",
            fontsize=16,
        )

        for col in range(n_cols):
            ax_top = axes[0][col]
            ax_bot = axes[1][col]
            ax_top.axis("off")
            ax_bot.axis("off")

            if col < n_top:
                img = load_image(misclassified_paths[col])
                ax_top.imshow(img)
                ax_top.set_title(
                    f"Misclassified class {args.class_idx}\n"
                    f"{safe_title(misclassified_paths[col])}",
                    fontsize=10,
                )

            if col < len(example_paths):
                img = load_image(example_paths[col])
                ax_bot.imshow(img)
                ax_bot.set_title(
                    f"Example of {pred_class}\n"
                    f"{safe_title(example_paths[col])}",
                    fontsize=10,
                )

        axes[0][0].set_ylabel(f"Class {args.class_idx} errors", fontsize=12)
        axes[1][0].set_ylabel(f"Examples of {pred_class}", fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        out_name = (
            f"class{args.class_idx}_as_{pred_idx}_{str(pred_class).replace(' ', '_')}.png"
        )
        out_path = os.path.join(args.output_dir, out_name)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path", type=str, required=True, help="CSV from evaluate.py"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset root organized by class folders, e.g. .../Final_Test/Images or .../Final_Training/Images",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        required=True,
        help="Target true class to inspect",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/class_report",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=3,
        help="Number of reference examples to show for each predicted class",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
