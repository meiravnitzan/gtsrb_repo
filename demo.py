# Colab demo: baseline vs augmented checkpoint on one image

from pathlib import Path
import sys
import csv
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import yaml

# --- Config ---
CONFIG = Path("/content/gtsrb_repo/config.yaml")
cfg = yaml.safe_load(CONFIG.read_text())

REPO = Path(cfg["workspace_root"])
BASELINE_CKPT = Path(cfg["baseline_checkpoint"])
AUGMENTED_CKPT = Path(cfg["gen22_checkpoint"])
TRAIN_ROOT = Path(cfg["train_root"])

assert CONFIG.exists(), f"Missing config file: {CONFIG}"
assert BASELINE_CKPT.exists(), f"Missing baseline checkpoint: {BASELINE_CKPT}"
assert AUGMENTED_CKPT.exists(), f"Missing augmented checkpoint: {AUGMENTED_CKPT}"

# --- Import project model builder ---
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model import build_model

device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# --- Labels ---
CLASS_NAMES = {
    0: "speed_limit_20", 1: "speed_limit_30", 2: "speed_limit_50", 3: "speed_limit_60",
    4: "speed_limit_70", 5: "speed_limit_80", 6: "end_speed_limit_80", 7: "speed_limit_100",
    8: "speed_limit_120", 9: "no_passing", 10: "no_passing_trucks", 11: "priority_at_next_intersection",
    12: "priority_road", 13: "yield", 14: "stop", 15: "no_vehicles",
    16: "trucks_prohibited", 17: "no_entry", 18: "general_caution", 19: "dangerous_curve_left",
    20: "dangerous_curve_right", 21: "double_curve", 22: "bumpy_road", 23: "slippery_road",
    24: "road_narrows_right", 25: "road_work", 26: "traffic_signals", 27: "pedestrians",
    28: "children_crossing", 29: "bicycles_crossing", 30: "beware_ice_snow", 31: "wild_animals_crossing",
    32: "end_speed_passing_limits", 33: "turn_right_ahead", 34: "turn_left_ahead", 35: "ahead_only",
    36: "go_straight_or_right", 37: "go_straight_or_left", 38: "keep_right", 39: "keep_left",
    40: "roundabout_mandatory", 41: "end_no_passing", 42: "end_no_passing_trucks"
}

# --- Preprocess ---
img_size = int(cfg["image_size"])
tfm = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model(ckpt_path):
    model = build_model(cfg["model_name"], cfg["num_classes"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    return model

baseline_model = load_model(BASELINE_CKPT)
augmented_model = load_model(AUGMENTED_CKPT)

@torch.no_grad()
def predict(model, image_path):
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    top3 = np.argsort(probs)[::-1][:3]
    return image, pred, conf, [(int(i), float(probs[i])) for i in top3]

def render_demo(image_path, true_label=None):
    image_path = Path(image_path)
    image, b_pred, b_conf, b_top3 = predict(baseline_model, image_path)
    _, a_pred, a_conf, a_top3 = predict(augmented_model, image_path)

    fixed = (true_label == 22 and b_pred != 22 and a_pred == 22) if true_label is not None else False

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.axis("off")
    title = f"Input image\n{image_path.name}"
    if true_label is not None:
        title += f"\nTrue label: {true_label} ({CLASS_NAMES.get(true_label, 'unknown')})"
    ax0.set_title(title, fontsize=12)

    def panel(ax, model_name, pred, conf, top3, color):
        ax.axis("off")
        lines = [
            model_name,
            "",
            f"Top-1: {pred} ({CLASS_NAMES.get(pred, 'unknown')})",
            f"Confidence: {conf:.3f}",
            "",
            "Top-3:"
        ]
        for k, (cls, p) in enumerate(top3, start=1):
            lines.append(f"{k}. {cls} - {CLASS_NAMES.get(cls, 'unknown')} ({p:.3f})")

        ax.text(
            0.02, 0.98, "\n".join(lines),
            va="top", ha="left", fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor=color,
                alpha=0.15,
                edgecolor=color,
                linewidth=2
            )
        )

    ax1 = fig.add_subplot(gs[0, 1])
    panel(ax1, "Baseline model", b_pred, b_conf, b_top3, "#d62728")

    ax2 = fig.add_subplot(gs[0, 2])
    panel(ax2, "Augmented model", a_pred, a_conf, a_top3, "#2ca02c")

    if fixed:
        fig.suptitle(
            "Class 22 recovered by synthetic augmentation",
            fontsize=18, color="#2ca02c", weight="bold"
        )
    elif true_label == 22:
        fig.suptitle("Class 22 comparison", fontsize=18)
    else:
        fig.suptitle("Baseline vs augmented comparison", fontsize=18)

    plt.tight_layout()
    plt.show()

def scan_target_class_examples(limit=30, class_id=None):
    if class_id is None:
        class_id = int(cfg["target_class_ids"][0])

    class_dir = TRAIN_ROOT / f"{class_id:05d}"
    files = sorted(
        [p for p in class_dir.iterdir() if p.suffix.lower() in [".ppm", ".png", ".jpg", ".jpeg"]]
    )[:limit]

    rows = []
    for p in files:
        _, b_pred, b_conf, _ = predict(baseline_model, p)
        _, a_pred, a_conf, _ = predict(augmented_model, p)
        rows.append({
            "file": p.name,
            "baseline_pred": b_pred,
            "baseline_conf": round(b_conf, 3),
            "augmented_pred": a_pred,
            "augmented_conf": round(a_conf, 3),
            "fixed_target": (b_pred != class_id and a_pred == class_id),
        })
    return rows

def show_one_model(model, image_path, model_name="Model", true_label=None):
    image_path = Path(image_path)
    image, pred, conf, top3 = predict(model, image_path)

    pred_dir = TRAIN_ROOT / f"{pred:05d}"
    pred_example = None
    if pred_dir.exists():
        pred_files = sorted(
            [p for p in pred_dir.iterdir() if p.suffix.lower() in [".ppm", ".png", ".jpg", ".jpeg"]]
        )
        if pred_files:
            pred_example = pred_files[0]

    fig = plt.figure(figsize=(15, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image)
    ax0.axis("off")
    title = f"Input image\n{image_path.name}"
    if true_label is not None:
        title += f"\nTrue label: {true_label} ({CLASS_NAMES.get(true_label, 'unknown')})"
    ax0.set_title(title, fontsize=12)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    lines = [
        model_name,
        "",
        f"Top-1: {pred} ({CLASS_NAMES.get(pred, 'unknown')})",
        f"Confidence: {conf:.3f}",
        "",
        "Top-3:"
    ]
    for k, (cls, p) in enumerate(top3, start=1):
        lines.append(f"{k}. {cls} - {CLASS_NAMES.get(cls, 'unknown')} ({p:.3f})")

    ax1.text(
        0.02, 0.98,
        "\n".join(lines),
        va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f3f4f6", edgecolor="#9ca3af")
    )

    ax2 = fig.add_subplot(gs[0, 2])
    if pred_example is not None:
        pred_img = Image.open(pred_example).convert("RGB")
        ax2.imshow(pred_img)
        ax2.axis("off")
        ax2.set_title(
            f"Predicted model example\nClass {pred} ({CLASS_NAMES.get(pred, 'unknown')})\n{pred_example.name}",
            fontsize=11
        )
    else:
        ax2.axis("off")
        ax2.text(
            0.5, 0.5,
            f"Predicted model example\nNo example found for class {pred}",
            ha="center", va="center", fontsize=12
        )

    plt.tight_layout()
    plt.show()

def list_misclassified_images_for_class(
    class_id,
    csv_path=cfg["baseline_predictions_csv"],
    predicted_as=None,
    limit=None,
):
    """
    Return misclassified images for a specific true class from a predictions CSV.
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_label = int(row["true_label"])
            pred_label = int(row["pred_label"])
            correct = int(row["correct"])

            if true_label != class_id:
                continue
            if correct == 1:
                continue
            if predicted_as is not None and pred_label != predicted_as:
                continue

            rows.append({
                "image_path": row["image_path"],
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": float(row["confidence"]),
            })

            if limit is not None and len(rows) >= limit:
                break

    return rows

def show_misclassified_examples(
    class_id,
    csv_path=cfg["baseline_predictions_csv"],
    predicted_as=None,
    max_show=5,
):
    examples = list_misclassified_images_for_class(
        class_id=class_id,
        csv_path=csv_path,
        predicted_as=predicted_as,
        limit=max_show,
    )

    if not examples:
        print("No matching misclassified images found.")
        return examples

    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 4))
    if len(examples) == 1:
        axes = [axes]

    for ax, ex in zip(axes, examples):
        img = Image.open(ex["image_path"]).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"true={ex['true_label']}\npred={ex['pred_label']}\nconf={ex['confidence']:.3f}",
            fontsize=10
        )

    plt.tight_layout()
    plt.show()
    return examples

def show_generated_images(class_id, from_idx, to_idx):
    """
    Show generated augmented images for one class in the inclusive index range [from_idx, to_idx].

    Example:
        show_generated_images(22, 0, 7)
    """
    generated_root = Path(cfg["generated_root"])
    class_dir = generated_root / f"{class_id:05d}"

    image_files = sorted(
        [p for p in class_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".ppm"]]
    )

    selected = image_files[from_idx:to_idx + 1]
    n = len(selected)

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    axes = axes.flatten()

    for ax, img_path in zip(axes, selected):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=10)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Generated images for class {class_id} ({CLASS_NAMES.get(class_id, 'unknown')})\n"
        f"Showing indices {from_idx} to {to_idx}",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()
